from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from commands.responders import DOMAIN_BIOMARKERS
from data import load_composites

# Okabe-Ito colors for three classifiers and permutation null
_CLF_COLORS = {
    'Random Forest':       '#0072B2',
    'Gradient Boosting':   '#009E73',
    'Logistic Regression': '#D55E00',
}
_COLOR_NULL = '#56B4E9'

_OUT_DIR = Path('output/plots_responders_ml')
_N_PERM  = 200


def _slug(names: list[str]) -> str:
    """'Biological Age', 'NAD+' → 'biological_age_nad'"""
    import re
    return '_'.join(re.sub(r'[^a-z0-9]+', '_', n.lower()).strip('_') for n in names)


def responders_ml(only: tuple[str, ...] = ()) -> None:
    """ML classification of Treatment vs Control using biomarker Δ values → stdout + plots."""
    df = load_composites()
    biomarkers = [bm for bm in DOMAIN_BIOMARKERS if not only or bm.name in only]
    if not biomarkers:
        raise ValueError(f'No biomarkers matched --only filter: {list(only)}')
    delta_cols = [bm.delta_col for bm in biomarkers]
    names      = [bm.name      for bm in biomarkers]
    out_dir    = _OUT_DIR / _slug(names) if only else _OUT_DIR

    # Drop rows with any missing Δ across the 17 features
    X_raw = df[delta_cols].dropna()
    y     = df.loc[X_raw.index, 'Treatment'].to_numpy()
    X     = X_raw.to_numpy()

    print(f'ML Classification: Treatment vs Control ({len(names)} biomarker Δ values)')
    print('=' * 65)
    if only:
        print(f'  Filter: {", ".join(names)}')
    print(f'  n = {len(y)}  (Treatment: {int(y.sum())}, Control: {int((y == 0).sum())})')
    print(f'  Features: {len(names)} domain-relevant biomarker Δ values')
    print(f'  Cross-validation: stratified 5-fold, metric: AUC-ROC')
    print(f'  Permutation test: {_N_PERM} permutations (shuffled labels, same CV)')
    print()
    print(f"{'Classifier':<22} {'Mean AUC':>10} {'SD':>6}  Perm p  Sig.")
    print('-' * 55)

    classifiers: dict[str, Pipeline] = {
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=500, random_state=42,
                                           class_weight='balanced')),
        ]),
        'Gradient Boosting': Pipeline([
            ('scaler', StandardScaler()),
            # GradientBoostingClassifier has no class_weight; imbalance noted in thesis
            ('clf', GradientBoostingClassifier(n_estimators=200, random_state=42)),
        ]),
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42,
                                       class_weight='balanced')),
        ]),
    }

    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rng = np.random.default_rng(42)

    cv_aucs:  dict[str, list[float]]           = {}
    all_fpr:  dict[str, list[np.ndarray]]      = {}
    all_tpr:  dict[str, list[np.ndarray]]      = {}
    rf_importances: np.ndarray | None          = None

    for clf_name, pipe in classifiers.items():
        fold_aucs: list[float]       = []
        fold_fpr:  list[np.ndarray]  = []
        fold_tpr:  list[np.ndarray]  = []

        for train_idx, test_idx in cv.split(X, y):
            pipe.fit(X[train_idx], y[train_idx])
            y_prob = pipe.predict_proba(X[test_idx])[:, 1]
            fold_aucs.append(float(roc_auc_score(y[test_idx], y_prob)))
            fpr, tpr, _ = roc_curve(y[test_idx], y_prob)
            fold_fpr.append(fpr)
            fold_tpr.append(tpr)

        cv_aucs[clf_name] = fold_aucs
        all_fpr[clf_name] = fold_fpr
        all_tpr[clf_name] = fold_tpr
        mean_obs = float(np.mean(fold_aucs))

        # Permutation test: shuffle labels, re-run CV, compare mean AUC
        null_aucs: list[float] = []
        for _ in range(_N_PERM):
            y_perm = rng.permutation(y)
            perm_fold: list[float] = []
            for train_idx, test_idx in cv.split(X, y_perm):
                pipe.fit(X[train_idx], y_perm[train_idx])
                y_prob_p = pipe.predict_proba(X[test_idx])[:, 1]
                perm_fold.append(float(roc_auc_score(y_perm[test_idx], y_prob_p)))
            null_aucs.append(float(np.mean(perm_fold)))

        # +1/(N+1) continuity correction avoids p = 0
        perm_p = float((np.sum(np.array(null_aucs) >= mean_obs) + 1) / (_N_PERM + 1))
        sig    = '*' if perm_p < 0.05 else ''
        print(f'{clf_name:<22} {mean_obs:>10.3f} {np.std(fold_aucs):>6.3f}  {perm_p:.3f}   {sig}')

        # Refit Random Forest on all data to extract stable feature importances
        if clf_name == 'Random Forest':
            pipe.fit(X, y)
            rf_importances = pipe.named_steps['clf'].feature_importances_

    print('* p < 0.05 (permutation test)')
    print(f'\nNote: with n={len(y)} and severe class imbalance (79:20), AUC estimates')
    print('  have wide variance; interpret with caution.')

    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_roc_curves(all_fpr, all_tpr, cv_aucs, out_dir, len(names))
    if rf_importances is not None:
        _plot_feature_importance(rf_importances, names, out_dir)
    print(f'\nPlots saved to {out_dir}/')


def _plot_roc_curves(
    all_fpr: dict[str, list[np.ndarray]],
    all_tpr: dict[str, list[np.ndarray]],
    cv_aucs: dict[str, list[float]],
    out_dir: Path,
    n_features: int,
) -> None:
    """Mean ROC curve across CV folds for each classifier, with AUC ± SD in legend."""
    fig, ax = plt.subplots(figsize=(6, 6))
    mean_fpr = np.linspace(0, 1, 100)

    for clf_name, fpr_list in all_fpr.items():
        tpr_list = all_tpr[clf_name]
        # np.interp: linearly interpolate each fold's TPR onto the shared FPR grid
        interp_tprs = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fpr_list, tpr_list)]
        mean_tpr = np.mean(interp_tprs, axis=0)
        mean_auc = float(np.mean(cv_aucs[clf_name]))
        std_auc  = float(np.std(cv_aucs[clf_name]))
        ax.plot(mean_fpr, mean_tpr, color=_CLF_COLORS[clf_name], linewidth=2,
                label=f'{clf_name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Chance')
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.set_title(f'ROC Curves – Treatment vs Control Classification\n(Stratified 5-Fold CV, {n_features} Biomarker Δ Features)',
                 fontsize=10)
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(out_dir / 'roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_dir}/roc_curves.png')


def _plot_feature_importance(importances: np.ndarray, names: list[str], out_dir: Path) -> None:
    """Horizontal bar chart of Random Forest feature importances, sorted ascending for readability."""
    idx = np.argsort(importances)   # ascending → bottom-to-top from least to most important
    sorted_names = [names[i] for i in idx]
    sorted_imp   = importances[idx]

    fig, ax = plt.subplots(figsize=(7, 6))
    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, sorted_imp * 100, color='#0072B2', alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.set_xlabel('Feature importance (%)', fontsize=10)
    ax.set_title(f'Random Forest Feature Importances\n(Trained on all participants, {len(names)} Biomarker Δ Features)',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_dir}/feature_importance.png')
