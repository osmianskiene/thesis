import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import levene, mannwhitneyu, shapiro, ttest_ind
from statsmodels.stats.multitest import multipletests

from biomarkers import AGE_ACCEL_BIOMARKERS, ANALYSIS_BIOMARKERS, LOG_BIOMARKERS
from data import load_composites


def _is_normal(s: pd.Series, alpha: float = 0.05) -> bool:
    s = s.dropna()
    if len(s) < 3:
        return False
    _, p = shapiro(s.sample(min(50, len(s)), random_state=42))
    return p > alpha


def _two_sample_test(a: pd.Series, b: pd.Series, alpha: float = 0.05) -> tuple[float, str]:
    """Choose and run the appropriate two-sample test.

    Both normal + equal variances   → Student t-test
    Both normal + unequal variances → Welch t-test
    Otherwise                       → Mann-Whitney U
    Returns (p_value, test_name).
    """
    both_normal = _is_normal(a) and _is_normal(b)
    if both_normal:
        _, p_lev = levene(a, b)
        equal_var = p_lev > alpha
        _, p_val = ttest_ind(a, b, equal_var=equal_var)
        test_name = "Student t" if equal_var else "Welch t"
    else:
        _, p_val = mannwhitneyu(a, b, alternative='two-sided')
        test_name = "Mann-Whitney U"
    return float(p_val), test_name


def _cohens_d(a: pd.Series, b: pd.Series) -> float:
    """Cohen's d using pooled standard deviation (Δ treatment − Δ control)."""
    na, nb = len(a), len(b)
    if na + nb <= 2:
        return float('nan')
    # Pooled SD: sqrt( ((n_a-1)*s_a² + (n_b-1)*s_b²) / (n_a + n_b - 2) )
    pooled_sd = np.sqrt(
        ((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2)
        / (na + nb - 2)
    )
    if pooled_sd == 0:
        return float('nan')
    return float((a.mean() - b.mean()) / pooled_sd)


def _effect_label(d: float) -> str:
    """Verbal label for |Cohen's d|: negligible / small / medium / large."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return 'negligible'
    if d_abs < 0.5:
        return 'small'
    if d_abs < 0.8:
        return 'medium'
    return 'large'


def _cluster_representatives(df: pd.DataFrame, delta_cols: list[str]) -> dict[str, str]:
    """Return a mapping col → representative_col using hierarchical clustering on Δ columns.

    Spearman pairwise correlation → distance = 1 − r²,
    average-linkage clustering, cut at 0.70 dissimilarity
    (r² > 0.30 → same cluster).
    Within each cluster the member with the highest mean |r| to others is the representative.
    """
    corr = df[delta_cols].corr(method='spearman')
    dist_arr = (1 - corr.values ** 2).clip(0)   # 1 − r², guarantees non-negative
    np.fill_diagonal(dist_arr, 0.0)
    dist_condensed = squareform(dist_arr, checks=False)
    Z = linkage(dist_condensed, method='average')
    cluster_ids = fcluster(Z, t=0.70, criterion='distance')

    clust_df = pd.DataFrame({'col': delta_cols, 'cluster': cluster_ids})
    cluster_info: dict[str, str] = {}   # col → representative_col

    for cid in sorted(clust_df['cluster'].unique()):
        members = clust_df[clust_df['cluster'] == cid]['col'].tolist()
        if len(members) == 1:
            rep = members[0]
        else:
            # Representative = member with highest mean absolute Spearman r to others
            mean_abs_r = {
                m: corr.loc[m, [x for x in members if x != m]].abs().mean()
                for m in members
            }
            rep = max(mean_abs_r, key=mean_abs_r.__getitem__)
        for m in members:
            cluster_info[m] = rep

    return cluster_info


def univariate_analysis() -> None:
    """Compare Δ values (treatment vs control) for all 20 biomarkers → stdout."""
    df    = load_composites()
    treat = df[df['Treatment'] == 1]
    ctrl  = df[df['Treatment'] == 0]

    _log_names   = {bm.name for bm in LOG_BIOMARKERS}
    _accel_names = {bm.name for bm in AGE_ACCEL_BIOMARKERS}

    def display_name(name: str) -> str:
        if name in _accel_names:
            return f"{name} (res)"
        if name in _log_names:
            return f"{name} (log)"
        return name

    # --- Per-biomarker raw statistics ---
    records = []
    for bm in ANALYSIS_BIOMARKERS:
        t_vals = treat[bm.delta_col].dropna()
        c_vals = ctrl[bm.delta_col].dropna()
        p_raw, test = _two_sample_test(t_vals, c_vals)
        d = _cohens_d(t_vals, c_vals)
        records.append({
            'name':     bm.name,
            'label':    display_name(bm.name),
            'col':      bm.delta_col,
            't_mean':   t_vals.mean(),
            'c_mean':   c_vals.mean(),
            'diff':     t_vals.mean() - c_vals.mean(),
            'test':     test,
            'p_raw':    p_raw,
            'd':        d,
            'effect':   _effect_label(d),
        })

    results = pd.DataFrame(records)

    # --- Naive FDR (Benjamini-Hochberg across all 20 tests) ---
    _, q_naive, _, _ = multipletests(results['p_raw'].values, method='fdr_bh')
    results['q_naive'] = q_naive

    # --- Cluster-aware FDR ---
    delta_cols = [bm.delta_col for bm in ANALYSIS_BIOMARKERS]
    cluster_info = _cluster_representatives(df, delta_cols)   # col → rep_col

    rep_cols = sorted(set(cluster_info.values()), key=delta_cols.index)
    rep_mask = results['col'].isin(rep_cols)
    _, q_rep, _, _ = multipletests(results.loc[rep_mask, 'p_raw'].values, method='fdr_bh')
    rep_q_map = dict(zip(results.loc[rep_mask, 'col'].values, q_rep))

    # Every biomarker inherits its cluster representative's q-value
    results['q_cluster'] = results['col'].map(
        lambda col: rep_q_map[cluster_info[col]]  # look up representative's q
    )
    results['is_rep'] = results['col'].isin(rep_cols)

    n_independent = len(rep_cols)
    n_tests = len(results)

    # --- Print results sorted by raw p-value ---
    header = (f"{'Biomarker':<29} {'Treat Δ':>9} {'Ctrl Δ':>9} {'Diff':>9} "
              f"{'Test':<16} {'p_raw':>7} {'q_naive':>8} {'q_clust':>8} "
              f"{'d':>7} {'Effect':<10}")
    print(header)
    print('-' * len(header))

    for _, row in results.sort_values('p_raw').iterrows():
        # Significance markers use cluster-aware q
        if row['q_cluster'] < 0.001:
            sig = '***'
        elif row['q_cluster'] < 0.01:
            sig = '**'
        elif row['q_cluster'] < 0.05:
            sig = '*'
        elif row['p_raw'] < 0.05:
            sig = '†'   # nominally significant but not FDR-corrected
        else:
            sig = ''
        rep_marker = '*' if row['is_rep'] else ' '   # mark cluster representatives
        print(f"{row['label']:<28}{rep_marker} {row['t_mean']:>9.3f} {row['c_mean']:>9.3f} "
              f"{row['diff']:>9.3f} {row['test']:<16} {row['p_raw']:>7.4f} "
              f"{row['q_naive']:>8.4f} {row['q_cluster']:>8.4f} "
              f"{row['d']:>7.3f} {row['effect']:<10}  {sig}")

    print(f"\n* = cluster representative used for cluster-aware FDR ({n_independent} clusters from {n_tests} biomarkers)")
    print("Significance (q_clust): *** q<0.001  ** q<0.01  * q<0.05  † p_raw<0.05 only")

    n_sig_raw    = (results['p_raw']    < 0.05).sum()
    n_sig_naive  = (results['q_naive']  < 0.05).sum()
    n_sig_clust  = (results['q_cluster'] < 0.05).sum()
    print(f"\nSignificant at p<0.05 (uncorrected):              {n_sig_raw}/{n_tests}")
    print(f"Significant at FDR<0.05 (naive,   n={n_tests} tests): {n_sig_naive}/{n_tests}")
    print(f"Significant at FDR<0.05 (cluster, n={n_independent} tests): {n_sig_clust}/{n_tests}")
