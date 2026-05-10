import pandas as pd
from scipy.stats import fisher_exact, chi2_contingency, shapiro, levene, ttest_ind, mannwhitneyu

from biomarkers import ANALYSIS_BIOMARKERS, AGE_ACCEL_BIOMARKERS, LOG_BIOMARKERS
from data import load_composites, load_data


def baseline_balance_sex() -> None:
    """Fisher's exact test for sex distribution between treatment and control."""
    df = load_data()

    treat = df[df['Treatment'] == 1]
    ctrl  = df[df['Treatment'] == 0]

    n_treat_f = int((treat['Sex'] == 'f').sum())
    n_treat_m = int((treat['Sex'] == 'm').sum())
    n_ctrl_f  = int((ctrl['Sex']  == 'f').sum())
    n_ctrl_m  = int((ctrl['Sex']  == 'm').sum())

    # Contingency table rows=sex (F/M), cols=group (treatment/control)
    _, p = fisher_exact([[n_treat_f, n_ctrl_f],
                         [n_treat_m, n_ctrl_m]])

    n_treat = len(treat)
    n_ctrl  = len(ctrl)

    print("Sex distribution (Fisher's exact test)")
    print(f'  Female: Treatment {n_treat_f}/{n_treat} ({n_treat_f/n_treat*100:.1f}%), '
          f'Control {n_ctrl_f}/{n_ctrl} ({n_ctrl_f/n_ctrl*100:.1f}%)')
    print(f'  Male:   Treatment {n_treat_m}/{n_treat} ({n_treat_m/n_treat*100:.1f}%), '
          f'Control {n_ctrl_m}/{n_ctrl} ({n_ctrl_m/n_ctrl*100:.1f}%)')
    print(f'  p = {p:.3f}')


def baseline_balance_age() -> None:
    """Chi-square test for age group distribution between treatment and control."""
    df = load_data()

    treat = df[df['Treatment'] == 1]
    ctrl  = df[df['Treatment'] == 0]

    age_groups = ['Young', 'Average', 'Old']  # Young <40, Average 40-54, Old >54
    counts_treat = [int((treat['Age Group'] == g).sum()) for g in age_groups]
    counts_ctrl  = [int((ctrl['Age Group']  == g).sum()) for g in age_groups]

    # chi2_contingency expects a 2-D array; rows=age groups, cols=groups
    _, p, _, _ = chi2_contingency(
        [[t, c] for t, c in zip(counts_treat, counts_ctrl)]
    )

    n_treat = len(treat)
    n_ctrl  = len(ctrl)

    print('Age group distribution (chi-square test)')
    for group, n_t, n_c in zip(age_groups, counts_treat, counts_ctrl):
        print(f'  {group:<8}: Treatment {n_t}/{n_treat} ({n_t/n_treat*100:.1f}%), '
              f'Control {n_c}/{n_ctrl} ({n_c/n_ctrl*100:.1f}%)')
    print(f'  p = {p:.3f}')


def _two_sample_test(a: pd.Series, b: pd.Series, alpha: float = 0.05) -> tuple[float, str]:
    """Choose and run the appropriate two-sample test.

    Both normal + equal variances   → Student t-test
    Both normal + unequal variances → Welch t-test
    Otherwise                       → Mann-Whitney U
    Returns (p_value, test_name).
    """
    def _is_normal(s: pd.Series) -> bool:
        s = s.dropna()
        if len(s) < 3:
            return False
        _, p = shapiro(s.sample(min(50, len(s)), random_state=42))
        return p > alpha

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


def baseline_balance_biomarkers() -> None:
    """Two-sample test on T0 values for all 20 biomarkers (treatment vs control) → stdout."""
    df    = load_composites()
    treat = df[df['Treatment'] == 1]
    ctrl  = df[df['Treatment'] == 0]

    def fmt(s: pd.Series) -> str:
        """mean±SD for normal data, median [Q1–Q3] otherwise."""
        from stats import is_normal  # inline import avoids top-level circular risk
        if is_normal(s):
            return f"{s.mean():.2f}±{s.std(ddof=1):.2f}"
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        return f"{s.median():.2f} [{q1:.2f}–{q3:.2f}]"

    _log_names   = {bm.name for bm in LOG_BIOMARKERS}
    _accel_names = {bm.name for bm in AGE_ACCEL_BIOMARKERS}

    def display_name(name: str) -> str:
        if name in _accel_names:
            return f"{name} (res)"
        if name in _log_names:
            return f"{name} (log)"
        return name

    print(f"{'Biomarker':<29} {'Treatment':<22} {'Control':<22} {'Test':<16} {'p':>6}  {'Sig'}")
    print('-' * 104)

    for bm in ANALYSIS_BIOMARKERS:
        t_vals = treat[bm.t0_col].dropna()
        c_vals = ctrl[bm.t0_col].dropna()
        p, test = _two_sample_test(t_vals, c_vals)
        sig = '*' if p < 0.05 else ''
        print(f"{display_name(bm.name):<29} {fmt(t_vals):<22} {fmt(c_vals):<22} {test:<16} {p:>6.3f}  {sig}")
