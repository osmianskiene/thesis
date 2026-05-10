import pandas as pd
from scipy import stats as scipy_stats


def is_normal(series: pd.Series, alpha: float = 0.05) -> bool:
    d = series.dropna()
    if len(d) < 3:
        return False
    sample = d.sample(min(50, len(d)), random_state=42)
    _, p = scipy_stats.shapiro(sample)
    return p > alpha


def fmt_stat(series: pd.Series) -> str:
    """mean±SD for normal data, median [Q1–Q3] otherwise."""
    d = series.dropna()
    if len(d) == 0:
        return '—'
    if is_normal(d):
        return f"{d.mean():.2f}±{d.std(ddof=1):.2f}"
    q1, q3 = d.quantile(0.25), d.quantile(0.75)
    return f"{d.median():.2f} [{q1:.2f}–{q3:.2f}]"


def central(series: pd.Series) -> float:
    d = series.dropna()
    if len(d) == 0:
        return 0.0
    return float(d.mean() if is_normal(d) else d.median())


def get_subgroups(df: pd.DataFrame, col: str) -> dict:
    # Keys: T=treatment, C=control; M=male, F=female; Y=Young, A=Average, O=Old
    return {
        'all': df[col].dropna(),
        'T':   df[df['Treatment'] == 1][col].dropna(),
        'C':   df[df['Treatment'] == 0][col].dropna(),
        'M':   df[df['Sex'] == 'm'][col].dropna(),
        'F':   df[df['Sex'] == 'f'][col].dropna(),
        'Y':   df[df['Age Group'] == 'Young'][col].dropna(),
        'A':   df[df['Age Group'] == 'Average'][col].dropna(),
        'O':   df[df['Age Group'] == 'Old'][col].dropna(),
    }
