import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_1samp, wilcoxon

from biomarkers import AGE_ACCEL_BIOMARKERS, ANALYSIS_BIOMARKERS, LOG_BIOMARKERS
from data import load_composites


def _within_test(delta: pd.Series, min_n: int = 6) -> tuple[float, str]:
    """Test whether Δ values differ significantly from zero.

    Normal distribution → one-sample t-test (H₀: μ = 0)
    Otherwise          → Wilcoxon signed-rank test (H₀: median = 0)
    Returns (p_value, test_name); p=nan if n < min_n.
    """
    d = delta.dropna()
    if len(d) < min_n:
        return float('nan'), '—'
    _, p_norm = shapiro(d.sample(min(50, len(d)), random_state=42))
    if p_norm > 0.05:
        _, p = ttest_1samp(d, 0)
        return float(p), 't'
    _, p = wilcoxon(d)
    return float(p), 'W'


def _fmt_p(p: float) -> str:
    """Format p-value in 6 characters: '<.001*', '.045 *', '.234  ', '  —   '."""
    if np.isnan(p):
        return '  —   '
    s = '<.001' if p < 0.001 else f'{p:.3f}'
    star = '*' if p < 0.05 else ' '
    return f'{s}{star}'


def _fmt_mean(delta: pd.Series) -> str:
    """Format mean change in 6 characters: '+1.23 ', '-0.05 ', '  —   '."""
    d = delta.dropna()
    if len(d) < 1:
        return '  —   '
    m = d.mean()
    s = f'{m:+.2f}'
    # Truncate if too long (e.g. '-10.23' is already 6 chars)
    return f'{s:<6}'[:6]


# All subgroups as (label, filter_function) pairs, grouped for display.
# Groups: single-factor, two-way (T×sex, T×age, sex×age), three-way (T×sex×age).
_SUBGROUPS: list[tuple[str, object]] = [
    # single
    ('all',   lambda df: df),
    ('T',     lambda df: df[df['Treatment'] == 1]),
    ('C',     lambda df: df[df['Treatment'] == 0]),
    ('M',     lambda df: df[df['Sex'] == 'm']),
    ('F',     lambda df: df[df['Sex'] == 'f']),
    ('Y',     lambda df: df[df['Age Group'] == 'Young']),
    ('Avg',   lambda df: df[df['Age Group'] == 'Average']),
    ('O',     lambda df: df[df['Age Group'] == 'Old']),
    # T × sex
    ('T×M',   lambda df: df[(df['Treatment'] == 1) & (df['Sex'] == 'm')]),
    ('T×F',   lambda df: df[(df['Treatment'] == 1) & (df['Sex'] == 'f')]),
    ('C×M',   lambda df: df[(df['Treatment'] == 0) & (df['Sex'] == 'm')]),
    ('C×F',   lambda df: df[(df['Treatment'] == 0) & (df['Sex'] == 'f')]),
    # T × age
    ('T×Y',   lambda df: df[(df['Treatment'] == 1) & (df['Age Group'] == 'Young')]),
    ('T×Avg', lambda df: df[(df['Treatment'] == 1) & (df['Age Group'] == 'Average')]),
    ('T×O',   lambda df: df[(df['Treatment'] == 1) & (df['Age Group'] == 'Old')]),
    ('C×Y',   lambda df: df[(df['Treatment'] == 0) & (df['Age Group'] == 'Young')]),
    ('C×Avg', lambda df: df[(df['Treatment'] == 0) & (df['Age Group'] == 'Average')]),
    ('C×O',   lambda df: df[(df['Treatment'] == 0) & (df['Age Group'] == 'Old')]),
    # sex × age
    ('M×Y',   lambda df: df[(df['Sex'] == 'm') & (df['Age Group'] == 'Young')]),
    ('M×Avg', lambda df: df[(df['Sex'] == 'm') & (df['Age Group'] == 'Average')]),
    ('M×O',   lambda df: df[(df['Sex'] == 'm') & (df['Age Group'] == 'Old')]),
    ('F×Y',   lambda df: df[(df['Sex'] == 'f') & (df['Age Group'] == 'Young')]),
    ('F×Avg', lambda df: df[(df['Sex'] == 'f') & (df['Age Group'] == 'Average')]),
    ('F×O',   lambda df: df[(df['Sex'] == 'f') & (df['Age Group'] == 'Old')]),
    # T × sex × age
    ('T×M×Y',   lambda df: df[(df['Treatment'] == 1) & (df['Sex'] == 'm') & (df['Age Group'] == 'Young')]),
    ('T×M×Avg', lambda df: df[(df['Treatment'] == 1) & (df['Sex'] == 'm') & (df['Age Group'] == 'Average')]),
    ('T×M×O',   lambda df: df[(df['Treatment'] == 1) & (df['Sex'] == 'm') & (df['Age Group'] == 'Old')]),
    ('T×F×Y',   lambda df: df[(df['Treatment'] == 1) & (df['Sex'] == 'f') & (df['Age Group'] == 'Young')]),
    ('T×F×Avg', lambda df: df[(df['Treatment'] == 1) & (df['Sex'] == 'f') & (df['Age Group'] == 'Average')]),
    ('T×F×O',   lambda df: df[(df['Treatment'] == 1) & (df['Sex'] == 'f') & (df['Age Group'] == 'Old')]),
    ('C×M×Y',   lambda df: df[(df['Treatment'] == 0) & (df['Sex'] == 'm') & (df['Age Group'] == 'Young')]),
    ('C×M×Avg', lambda df: df[(df['Treatment'] == 0) & (df['Sex'] == 'm') & (df['Age Group'] == 'Average')]),
    ('C×M×O',   lambda df: df[(df['Treatment'] == 0) & (df['Sex'] == 'm') & (df['Age Group'] == 'Old')]),
    ('C×F×Y',   lambda df: df[(df['Treatment'] == 0) & (df['Sex'] == 'f') & (df['Age Group'] == 'Young')]),
    ('C×F×Avg', lambda df: df[(df['Treatment'] == 0) & (df['Sex'] == 'f') & (df['Age Group'] == 'Average')]),
    ('C×F×O',   lambda df: df[(df['Treatment'] == 0) & (df['Sex'] == 'f') & (df['Age Group'] == 'Old')]),
]

# Visual group separators: label of the first column in each group → group header
_GROUP_HEADERS: dict[str, str] = {
    'all':    'Single-factor',
    'T×M':    'Treatment × Sex',
    'T×Y':    'Treatment × Age',
    'M×Y':    'Sex × Age',
    'T×M×Y':  'Treatment × Sex × Age',
}


def within_change() -> None:
    """Test within-group T0→T1 change for all biomarkers across all subgroups → stdout."""
    df = load_composites()

    _accel_names = {bm.name for bm in AGE_ACCEL_BIOMARKERS}
    _log_names   = {bm.name for bm in LOG_BIOMARKERS}

    def display_name(name: str) -> str:
        if name in _accel_names:
            return f"{name} (res)"
        if name in _log_names:
            return f"{name} (log)"
        return name

    col_w   = 6    # width of each p-value cell
    name_w  = 29   # width of biomarker name column

    labels = [label for label, _ in _SUBGROUPS]

    # Print group-header row
    header_line = ' ' * name_w
    for label in labels:
        if label in _GROUP_HEADERS:
            gh = _GROUP_HEADERS[label]
            # Count how many columns belong to this group
            start = labels.index(label)
            if label == list(_GROUP_HEADERS.keys())[-1]:  # last group
                # *unpacking: _ discards the filter function
                count = sum(1 for l, _ in _SUBGROUPS[start:])
            else:
                keys = list(_GROUP_HEADERS.keys())
                next_key = keys[keys.index(label) + 1]
                count = labels.index(next_key) - start
            gh_truncated = gh[:count * col_w - 1]
            header_line += f'{gh_truncated:<{count * col_w}}'
    print(header_line.rstrip())

    # Print column-label row
    label_line = f"{'Biomarker':<{name_w}}"
    for label in labels:
        label_line += f'{label:>{col_w}}'
    print(label_line)
    print('-' * len(label_line))

    # Print three rows per biomarker: name, p-value, mean change
    for bm in ANALYSIS_BIOMARKERS:
        p_vals: list[float] = []
        mean_cells: list[str] = []
        for _, filt in _SUBGROUPS:
            subdf = filt(df)
            p, _ = _within_test(subdf[bm.delta_col])
            p_vals.append(p)
            mean_cells.append(_fmt_mean(subdf[bm.delta_col]))

        name_row = f'{display_name(bm.name):<{name_w}}' + ' ' * (len(labels) * col_w)
        p_row    = f'{"  p-value":<{name_w}}' + ''.join(_fmt_p(p) for p in p_vals)
        mean_row = f'{"  mean Δ":<{name_w}}' + ''.join(mean_cells)
        print(name_row.rstrip())
        print(p_row)
        print(mean_row)

    print()
    print('* p < 0.05  |  — n < 6  |  t = one-sample t-test  |  W = Wilcoxon signed-rank')
    print('Subgroup keys: T=treatment, C=control, M=male, F=female, Y=young, Avg=average, O=old')
