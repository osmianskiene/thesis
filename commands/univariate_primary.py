import pandas as pd

from biomarkers import ANALYSIS_BIOMARKERS, Biomarker
from commands.univariate import _cohens_d, _effect_label, _two_sample_test
from data import load_composites

# Inflammation composite is not in ANALYSIS_BIOMARKERS yet (no FDR role),
# so define its Biomarker entry locally for this command.
_INFLAMMATION_COMPOSITE = Biomarker(
    'Inflammation composite',
    'Inflammation composite T0',
    'Inflammation composite Δ',
    'down',
)

# Analysis tiers — ordered as they should appear in output.
# Primary: pre-specified primary endpoint; tested first, interpreted strictly.
# Secondary: pre-specified secondary endpoints; nominally significant findings reported.
# Exploratory: remaining 16 biomarkers (20 ANALYSIS_BIOMARKERS minus primary and secondary); hypothesis-generating only.
_PRIMARY_NAMES   = ['Biological Age']
_SECONDARY_NAMES = ['NAD⁺', 'Inflammation composite', 'HOMA-IR', 'LDL Cholesterol']

_analysis_by_name = {bm.name: bm for bm in ANALYSIS_BIOMARKERS}

_PRIMARY    = [_analysis_by_name[n] for n in _PRIMARY_NAMES]
_SECONDARY  = [
    _INFLAMMATION_COMPOSITE if n == 'Inflammation composite' else _analysis_by_name[n]
    for n in _SECONDARY_NAMES
]
_used_names = set(_PRIMARY_NAMES) | set(_SECONDARY_NAMES)
_EXPLORATORY = [bm for bm in ANALYSIS_BIOMARKERS if bm.name not in _used_names]


def _print_tier(
    label: str,
    biomarkers: list[Biomarker],
    treat: pd.DataFrame,
    ctrl: pd.DataFrame,
    accel_names: set[str],
    log_names: set[str],
) -> None:
    def display_name(name: str) -> str:
        if name in accel_names:
            return f"{name} (res)"
        if name in log_names:
            return f"{name} (log)"
        return name

    header = (f"{'Biomarker':<29} {'Treat Δ':>9} {'Ctrl Δ':>9} {'Diff':>9} "
              f"{'Test':<16} {'p':>7} {'d':>7} {'Effect':<10}")
    divider = '-' * len(header)

    print(f"\n{label}")
    print(divider)
    print(header)
    print(divider)

    for bm in biomarkers:
        t_vals = treat[bm.delta_col].dropna()
        c_vals = ctrl[bm.delta_col].dropna()
        p, test = _two_sample_test(t_vals, c_vals)
        d = _cohens_d(t_vals, c_vals)
        sig = '*' if p < 0.05 else ''
        print(f"{display_name(bm.name):<29} {t_vals.mean():>9.3f} {c_vals.mean():>9.3f} "
              f"{t_vals.mean() - c_vals.mean():>9.3f} {test:<16} {p:>7.4f} "
              f"{d:>7.3f} {_effect_label(d):<10}  {sig}")


def univariate_primary(max_row: int | None = None) -> None:
    """Tiered Δ analysis (primary / secondary / exploratory) without FDR correction → stdout."""
    df = load_composites()
    if max_row is not None:
        df = df[df['Row'] <= max_row]
    treat = df[df['Treatment'] == 1]
    ctrl  = df[df['Treatment'] == 0]

    from biomarkers import AGE_ACCEL_BIOMARKERS, LOG_BIOMARKERS  # inline to match balance.py pattern
    accel_names = {bm.name for bm in AGE_ACCEL_BIOMARKERS}
    log_names   = {bm.name for bm in LOG_BIOMARKERS}

    subset = f" (rows 1–{max_row})" if max_row is not None else ""
    print(f"Tiered univariate analysis — no FDR correction applied{subset}")
    print("* = p < 0.05 (nominal)")

    _print_tier(f"PRIMARY (n={len(_PRIMARY)})",      _PRIMARY,     treat, ctrl, accel_names, log_names)
    _print_tier(f"SECONDARY (n={len(_SECONDARY)})", _SECONDARY,   treat, ctrl, accel_names, log_names)
    _print_tier(f"EXPLORATORY (n={len(_EXPLORATORY)})", _EXPLORATORY, treat, ctrl, accel_names, log_names)
