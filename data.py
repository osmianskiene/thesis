from functools import cache

import numpy as np
import pandas as pd

DATA_PATH = 'data/patient_data.csv'

# Biomarkers analysed on the natural-log scale: (log_t0_col, raw_t0_col, log_delta_col, raw_t1_col)
# Δ on the log scale = ln(T1/T0), interpretable as a proportional (fold) change.
_LOG_SCALE_BIOMARKERS: list[tuple[str, str, str, str]] = [
    ('CRP log T0',     'CRB T0',      'CRP log Δ',     'CRB T1'),
    ('Insulin log T0', 'Insulinas T0','Insulin log Δ',  'Insulinas T1'),
    ('HOMA-IR log T0', 'HOMA IR T0',  'HOMA-IR log Δ', 'HOMA IR T1'),
]

# Age biomarkers for which age acceleration is computed: (name, t0_col, t1_col).
# OLS is fitted on T0 data; residuals at T0 and T1 are the age-acceleration estimates.
_AGE_BIOMARKERS: list[tuple[str, str, str]] = [
    ('Biological Age', 'Biological Age T0', 'Biological Age T1'),
    ('Eye Age',        'Eye Age T0',        'Eye Age T1'),
    ('Hearing Age',    'Hearing Age T0',    'Hearing Age T1'),
    ('Memory Age',     'Memory Age T0',     'Memory Age T1'),
]


# Cached: CSV is read and processed only once per process.
@cache
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, sep=';', decimal=',', encoding='utf-8-sig')
    df.rename(columns={
        'Eilės nr.': 'Row',
        'Lytis': 'Sex',
        'Grupė': 'Group',
        'Tyrimų centras': 'Center',
    }, inplace=True)
    df['Treatment'] = (df['Group'] == 'Tyrimai + papildai').astype(int)
    df['Chronological Age Δ'] = df['Chronological Age T1'] - df['Chronological Age T0']
    df['Age Group'] = pd.cut(
        df['Chronological Age T0'],
        bins=[-np.inf, 39, 54, np.inf],
        labels=['Young', 'Average', 'Old'],
    )
    return df


# Cached: computed only once per process.
@cache
def load_data_log_fold_change() -> pd.DataFrame:
    """Return load_data() extended with ln(T0) and ln(T1/T0) columns for CRP, Insulin, HOMA-IR."""
    df = load_data().copy()
    for log_t0_col, raw_t0_col, log_delta_col, raw_t1_col in _LOG_SCALE_BIOMARKERS:
        t0 = df[raw_t0_col]
        t1 = df[raw_t1_col]
        df[log_t0_col]    = np.log(t0)            # ln(baseline)
        df[log_delta_col] = np.log(t1 / t0)       # ln(T1/T0) = log fold-change
    return df

# Cached: computed only once per process.
@cache
def load_data_age_accel() -> pd.DataFrame:
    """Return load_data() extended with age-acceleration columns for all four age biomarkers.

    For each biomarker an OLS model (biomarker_T0 ~ Chronological Age T0) is fitted on T0 data.
    Age acceleration = observed value − model-predicted value given chronological age.
    Added columns per biomarker (e.g. 'Biological Age'):
      '{name} AgeAccel T0'  — residual at baseline
      '{name} AgeAccel Δ'   — change in acceleration (T1 residual − T0 residual)
    """
    df = load_data_log_fold_change().copy()
    chron_t0 = df['Chronological Age T0']
    chron_t1 = df['Chronological Age T1']

    for name, t0_col, t1_col in _AGE_BIOMARKERS:
        bm_t0 = df[t0_col]
        bm_t1 = df[t1_col]

        # Fit OLS on rows where both chronological age and biomarker T0 are observed.
        mask = chron_t0.notna() & bm_t0.notna()
        # np.polyfit degree-1 returns [slope, intercept].
        slope, intercept = np.polyfit(chron_t0[mask], bm_t0[mask], 1)

        # Age acceleration = observed − expected; NaN propagates automatically.
        accel_t0 = bm_t0 - (slope * chron_t0 + intercept)
        accel_t1 = bm_t1 - (slope * chron_t1 + intercept)

        df[f'{name} AgeAccel T0'] = accel_t0
        df[f'{name} AgeAccel Δ']  = accel_t1 - accel_t0   # ΔAgeAccel = AgeAccel_T1 − AgeAccel_T0

    return df


# Cached: computed only once per process.
@cache
def load_composites() -> pd.DataFrame:
    """Return load_data_age_accel() extended with an Inflammation composite column.

    Inflammation composite = mean of z-scored log(CRP), IL-6, and Fibrinogen.
    Each component is z-scored using its own T0 mean and SD so all three
    contribute equally regardless of their original units.
    T0 statistics are used for both T0 and T1 standardisation to avoid
    data leakage from the post-treatment measurements.

    Added columns:
      'Inflammation composite T0'  — composite score at baseline
      'Inflammation composite T1'  — composite score post-treatment
      'Inflammation composite Δ'   — change (T1 − T0)
    """
    df = load_data_age_accel().copy()

    # CRP log T1 = log T0 + log Δ  (both already in df from load_data_log_fold_change)
    crp_t0 = df['CRP log T0']
    crp_t1 = df['CRP log T0'] + df['CRP log Δ']   # ln(T1)
    il6_t0 = df['IL 6 T0']
    il6_t1 = df['IL 6 T1']
    fib_t0 = df['Fibrinogenas T0']
    fib_t1 = df['Fibrinogenas T1']

    def _zscore(t0: pd.Series, t1: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Z-score both timepoints using T0 mean and SD."""
        mean, sd = t0.mean(), t0.std(ddof=1)
        return (t0 - mean) / sd, (t1 - mean) / sd

    z_crp_t0, z_crp_t1 = _zscore(crp_t0, crp_t1)
    z_il6_t0, z_il6_t1 = _zscore(il6_t0, il6_t1)
    z_fib_t0, z_fib_t1 = _zscore(fib_t0, fib_t1)

    # Average the three z-scores; NaN propagates if any component is missing
    df['Inflammation composite T0'] = (z_crp_t0 + z_il6_t0 + z_fib_t0) / 3
    df['Inflammation composite T1'] = (z_crp_t1 + z_il6_t1 + z_fib_t1) / 3
    df['Inflammation composite Δ']  = df['Inflammation composite T1'] - df['Inflammation composite T0']

    return df


