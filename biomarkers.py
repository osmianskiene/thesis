from typing import NamedTuple


class Biomarker(NamedTuple):
    name: str
    t0_col: str       # column name for baseline value
    delta_col: str    # column name for change (T1 − T0)
    direction: str | None  # 'up' = higher is better, 'down' = lower is better


# Age biomarkers expressed as age acceleration (see data.load_data_age_accel).
# Column names match those appended by that function.
AGE_ACCEL_BIOMARKERS: list[Biomarker] = [
    Biomarker('Biological Age', 'Biological Age AgeAccel T0', 'Biological Age AgeAccel Δ', 'down'),
    Biomarker('Eye Age',        'Eye Age AgeAccel T0',        'Eye Age AgeAccel Δ',        'down'),
    Biomarker('Hearing Age',    'Hearing Age AgeAccel T0',    'Hearing Age AgeAccel Δ',    'down'),
    Biomarker('Memory Age',     'Memory Age AgeAccel T0',     'Memory Age AgeAccel Δ',     'down'),
]

# The three biomarkers re-analysed on the natural-log scale (see data.load_data_log_fold_change).
# Column names match those appended by that function.
LOG_BIOMARKERS: list[Biomarker] = [
    Biomarker('CRP',     'CRP log T0',     'CRP log Δ',     'down'),
    Biomarker('Insulin', 'Insulin log T0', 'Insulin log Δ', 'down'),
    Biomarker('HOMA-IR', 'HOMA-IR log T0', 'HOMA-IR log Δ', 'down'),
]

def _preferred(base: list['Biomarker'], *overrides: list['Biomarker']) -> list['Biomarker']:
    """Build a biomarker list by overriding base entries with preferred-scale versions.

    For each biomarker in base, if a same-named entry exists in any override list
    (checked left to right, last wins), that entry's columns are used instead.
    """
    # *unpacking: iterates all override lists into a single flat sequence
    lookup = {bm.name: bm for overrides_list in overrides for bm in overrides_list}
    return [lookup.get(bm.name, bm) for bm in base]


BIOMARKERS: list[Biomarker] = [
    Biomarker('NAD⁺',               'NAD+ T0',                   'NAD+ Δ',                   'up'),
    Biomarker('CRP',                 'CRB T0',                    'CRB Δ',                    'down'),
    Biomarker('IL-6',                'IL 6 T0',                   'IL 6 Δ',                   'down'),
    Biomarker('Fibrinogen',          'Fibrinogenas T0',            'Fibrinogenas Δ',           'down'),
    Biomarker('Glucose',             'Gliukozė T0',               'Gliukozė Δ',               'down'),
    Biomarker('Insulin',             'Insulinas T0',              'Insulinas Δ',              'down'),
    Biomarker('HOMA-IR',             'HOMA IR T0',                'HOMA IR Δ',                'down'),
    Biomarker('HbA1c',               'HbA1c mmol T0',             'HbA1c mmol Δ',             'down'),
    Biomarker('Triglycerides',       'Triacilgliceroliai T0',     'Triacilgliceroliai Δ',     'down'),
    Biomarker('HDL Cholesterol',     'Cholesterolis DTL T0',      'Cholesterolis DTL Δ',      'up'),
    Biomarker('LDL Cholesterol',     'Cholesterolis MTL T0',      'Cholesterolis MTL Δ',      'down'),
    Biomarker('Non-HDL Cholesterol', 'Cholesterolis Ne DTL T0',   'Cholesterolis Ne DTL Δ',   'down'),
    Biomarker('Total Cholesterol',   'Cholesterolis bendras T0',  'Cholesterolis bendras Δ',  'down'),
    Biomarker('AFP',                 'AFP T0',                    'AFP Δ',                    'down'),
    Biomarker('LDH',                 'LDH T0',                    'LDH Δ',                    'down'),
    Biomarker('Biological Age',      'Biological Age T0',          'Biological Age Δ',         'down'),
    Biomarker('Eye Age',             'Eye Age T0',                'Eye Age Δ',                'down'),
    Biomarker('Hearing Age',         'Hearing Age T0',             'Hearing Age Δ',            'down'),
    Biomarker('Memory Age',          'Memory Age T0',              'Memory Age Δ',             'down'),
    Biomarker('Inflammation Score',  'Inflammation Score T0',     'Inflammation Score Δ',     'down'),
]

# The 20 biomarkers resolved to their preferred analysis column:
# age-acceleration residual > log-transformed > raw.
# Use this list (with load_data_age_accel()) for all analysis commands.
ANALYSIS_BIOMARKERS: list[Biomarker] = _preferred(BIOMARKERS, LOG_BIOMARKERS, AGE_ACCEL_BIOMARKERS)
