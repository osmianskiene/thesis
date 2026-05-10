import click

from commands.ancova import ancova, ancova_primary
from commands.balance import baseline_balance_age, baseline_balance_biomarkers, baseline_balance_sex
from commands.descriptive import descriptive_docx
from commands.outliers import outliers_cooks, outliers_studentized, outliers_zscore
from commands.plots import descriptive_plots, descriptive_plots_age_accel, descriptive_plots_log
from commands.univariate import univariate_analysis
from commands.univariate_primary import univariate_primary
from commands.univariate_strata import univariate_strata, univariate_strata_docx
from commands.univariate_il6 import univariate_il6
from commands.composite_scores import composite_scores
from commands.mahalanobis import mahalanobis_analysis
from commands.pca import pca_analysis
from commands.responders import responders_classify
from commands.responders_ml import responders_ml
from commands.responders_ml2 import (
    responders_ml2,
    TARGET_BIOLOGICAL_AGE, TARGET_STRICT,
    TARGET_INFLAMMATION, TARGET_GLYCEMIC, TARGET_LIPID, TARGET_AGEING, TARGET_CELLULAR,
)
from commands.responders_bootstrap import responders_bootstrap
from commands.within import within_change


@click.group(context_settings={'max_content_width': 200, 'terminal_width': 200})
def cli() -> None:
    pass


@cli.command(name='descriptive:docx')
def cmd_descriptive_docx() -> None:
    """Descriptive statistics table → output/a1_descriptive.docx"""
    descriptive_docx()


@cli.command(name='descriptive:plots:t0')
def cmd_descriptive_plots_t0() -> None:
    """Baseline distribution plots → output/plots_t0/*.png"""
    descriptive_plots('t0')


@cli.command(name='descriptive:plots:delta')
def cmd_descriptive_plots_delta() -> None:
    """Change-from-baseline distribution plots → output/plots_delta/*.png"""
    descriptive_plots('delta')


@cli.command(name='descriptive:plots:age-accel:t0')
def cmd_descriptive_plots_age_accel_t0() -> None:
    """Age-acceleration baseline plots for age biomarkers → output/plots_age_accel_t0/*.png"""
    descriptive_plots_age_accel('t0')


@cli.command(name='descriptive:plots:age-accel:delta')
def cmd_descriptive_plots_age_accel_delta() -> None:
    """Age-acceleration change plots for age biomarkers → output/plots_age_accel_delta/*.png"""
    descriptive_plots_age_accel('delta')


@cli.command(name='descriptive:plots:log:t0')
def cmd_descriptive_plots_log_t0() -> None:
    """Log-scale baseline plots for CRP/Insulin/HOMA-IR → output/plots_log_t0/*.png"""
    descriptive_plots_log('t0')


@cli.command(name='descriptive:plots:log:delta')
def cmd_descriptive_plots_log_delta() -> None:
    """Log fold-change plots for CRP/Insulin/HOMA-IR → output/plots_log_delta/*.png"""
    descriptive_plots_log('delta')


@cli.command(name='baseline:balance:biomarkers')
def cmd_balance_biomarkers() -> None:
    """T0 two-sample test for all biomarkers → stdout"""
    baseline_balance_biomarkers()


@cli.command(name='baseline:balance:sex')
def cmd_balance_sex() -> None:
    """Sex distribution Fisher's exact test → stdout"""
    baseline_balance_sex()


@cli.command(name='baseline:balance:age')
def cmd_balance_age() -> None:
    """Age group distribution chi-square test → stdout"""
    baseline_balance_age()


@cli.command(name='outliers:zscore')
def cmd_outliers_zscore() -> None:
    """Z-score outliers (|z| > 3) on Δ values → output/a1_outliers_zscore.docx"""
    outliers_zscore()


@cli.command(name='outliers:studentized')
def cmd_outliers_studentized() -> None:
    """Studentized residuals (|r*| > 3) from ANCOVA → output/a1_outliers_studentized.docx"""
    outliers_studentized()


@cli.command(name='outliers:cooks')
def cmd_outliers_cooks() -> None:
    """Cook's distance (D > 4/n) from ANCOVA → output/a1_outliers_cooks.docx"""
    outliers_cooks()


@cli.command(name='univariate')
def cmd_univariate() -> None:
    """Δ two-sample tests, Cohen's d, and FDR correction for all 20 biomarkers → stdout"""
    univariate_analysis()


@cli.command(name='univariate:primary')
def cmd_univariate_primary() -> None:
    """Tiered Δ analysis (primary / secondary / exploratory) without FDR → stdout"""
    univariate_primary()


@cli.command(name='univariate:strata')
def cmd_univariate_strata() -> None:
    """T vs C Δ comparison within each sex/age stratum for all biomarkers → stdout"""
    univariate_strata()


@cli.command(name='univariate:strata:docx')
def cmd_univariate_strata_docx() -> None:
    """T vs C Δ comparison within sex/age strata for all biomarkers → output/a1_univariate_strata.docx"""
    univariate_strata_docx()


@cli.command(name='univariate:il6')
def cmd_univariate_il6() -> None:
    """Two-part IL-6 analysis addressing left-censoring at LOD → stdout + output/a1_univariate_il6.docx"""
    univariate_il6()


@cli.command(name='ancova')
@click.option('--t0',      is_flag=True, help='Include T0 (baseline) as covariate')
@click.option('--sex',     is_flag=True, help='Include Sex as covariate')
@click.option('--age',     is_flag=True, help='Include Age as covariate')
@click.option('--docx',    is_flag=True, help='Also save results table as output/ancova[...].docx')
@click.option('--primary', is_flag=True, help='Group results by endpoint tier (primary/secondary/exploratory)')
def cmd_ancova(t0: bool, sex: bool, age: bool, docx: bool, primary: bool) -> None:
    """ANCOVA T1 ~ Treatment [+ T0] [+ Sex] [+ Age] for all biomarkers → stdout + output/plots_ancova[...]/"""
    if primary:
        ancova_primary(t0=t0, sex=sex, age=age, docx=docx)
    else:
        ancova(t0=t0, sex=sex, age=age, docx=docx)


@cli.command(name='pca')
def cmd_pca() -> None:
    """PCA on 20 standardized biomarker Δ values: variance, loadings, group separation → stdout"""
    pca_analysis()


@cli.command(name='composite-scores')
def cmd_composite_scores() -> None:
    """Three composite ageing scores (equal-weighted, PC1-weighted, ageing-specific) → stdout + output/a2_composite_scores.docx"""
    composite_scores()


@cli.command(name='mahalanobis')
@click.option('--corr-filter', 'corr_threshold', default=None, type=float,
              help='Drop one biomarker from each pair with |r| > THRESHOLD before computing D²')
def cmd_mahalanobis(corr_threshold: float | None) -> None:
    """Mahalanobis D² between group centroids with 5,000-permutation test → stdout + output/plots_mahalanobis/ + output/a2_mahalanobis.docx"""
    mahalanobis_analysis(corr_threshold=corr_threshold)


@cli.command(name='responders')
def cmd_responders() -> None:
    """Multi-domain responder classification (5 domains, strict/lenient) → stdout + output/a3_responders.docx + output/plots_responders/"""
    responders_classify()


@cli.command(name='responders:ml')
@click.option('--only', multiple=True, metavar='NAME',
              help='Restrict to named biomarker(s); repeat for multiple.')
def cmd_responders_ml(only: tuple[str, ...]) -> None:
    """ML classification Treatment vs Control (RF, GBM, LR, 5-fold CV, permutation test) → stdout + output/plots_responders_ml/"""
    responders_ml(only=only)


@cli.command(name='responders:ml2')
@click.option('--only', multiple=True, metavar='NAME',
              help='Restrict predictors to named biomarker(s)/Sex/Chronological Age; repeat for multiple.')
@click.option('--target-strict-responder',           is_flag=True, help='Predict strict responder (5/5 domains met)')
@click.option('--target-inflammation-responder',     is_flag=True, help='Predict Inflammation domain responder')
@click.option('--target-glycemic-health-responder',  is_flag=True, help='Predict Glycemic health domain responder')
@click.option('--target-lipid-profile-responder',    is_flag=True, help='Predict Lipid profile domain responder')
@click.option('--target-biological-ageing-responder',is_flag=True, help='Predict Biological ageing domain responder')
@click.option('--target-cellular-energy-responder',  is_flag=True, help='Predict Cellular energy domain responder')
@click.option('--test', is_flag=True, help='Train on 70 random treatment patients; evaluate on remaining treatment patients instead of controls')
def cmd_responders_ml2(
    only: tuple[str, ...],
    target_strict_responder: bool,
    target_inflammation_responder: bool,
    target_glycemic_health_responder: bool,
    target_lipid_profile_responder: bool,
    target_biological_ageing_responder: bool,
    target_cellular_energy_responder: bool,
    test: bool,
) -> None:
    """ML prediction from T0 biomarker values + sex + age (default: Biological Age improvement) → stdout + output/plots_responders_ml2[...]/"""
    # Map each flag to its Target; only one may be set
    flag_targets = [
        (target_strict_responder,           TARGET_STRICT),
        (target_inflammation_responder,     TARGET_INFLAMMATION),
        (target_glycemic_health_responder,  TARGET_GLYCEMIC),
        (target_lipid_profile_responder,    TARGET_LIPID),
        (target_biological_ageing_responder,TARGET_AGEING),
        (target_cellular_energy_responder,  TARGET_CELLULAR),
    ]
    selected = [t for flag, t in flag_targets if flag]
    if len(selected) > 1:
        raise click.UsageError('At most one --target-* flag may be specified.')
    target = selected[0] if selected else TARGET_BIOLOGICAL_AGE
    responders_ml2(only=only, target=target, test=test)


@cli.command(name='responders:bootstrap')
def cmd_responders_bootstrap() -> None:
    """Bootstrap 95% CIs for mean T−C Δ per biomarker (2,000 resamples) → stdout + output/a3_bootstrap_ci.docx + output/plots_bootstrap/"""
    responders_bootstrap()


@cli.command(name='within')
def cmd_within() -> None:
    """Within-group T0→T1 change test for all biomarkers across all subgroups → stdout"""
    within_change()


@cli.command(name='univariate:primary-75')
def cmd_univariate_primary_75() -> None:
    """Same as univariate:primary but restricted to rows 1–75 → stdout"""
    univariate_primary(max_row=75)


@cli.command(name='all')
def cmd_all() -> None:
    """Run all commands to reproduce the thesis Results section."""
    # Approach 1: Classical Biostatistical Analysis
    descriptive_docx()
    descriptive_plots('t0')
    descriptive_plots('delta')
    outliers_zscore()
    outliers_studentized()
    outliers_cooks()
    descriptive_plots_log('t0')
    descriptive_plots_log('delta')
    descriptive_plots_age_accel('t0')
    descriptive_plots_age_accel('delta')
    baseline_balance_biomarkers()
    baseline_balance_sex()
    baseline_balance_age()
    within_change()
    univariate_primary()
    univariate_strata_docx()
    univariate_il6()
    ancova_primary(t0=True, sex=False, age=False, docx=True)
    ancova_primary(t0=True, sex=True,  age=False, docx=True)
    ancova_primary(t0=True, sex=False, age=True,  docx=True)
    ancova_primary(t0=True, sex=True,  age=True,  docx=True)
    # Approach 2: Multivariate Systems Biology Analysis
    pca_analysis()
    composite_scores()
    mahalanobis_analysis(corr_threshold=0.9)
    # Approach 3: Machine Learning and Responder Analysis
    responders_classify()
    responders_ml2(
        only=(
            'NAD⁺', 'CRP', 'IL-6', 'Fibrinogen', 'Glucose', 'Insulin',
            'HOMA-IR', 'HbA1c', 'Triglycerides', 'HDL Cholesterol',
            'LDL Cholesterol', 'Total Cholesterol', 'Sex', 'Chronological Age',
        ),
        target=TARGET_STRICT,
        test=False,
    )


def main() -> None:
    cli()
