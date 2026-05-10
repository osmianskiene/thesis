"""Generate box plots, histograms, and Q-Q plots for each biomarker across strata."""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from biomarkers import AGE_ACCEL_BIOMARKERS, BIOMARKERS, LOG_BIOMARKERS
from data import load_composites, load_data, load_data_log_fold_change

# Okabe-Ito colorblind-friendly palette.
# Each stratum maps group key → hex colour.
PALETTE: dict[str, dict[str, str]] = {
    'Treatment': {'T': '#0072B2', 'C': '#E69F00'},                    # blue / orange
    'Sex':       {'M': '#0072B2', 'F': '#CC79A7'},                    # blue / reddish-purple
    'Age Group': {'Y': '#009E73', 'A': '#E69F00', 'O': '#D55E00'},    # bluish-green / orange / vermillion
}

# Ordered strata: (stratum column, {key: display label})
STRATA: list[tuple[str, dict[str, str]]] = [
    ('Treatment', {'T': 'Treatment', 'C': 'Control'}),
    ('Sex',       {'M': 'Male',      'F': 'Female'}),
    ('Age Group', {'Y': 'Young',     'A': 'Average', 'O': 'Old'}),
]


def _stratum_series(df: pd.DataFrame, col: str, stratum: str) -> list[tuple[str, pd.Series]]:
    """Return [(display_label, series), ...] for each group in a stratum."""
    _, label_map = next(s for s in STRATA if s[0] == stratum)

    if stratum == 'Treatment':
        raw = [('T', df[df['Treatment'] == 1][col]),
               ('C', df[df['Treatment'] == 0][col])]
    elif stratum == 'Sex':
        raw = [('M', df[df['Sex'] == 'm'][col]),
               ('F', df[df['Sex'] == 'f'][col])]
    else:  # Age Group
        raw = [('Y', df[df['Age Group'] == 'Young'][col]),
               ('A', df[df['Age Group'] == 'Average'][col]),
               ('O', df[df['Age Group'] == 'Old'][col])]

    return [(label_map[key], series.dropna()) for key, series in raw if series.dropna().size > 0]


COLOR_ALL = '#56B4E9'  # Okabe-Ito sky-blue; used for the ungrouped "All patients" column


def _draw_boxplot(ax: plt.Axes, groups: list[tuple[str, pd.Series]], colors: list[str]) -> None:
    data   = [g[1].values for g in groups]
    labels = [g[0]        for g in groups]

    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    medianprops={'color': 'black', 'linewidth': 1.5},
                    whiskerprops={'linewidth': 1},
                    capprops={'linewidth': 1},
                    flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.5})
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.tick_params(axis='both', labelsize=7)


def _draw_violinplot(ax: plt.Axes, groups: list[tuple[str, pd.Series]], colors: list[str]) -> None:
    data   = [g[1].values for g in groups]
    labels = [g[0]        for g in groups]

    # Draw violin bodies only; box elements are added manually below.
    parts = ax.violinplot(data, showmedians=False, showextrema=False)
    for patch, color in zip(parts['bodies'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Overlay a miniature box plot inside each violin (same style as the reference image):
    # thin whisker line (1.5×IQR, clipped to data), thick IQR bar, white median dot.
    for pos, arr in enumerate(data, start=1):  # violinplot uses 1-based x positions
        q1, median, q3 = np.percentile(arr, [25, 50, 75])
        iqr = q3 - q1
        # Whiskers extend to the most extreme point within 1.5×IQR of the hinges.
        whisker_lo = arr[arr >= q1 - 1.5 * iqr].min()
        whisker_hi = arr[arr <= q3 + 1.5 * iqr].max()
        ax.vlines(pos, whisker_lo, whisker_hi, color='black', linewidth=1, zorder=2)  # whisker stem
        ax.vlines(pos, q1, q3, color='black', linewidth=5, zorder=3)                 # IQR box
        ax.scatter(pos, median, color='white', s=15, zorder=4)                        # median dot

    ax.set_xticks(range(1, len(labels) + 1))  # violinplot uses 1-based positions
    ax.set_xticklabels(labels)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.tick_params(axis='both', labelsize=7)


def _draw_histogram(ax: plt.Axes, groups: list[tuple[str, pd.Series]], colors: list[str]) -> None:
    for (label, series), color in zip(groups, colors):
        ax.hist(series, bins=12, alpha=0.5, label=label, color=color, edgecolor='none')
    ax.legend(fontsize=6, loc='upper right')
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.set_ylabel('Count', fontsize=7)
    ax.tick_params(axis='both', labelsize=7)


def _draw_qqplot(ax: plt.Axes, groups: list[tuple[str, pd.Series]], colors: list[str]) -> None:
    """Q-Q plot vs. theoretical normal distribution; one scatter+line per group."""
    for (label, series), color in zip(groups, colors):
        # probplot returns theoretical quantiles, sample quantiles, and fit params.
        (osm, osr), (slope, intercept, _) = scipy_stats.probplot(series.values, dist='norm')
        ax.scatter(osm, osr, s=8, alpha=0.5, color=color, label=label, zorder=3)
        ax.plot([osm[0], osm[-1]],
                [osm[0] * slope + intercept, osm[-1] * slope + intercept],
                color=color, linewidth=1, zorder=2)
    ax.legend(fontsize=6, loc='upper left')
    ax.set_xlabel('Theoretical quantiles', fontsize=7)
    ax.set_ylabel('Sample quantiles', fontsize=7)
    ax.tick_params(axis='both', labelsize=7)


ZERO_LINE_ROWS = {0, 1}  # row indices (box, violin) that get a y=0 reference line on delta plots


def _fill_plot_columns(
    axes: np.ndarray, df: pd.DataFrame, col: str,
    col_offset: int, zero_line: bool, show_row_labels: bool,
) -> None:
    """Fill 4 columns of a subplot grid (col_offset … col_offset+3) for one data column."""
    row_labels = ['Box plot', 'Violin plot', 'Histogram', 'Q-Q plot']
    draw_fns   = [_draw_boxplot, _draw_violinplot, _draw_histogram, _draw_qqplot]

    # First of the 4 columns: all patients ungrouped.
    all_groups = [('All', df[col].dropna())]
    all_colors = [COLOR_ALL]
    for row_idx, (row_label, fn) in enumerate(zip(row_labels, draw_fns)):
        ax = axes[row_idx, col_offset]
        fn(ax, all_groups, all_colors)
        if zero_line and row_idx in ZERO_LINE_ROWS:
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
        if row_idx == 0:
            ax.set_title('All patients', fontsize=9, pad=4)
        if show_row_labels:
            current_ylabel = ax.get_ylabel()
            prefix = f'{row_label}\n' if not current_ylabel else f'{row_label} — '
            ax.set_ylabel(f'{prefix}{current_ylabel}', fontsize=7)

    # Remaining 3 columns: one per stratum.
    for i, (stratum, _) in enumerate(STRATA, start=1):
        groups = _stratum_series(df, col, stratum)
        colors = list(PALETTE[stratum].values())
        for row_idx, (_, fn) in enumerate(zip(row_labels, draw_fns)):
            ax = axes[row_idx, col_offset + i]
            fn(ax, groups, colors)
            if zero_line and row_idx in ZERO_LINE_ROWS:
                ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
            if row_idx == 0:
                ax.set_title(f'By {stratum}', fontsize=9, pad=4)


def _save_biomarker_png(
    out_dir: Path, df: pd.DataFrame, title: str, col: str, filename: str,
    zero_line: bool = False,
) -> None:
    """Save one 4×4 PNG: rows = box/violin/hist/qq, columns = all/treatment/sex/age stratum."""
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.0)
    _fill_plot_columns(axes, df, col, col_offset=0, zero_line=zero_line, show_row_labels=True)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def _save_biomarker_png_paired(
    out_dir: Path,
    df_orig: pd.DataFrame, col_orig: str, left_label: str,
    df_log: pd.DataFrame,  col_log: str,  right_label: str,
    title: str, filename: str,
    zero_line_left: bool = False,   # y=0 reference line on left 4 columns (box/violin rows)
    zero_line_right: bool = False,  # y=0 reference line on right 4 columns (box/violin rows)
) -> None:
    """Save one 4×8 PNG: left 4 columns = original scale, right 4 = log scale."""
    fig, axes = plt.subplots(4, 8, figsize=(32, 12))
    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.0)

    _fill_plot_columns(axes, df_orig, col_orig, col_offset=0,
                       zero_line=zero_line_left, show_row_labels=True)
    _fill_plot_columns(axes, df_log,  col_log,  col_offset=4,
                       zero_line=zero_line_right, show_row_labels=False)

    # tight_layout must run first so get_position() reflects final axis positions.
    fig.tight_layout()

    # Section header spanning each half, placed just above the top row.
    for col_span_start, label in ((0, left_label), (4, right_label)):
        ax0 = axes[0, col_span_start]
        ax3 = axes[0, col_span_start + 3]
        x0  = ax0.get_position().x0
        x1  = ax3.get_position().x1
        y_top = ax0.get_position().y1
        fig.text((x0 + x1) / 2, y_top + 0.01, label,
                 ha='center', va='bottom', fontsize=10, fontstyle='italic',
                 transform=fig.transFigure)

    # Vertical separator line between the two blocks.
    x_sep = (axes[0, 3].get_position().x1 + axes[0, 4].get_position().x0) / 2
    fig.add_artist(plt.Line2D([x_sep, x_sep], [0, 1],
                              transform=fig.transFigure,
                              color='gray', linewidth=0.8, linestyle='--'))
    path = out_dir / filename
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def descriptive_plots(measurement: str) -> None:
    """
    Generate box plots, histograms, and Q-Q plots for each biomarker.

    measurement: 't0' → baseline columns; 'delta' → change columns.
    """
    df       = load_data()
    suffix   = 'T0' if measurement == 't0' else 'Δ'
    out_dir  = Path(f'output/plots_{measurement}')
    zero_line = measurement == 'delta'  # mark y=0 on delta plots to show direction of change
    out_dir.mkdir(parents=True, exist_ok=True)

    # Chronological Age.
    _save_biomarker_png(out_dir, df,
                        title=f'Chronological Age {suffix}',
                        col=f'Chronological Age {suffix}',
                        filename='00_chronological_age.png',
                        zero_line=zero_line)

    for i, bm in enumerate(BIOMARKERS, start=1):
        col  = bm.t0_col if measurement == 't0' else bm.delta_col
        safe = bm.name.replace('⁺', '+').replace(' ', '_').replace('/', '-')
        _save_biomarker_png(out_dir, df,
                            title=f'{bm.name} — {suffix}',
                            col=col,
                            filename=f'{i:02d}_{safe}.png',
                            zero_line=zero_line)


def descriptive_plots_age_accel(measurement: str) -> None:
    """
    Generate paired distribution plots for all four age biomarkers.

    Left 4 columns: raw age on the original scale.  Right 4 columns: age acceleration residuals.
    measurement: 't0' → baseline (raw T0 vs. AgeAccel T0); 'delta' → change (Δ vs. ΔAgeAccel).
    """
    df_raw       = load_data()
    df_accel     = load_composites()
    orig_suffix  = 'T0' if measurement == 't0' else 'Δ'
    accel_suffix = 'AgeAccel T0' if measurement == 't0' else 'AgeAccel Δ'
    out_dir      = Path(f'output/plots_age_accel_{measurement}')
    # Age acceleration residuals are centred at 0, so right columns always get a y=0 line.
    # Left columns (raw age) only get one for delta plots.
    zero_line_left  = measurement == 'delta'
    zero_line_right = True
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, bm in enumerate(AGE_ACCEL_BIOMARKERS, start=1):
        # Raw column: look up the original-scale column from BIOMARKERS.
        orig_bm   = next(b for b in BIOMARKERS if b.name == bm.name)
        col_raw   = orig_bm.t0_col if measurement == 't0' else orig_bm.delta_col
        col_accel = bm.t0_col      if measurement == 't0' else bm.delta_col
        safe      = bm.name.replace(' ', '_')
        _save_biomarker_png_paired(
            out_dir,
            df_orig=df_raw,   col_orig=col_raw,   left_label=f'Raw age — {orig_suffix}',
            df_log=df_accel,  col_log=col_accel,  right_label=f'Age acceleration — {accel_suffix}',
            title=f'{bm.name}',
            filename=f'{i:02d}_{safe}_age_accel.png',
            zero_line_left=zero_line_left,
            zero_line_right=zero_line_right,
        )


def descriptive_plots_log(measurement: str) -> None:
    """
    Generate paired distribution plots for CRP, Insulin, and HOMA-IR.

    Left 4 columns: original scale.  Right 4 columns: natural-log scale.
    measurement: 't0' → baseline (T0 vs ln(T0)); 'delta' → change (Δ vs ln(T1/T0)).
    """
    df_orig   = load_data()
    df_log    = load_data_log_fold_change()
    orig_suffix = 'T0' if measurement == 't0' else 'Δ'
    log_suffix  = 'ln(T0)' if measurement == 't0' else 'ln(T1/T0)'
    out_dir   = Path(f'output/plots_log_{measurement}')
    zero_line = measurement == 'delta'  # mark y=0 to show direction of change
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, bm in enumerate(LOG_BIOMARKERS, start=1):
        # Look up the original-scale column for this biomarker in BIOMARKERS.
        orig_bm  = next(b for b in BIOMARKERS if b.name == bm.name)
        col_orig = orig_bm.t0_col  if measurement == 't0' else orig_bm.delta_col
        col_log  = bm.t0_col       if measurement == 't0' else bm.delta_col
        safe     = bm.name.replace('⁺', '+').replace(' ', '_').replace('/', '-')
        _save_biomarker_png_paired(
            out_dir,
            df_orig=df_orig, col_orig=col_orig, left_label=f'Original — {orig_suffix}',
            df_log=df_log,   col_log=col_log,   right_label=f'Log scale — {log_suffix}',
            title=f'{bm.name}',
            filename=f'{i:02d}_{safe}_log.png',
            zero_line_left=zero_line,
            zero_line_right=zero_line,
        )
