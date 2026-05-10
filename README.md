# Thesis

This repository accompanies my master thesis in systems biology named "THE COMBINED EFFECT OF NAD+, RESVERATROL, BERBERINE, QUERCETIN, AND FISETIN ON LONGEVITY". 

It reads patient data from a CSV file and performs multiple analyses including descriptive statistics, baseline checks, ourlier analysis, pre-post analysis, univariate analysis, ANCOVA, PSA, composite scores, Mahalanobis distance, machine learning classification.

## Installation

1. Install Python 3.11 or newer.
2. Install [`uv` package manager](https://docs.astral.sh/uv/getting-started/installation/). On MacOS, Linux and WSL2, run its installation script:

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

3. Clone this repository:

    ```bash
    git clone https://github.com/osmianskiene/thesis.git
    ```

4. Install used packages:

    ```bash
    cd thesis

    uv sync
    ```

## Usage

This repository has a command for each analysis. Before running any command, change directory to the repository:

```bash
cd thesis
```

List all commands using the following command:

```bash
uv run thesis
```

For example, to run univariate analysis, use the following command:

```bash
uv run thesis univariate
```

Some commands have additional parameters. List the parameters of a command by adding `--help` to it. For example:

```bash
uv run thesis ancova --help
```

Provide parameters after the command name, for example:

```bash
uv run thesis ancova --t0
```

## Reproducing all analyses

To run all analyses at once (time consuming), use the `all` command:

```bash
uv run thesis all
```

The analysis results are provided as command output and files in the `output/` directory.

Alternatively, run the following individual commands to perform each analysis separately.

### Approach 1: Classical biostatistics

```bash
# Descriptive statistics
uv run thesis descriptive:docx

# Box plots, violin plots, histograms, and Q-Q plots
uv run thesis descriptive:plots:t0
uv run thesis descriptive:plots:delta

# Outlier analysis
uv run thesis outliers:zscore
uv run thesis outliers:studentized
uv run thesis outliers:cooks

# Log transformation of skewed biomerkers
uv run thesis descriptive:plots:log:t0
uv run thesis descriptive:plots:log:delta

# Age acceleration residuals
uv run thesis descriptive:plots:age-accel:t0
uv run thesis descriptive:plots:age-accel:delta

# Baseline balance checks
uv run thesis baseline:balance:biomarkers
uv run thesis baseline:balance:sex
uv run thesis baseline:balance:age

# Pre-post analysis
uv run thesis within

# Univariate analysis
uv run thesis univariate:primary

# Univariate analysis strata by sex and age groups
uv run thesis univariate:strata:docx

# Two-part IL-6 analysis that addresses left-censoring
uv run thesis univariate:il6

# ANCOVA
uv run thesis ancova --t0 --primary --docx

# ANCOVA sensitivity analyses
uv run thesis ancova --t0 --sex --primary --docx
uv run thesis ancova --t0 --age --primary --docx
uv run thesis ancova --t0 --sex --age --primary --docx
```

### Approach 2: Multivariate Systems Biology Analysis

```bash
# PCA and multivariate group separation
uv run thesis pca

# Composite aging scores
uv run thesis composite-scores

# Mahalanobis distance, excluding highly correlated biomarkers (|r| > 0.9)
uv run thesis mahalanobis --corr-filter 0.9
```

### Approach 3: Machine Learning and Responder Analysis

```bash
# Multi-domain responder classification
uv run thesis responders

# Machine learning classification (12 blood biomarkers + sex + age → strict responder)
uv run thesis responders:ml2 \
  --target-strict-responder \
  --only 'NAD⁺' \
  --only 'CRP' \
  --only 'IL-6' \
  --only 'Fibrinogen' \
  --only 'Glucose' \
  --only 'Insulin' \
  --only 'HOMA-IR' \
  --only 'HbA1c' \
  --only 'Triglycerides' \
  --only 'HDL Cholesterol' \
  --only 'LDL Cholesterol' \
  --only 'Total Cholesterol' \
  --only 'Sex' \
  --only 'Chronological Age'
```

## Dataset

All analyses are performed against the same dataset, located in the `data/patient_data.csv` in this repository.

