# Predictive Modeling of Nicotine Vaping Among Students: A Longitudinal Analysis (2017-2023)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![R 4.0+](https://img.shields.io/badge/R-4.0+-blue.svg)](https://www.r-project.org/)

## Overview

This repository contains the complete computational pipeline for analyzing nicotine vaping trends among students using longitudinal survey data spanning 2017-2023. The analysis employs machine learning classifiers to predict nicotine vaping usage based on survey responses and investigates key risk factors using interpretable AI techniques.

**Note**: This code accompanies a manuscript submitted to the *Journal of Quantitative Criminology*.

---

## Project Status

**Current Version**: 2.1 (November 2025)
**Status**: Comprehensive revisions complete + **🆕 Interactive Jupyter Notebook Workflow**

### 🎉 What's New in Version 2.1

**Major Update**: The entire analysis pipeline has been reorganized for interactive execution:

- ✅ **All 19 analyses** now available as Jupyter notebooks (.ipynb)
- ✅ **Master interactive guide** (`00_MASTER_INTERACTIVE_WORKFLOW.ipynb`)
- ✅ **Cell-by-cell execution** for step-by-step understanding
- ✅ **Better for teaching**, peer review, and exploratory analysis
- ✅ **Legacy scripts** moved to `retired/` folder with documentation

**Workflow Type**: Interactive notebooks (recommended) + Automated shell scripts (still available)

This repository has undergone extensive methodological enhancements:

### Phase 1: Initial Development (v0.5)
- Core ML pipeline with 6 classifiers
- SHAP-based feature importance analysis
- Basic exploratory analysis and visualization

### Phase 2: JQC Initial Revisions (v1.0)
- ✅ Survey weights implementation (preserving MTF sampling design)
- ✅ Temporal validation (train 2017-2021, test 2022-2023)
- ✅ Baseline comparisons (framework vs. simpler approaches)
- ✅ Comprehensive robustness checks
- ✅ Statistical vs. practical significance analysis
- ✅ Strengthened criminological framing

### Phase 3: Advanced Methodological Enhancements (v2.0)
**13 Critical Improvements Implemented** (see [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)):

**Methodological Rigor**:
1. Interaction terms in regression (validating ML discoveries)
2. Multiple imputation using MICE (proper uncertainty quantification)
3. Structural break analysis (COVID-19 threshold testing)
4. Model calibration assessment
5. SHAP stability analysis (bootstrap validation)
6. SHAP dependence plots (interaction visualization)
7. Incremental predictive value testing

**Substantive Criminological Analyses**:
8. Gender-specific substance use pathways
9. Racial/ethnic disparities in vaping trends
10. Regional variation analysis

**Manuscript Enhancements**:
11. Improved abstract (more specific methods and findings)
12. New section on interaction testing in regression
13. Enhanced limitations section with formal remedies

For detailed implementation notes, see:
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Complete technical details of all 13 enhancements
- **[REVISION_NOTES.md](REVISION_NOTES.md)** - JQC revision documentation

---

## Table of Contents

- [Project Status](#project-status)
- [Repository Structure](#repository-structure)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Data Availability](#data-availability)
- [Reproducibility Instructions](#reproducibility-instructions)
- [Code Availability Statement](#code-availability-statement)
- [Computational Pipeline](#computational-pipeline)
- [Key Findings](#key-findings)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Repository Structure

```
vaping_project/
├── README.md                          # This file - Project overview
├── IMPLEMENTATION_SUMMARY.md          # Summary of 13 methodological enhancements
├── REVISION_NOTES.md                  # JQC revision implementation notes
├── LICENSE                            # MIT License
├── CITATION.cff                       # Citation metadata
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment specification
├── run_all.sh                         # Master script for base analyses
├── run_revision_analyses.sh           # Master script for all revision analyses
│
├── scripts/                           # R preprocessing scripts
│   ├── 01_importing_data.R           # Data loading from raw TSV files
│   ├── 02_preprocessing.R            # Feature engineering & cleaning
│   ├── 02b_preprocessing_with_weights.R  # Preprocessing preserving survey weights
│   ├── 03_EDA.R                      # Exploratory data analysis
│   └── 03_mice_imputation.R          # Multiple imputation using MICE
│
├── src/                               # Python source modules
│   ├── __init__.py
│   ├── data_processing.py            # Data loading and preprocessing utilities
│   ├── modeling.py                   # Machine learning model implementations
│   ├── interpretability.py           # SHAP and feature importance analysis
│   └── visualization.py              # Plotting functions
│
├── notebooks/                         # 🎯 Interactive Jupyter Notebooks (ALL ANALYSES)
│   ├── 00_MASTER_INTERACTIVE_WORKFLOW.ipynb  # 🌟 START HERE - Interactive guide
│   │
│   ├── # Core Analysis Pipeline (01-05)
│   ├── 01_preprocessing_12.ipynb     # Python preprocessing (12th grade)
│   ├── 02_preprocessing_08_10.ipynb  # Python preprocessing (8th-10th grade)
│   ├── 03_modelling.ipynb            # 🔥 Main ML pipeline & model comparison
│   ├── 04_regression.ipynb           # Regression analysis
│   ├── 05_charts.ipynb               # Publication-ready visualizations
│   │
│   ├── # JQC Revision Analyses (06-09)
│   ├── 06_weighted_regression.ipynb  # Survey-weighted regression
│   ├── 07_temporal_validation.ipynb  # Temporal validation (2017-2021 → 2022-2023)
│   ├── 08_baseline_comparisons.ipynb # Framework validation vs. alternatives
│   ├── 09_robustness_checks.ipynb    # Robustness to methodological choices
│   │
│   ├── # Advanced Methodological Analyses (10-13, 17-19)
│   ├── 10_interaction_regression.ipynb  # Interaction terms in regression
│   ├── 11_regression_with_mice.ipynb    # Regression with multiple imputation
│   ├── 12_structural_break_analysis.ipynb  # COVID-19 structural break tests
│   ├── 13_model_calibration.ipynb       # Model calibration analysis
│   ├── 17_shap_stability.ipynb          # SHAP feature importance stability
│   ├── 18_shap_dependence.ipynb         # SHAP interaction visualizations
│   ├── 19_incremental_value.ipynb       # Incremental predictive value analysis
│   │
│   └── # Substantive Criminological Analyses (14-16)
│       ├── 14_gender_interactions.ipynb     # Gender-specific pathways
│       ├── 15_racial_disparities.ipynb      # Racial/ethnic disparities
│       └── 16_regional_analysis.ipynb       # Regional variation analysis
│
├── retired/                           # Legacy Python scripts (converted to notebooks)
│   └── README.md                     # Documentation of retired scripts
│
├── docs/                              # Documentation & Manuscript
│   ├── DATA_AVAILABILITY.md          # Data access instructions
│   ├── CODE_AVAILABILITY.md          # Code sharing statement
│   ├── COMPUTATIONAL_NOTES.md        # Technical details
│   └── main.tex                      # LaTeX manuscript for JQC
│
├── outputs/                           # Generated analysis outputs
│   ├── models/                       # Trained model artifacts (.joblib, .pkl)
│   ├── predictions/                  # Model predictions and metrics
│   └── tables/                       # Summary statistics tables
│
└── figures/                           # Publication figures
    ├── shap_summary_plot.png
    ├── aggregated_shap_importance.png
    ├── top_20_aggregated_shap_importance.png
    └── [Additional figures generated by analyses 06-19]
```

---

## System Requirements

### Hardware
- **RAM**: Minimum 16 GB (32 GB recommended for full dataset)
- **Storage**: 10 GB free disk space
- **CPU**: Multi-core processor (parallel processing used in R and Python)
- **OS**: Linux, macOS, or Windows with WSL

### Software
- **R**: Version 4.0 or higher
- **Python**: Version 3.8 or higher
- **Git**: For version control
- **Conda** (optional but recommended): For environment management

---

## Installation

### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/siyangni/vaping_project.git
cd vaping_project

# Create conda environment
conda env create -f environment.yml
conda activate vaping_env

# Install R packages (run in R console)
Rscript -e "install.packages('pacman')"
Rscript -e "pacman::p_load(tidyverse, here, janitor, skimr, purrr, caret, gplots, pheatmap, glmnet, randomForest, doParallel)"
```

### Option 2: Using pip + R

```bash
# Clone repository
git clone https://github.com/siyangni/vaping_project.git
cd vaping_project

# Create Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install R packages (run in R console)
Rscript -e "install.packages(c('tidyverse', 'here', 'janitor', 'skimr', 'purrr', 'caret', 'gplots', 'pheatmap', 'glmnet', 'randomForest', 'doParallel'))"
```

---

## Data Availability

The analysis uses data from the **Monitoring the Future (MTF)** study, a nationally representative survey of American adolescents conducted by the University of Michigan's Institute for Social Research.

### Data Access

The MTF data are publicly available with restrictions:

1. **Public Use Files**: Visit [https://www.icpsr.umich.edu/web/ICPSR/series/35](https://www.icpsr.umich.edu/web/ICPSR/series/35)
2. **Registration**: Create a free account with ICPSR (Inter-university Consortium for Political and Social Research)
3. **Download**: Access survey data for years 2017-2023, grade levels 8, 10, and 12
4. **Data Use Agreement**: Researchers must agree to ICPSR's Terms of Use

### Expected Data Structure

After downloading, place TSV files in `~/work/vaping_project_data/original_all_core/` with the following naming convention:

```
original_core_2017_0012.tsv  # 2017, 12th grade
original_core_2017_0810.tsv  # 2017, 8th-10th grade
original_core_2018_0012.tsv
original_core_2018_0810.tsv
...
original_core_2023_0012.tsv
original_core_2023_0810.tsv
```

**Data not included in repository**: Raw survey data files are not included due to ICPSR redistribution restrictions. Researchers must obtain data independently.

---

## Reproducibility Instructions

### 🆕 Interactive Workflow (RECOMMENDED)

**Version 2.0** now provides a fully interactive Jupyter notebook workflow for easier exploration, teaching, and reproducibility.

#### Quick Start: Interactive Analysis

```bash
# 1. Ensure data is prepared (run R preprocessing first)
Rscript scripts/01_importing_data.R
Rscript scripts/02_preprocessing.R

# 2. Launch Jupyter
jupyter notebook notebooks/

# 3. Open 00_MASTER_INTERACTIVE_WORKFLOW.ipynb
# This master notebook guides you through all 19 analyses
```

**Benefits of Interactive Workflow**:
- ✅ Run analyses cell-by-cell for step-by-step understanding
- ✅ Modify parameters and immediately see results
- ✅ Add custom visualizations and exploratory analysis
- ✅ Better for teaching, presentations, and peer review
- ✅ All analyses now in Jupyter notebooks (no separate scripts)

---

### Complete Analysis Pipeline (Automated)

For batch execution, the project provides two master scripts:

#### Option 1: Base Analysis Pipeline

To reproduce the core ML analysis (notebooks 01-05):

```bash
# Ensure data files are in ~/work/vaping_project_data/original_all_core/
bash run_all.sh
```

This script executes:
1. R preprocessing (data loading, feature engineering, cleaning)
2. Python modeling (6 ML classifiers with hyperparameter tuning)
3. Interpretability analysis (SHAP values, feature importance)
4. Visualization generation (publication figures)

**Expected runtime**: 2-6 hours

#### Option 2: Complete Revision Analysis (Recommended)

To reproduce ALL analyses including JQC revisions and enhancements (notebooks 06-19):

```bash
# Ensure processed data exists (run run_all.sh first if needed)
bash run_revision_analyses.sh
```

This comprehensive script executes:
1. Survey-weighted regression analysis
2. Temporal validation (2017-2021 → 2022-2023)
3. Baseline comparisons (framework validation)
4. Robustness checks (imputation, thresholds, splits)
5. Interaction regression (Wave × Substance interactions)
6. MICE multiple imputation
7. Structural break analysis (COVID-19 testing)
8. Model calibration analysis
9. Gender interaction analysis
10. Racial disparity analysis
11. Regional variation analysis
12. SHAP stability analysis
13. SHAP dependence plots
14. Incremental value analysis

**Expected runtime**: 4-8 hours total

#### Option 3: Run Individual Analyses Interactively

To run specific analyses, open notebooks in Jupyter:

```bash
# Launch Jupyter
jupyter notebook notebooks/

# Then open any notebook:
# - 10_interaction_regression.ipynb (Critical methodological fixes)
# - 11_regression_with_mice.ipynb (Multiple imputation)
# - 12_structural_break_analysis.ipynb (COVID-19 structural break)
# - 13_model_calibration.ipynb (Model calibration)
# - 14_gender_interactions.ipynb (Gender pathways)
# - 15_racial_disparities.ipynb (Racial/ethnic trends)
# - 16_regional_analysis.ipynb (Regional variation)
# - 17_shap_stability.ipynb (SHAP stability)
# - 18_shap_dependence.ipynb (SHAP interactions)
# - 19_incremental_value.ipynb (Nested models)
```

**Note**: Legacy Python scripts (.py files) have been moved to `retired/` folder.
All analyses are now available as interactive Jupyter notebooks (.ipynb).

### Step-by-Step Manual Execution

#### Step 1: Data Preprocessing (R)

```bash
# Load raw data
Rscript scripts/01_importing_data.R

# Preprocess and engineer features
Rscript scripts/02_preprocessing.R

# Exploratory data analysis (optional)
Rscript scripts/03_EDA.R
```

**Outputs**:
- `~/work/vaping_project_data/processed_data_g12.csv`
- `~/work/vaping_project_data/processed_data_g12n.csv` (final cleaned data)

#### Step 2: Machine Learning Modeling (Python - Interactive)

```bash
# Activate environment
conda activate vaping_env  # or: source venv/bin/activate

# Launch Jupyter and run notebooks interactively
jupyter notebook notebooks/

# Recommended order:
# 1. 03_modelling.ipynb - Main ML analysis
# 2. 04_regression.ipynb - Regression analysis
# 3. 05_charts.ipynb - Visualizations
# 4. 06-19 (any order based on research questions)
```

**Key configurations** (set in `notebooks/03_modelling.ipynb`):
- `RANDOM_STATE = 42` (reproducibility seed)
- `TEST_SIZE = 0.2` (80/20 train-test split)
- `CV_FOLDS = 5` (cross-validation folds)

**Models trained**:
1. Logistic Regression (Lasso regularization)
2. Random Forest Classifier
3. Gradient Boosting Classifier
4. Histogram-based Gradient Boosting
5. XGBoost Classifier
6. CatBoost Classifier

**Outputs**:
- `outputs/models/*.joblib` (trained models)
- `outputs/predictions/*.csv` (test set predictions)
- `outputs/tables/model_comparison.csv` (performance metrics)

#### Step 3: Model Interpretation & Visualization

```bash
# Generate SHAP values and feature importance
jupyter notebook notebooks/03_modelling.ipynb  # Sections 8-9

# Create publication figures
jupyter notebook notebooks/05_charts.ipynb
```

**Outputs**:
- `figures/*.png` (SHAP plots, feature importance charts)

---

## Code Availability Statement

All code necessary to reproduce the analyses presented in the manuscript is provided in this repository. The code is shared under the MIT License, allowing free use, modification, and distribution with attribution.

### Version Control

This repository is version-controlled using Git and hosted on GitHub:
- **Repository**: [https://github.com/siyangni/vaping_project](https://github.com/siyangni/vaping_project)
- **DOI**: [Add Zenodo DOI after archiving]

### Long-Term Archiving

Upon manuscript acceptance, this repository will be archived on **Zenodo** with a permanent DOI for long-term preservation and citation.

### Reproducibility Guarantee

- **Random Seeds**: All random processes use `RANDOM_STATE = 42`
- **Dependencies**: Exact package versions specified in `requirements.txt` and `environment.yml`
- **Hardware**: Results obtained on Linux system; cross-platform tested on macOS and Windows
- **Computational Time**: ~3 hours on 16-core 3.2 GHz CPU with 32 GB RAM

---

## Computational Pipeline

### Overview

The analysis pipeline consists of four main stages:

```
Raw TSV Data (14 files, 2017-2023)
    ↓
[R] Data Loading & Preprocessing
    ├─ Feature engineering (grade, wave, demographics)
    ├─ Missing data recoding
    ├─ Feature selection (correlation, missingness)
    └─ Target variable standardization
    ↓
Cleaned CSV Data (~100 features, 50K+ observations)
    ↓
[Python] Machine Learning Pipeline
    ├─ Train-test split (stratified 80/20)
    ├─ Preprocessing (scaling, encoding, imputation)
    ├─ Model training (6 classifiers)
    ├─ Hyperparameter tuning (GridSearchCV)
    └─ Cross-validation (5-fold stratified)
    ↓
Model Evaluation & Interpretation
    ├─ Performance metrics (ROC-AUC, F1, precision, recall)
    ├─ SHAP values (feature importance)
    ├─ Permutation importance
    ├─ Partial dependence plots
    └─ 3-way feature interactions
    ↓
Publication Outputs
    ├─ Trained models (.joblib)
    ├─ Performance tables (.csv)
    └─ Figures (.png)
```

### Key Methodological Decisions

1. **Target Variable**: Binary classification of nicotine vaping (yes/no)
2. **Feature Selection**: Removed 50+ features based on:
   - High missingness (>70%)
   - High correlation (r > 0.5)
   - Information leakage (vaping-specific questions)
3. **Class Imbalance**: Stratified sampling ensures balanced representation
4. **Missing Data**: SimpleImputer (median strategy) + MissingIndicator flags
5. **Evaluation Metric**: ROC-AUC (primary), F1-score (secondary)
6. **Feature Importance**: SHAP values (model-agnostic interpretation)

---

## Key Findings

### Methodological Contributions

1. **Multi-Model Consensus Framework**: Achieves superior performance (AUC ≈ 0.92) compared to single-model approaches while protecting against model-specific artifacts
2. **Temporal Stability**: Framework shows excellent generalization across time periods (3-4% AUC degradation from 2017-2021 to 2022-2023)
3. **Robustness Validation**: Results stable across multiple methodological choices (imputation strategy, consensus threshold, train-test splits)
4. **Calibration Assessment**: Tree-based models achieve high discrimination but require recalibration for risk scoring applications

### Substantive Findings

**Temporal Dynamics**:
- **Dramatic COVID-19 Effect**: Survey wave is the strongest predictor (OR = 1.64), with structural break analysis confirming significant regime shift in 2020-2021
- Vaping prevalence doubled during pandemic period, suggesting natural experiment for studying substance use under reduced supervision

**Counterintuitive Polysubstance Patterns**:
- **Marijuana Protective Effect**: Marijuana use associated with 10% lower vaping odds (OR = 0.90), challenging traditional gateway theory
- **Alcohol Protective Effect**: Alcohol use shows 6% lower vaping odds (OR = 0.94)
- Suggests substance substitution rather than complementarity, or reflects changing substance hierarchies among youth

**Demographic Disparities**:
- **Gender Differences**: Substance use effects vary by gender (stronger marijuana protective effect among females)
- **Racial Disparities**: Significant variation across racial/ethnic groups with evolving trends over time
- **Regional Variation**: Geographic differences persist, potentially reflecting state policy variation

**Effect Size Hierarchy**:
- **Large effects** (>15% change): Temporal trends, developmental trajectories
- **Medium effects** (6-15% change): Substance use patterns, peer influences
- **Small effects** (1-5% change): Demographic factors, geographic variation
- **Trivial effects** (<1% change): Some lower-tier consensus features (statistically significant but substantively negligible)

### Theoretical Implications

Results engage multiple criminological frameworks:
- **Social Control Theory**: Reduced supervision during COVID-19
- **General Strain Theory**: Pandemic stress and coping mechanisms
- **Social Learning Theory**: Changing peer norm diffusion
- **Developmental Criminology**: Age-graded trajectories in vaping adoption
- **Problem Behavior Theory**: Unexpected substance substitution patterns

For complete details, see manuscript in [docs/main.tex](docs/main.tex)

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{ni2025vaping,
  author       = {Ni, Siyang},
  title        = {Predictive Modeling of Nicotine Vaping Among Students},
  year         = 2025,
  publisher    = {GitHub},
  url          = {https://github.com/siyangni/vaping_project},
  note         = {Submitted to Journal of Quantitative Criminology}
}
```

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Third-Party Data

The Monitoring the Future (MTF) survey data are subject to ICPSR's Terms of Use. Users must independently obtain data access and comply with all data use restrictions.

---

## Contact

**Principal Investigator**: Siyang Ni

For questions about the code or analysis:
- **GitHub Issues**: [https://github.com/siyangni/vaping_project/issues](https://github.com/siyangni/vaping_project/issues)
- **Email**: [Add email if desired]

For questions about the MTF data:
- **MTF Website**: [https://monitoringthefuture.org/](https://monitoringthefuture.org/)
- **ICPSR Help**: [https://www.icpsr.umich.edu/web/pages/support/](https://www.icpsr.umich.edu/web/pages/support/)

---

## Acknowledgments

This research uses data from the Monitoring the Future study, which is funded by the National Institute on Drug Abuse (NIDA) and conducted by the University of Michigan's Institute for Social Research. The authors thank the MTF team and survey participants.

---

**Last Updated**: 2025-11-07
**Repository Version**: 2.0.0
**Analysis Status**: All 13 methodological enhancements implemented
**Manuscript Status**: Ready for JQC submission
