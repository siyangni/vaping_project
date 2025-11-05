# Predictive Modeling of Nicotine Vaping Among Students: A Longitudinal Analysis (2017-2023)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![R 4.0+](https://img.shields.io/badge/R-4.0+-blue.svg)](https://www.r-project.org/)

## Overview

This repository contains the complete computational pipeline for analyzing nicotine vaping trends among students using longitudinal survey data spanning 2017-2023. The analysis employs machine learning classifiers to predict nicotine vaping usage based on survey responses and investigates key risk factors using interpretable AI techniques.

**Note**: This code accompanies a manuscript submitted to the *Journal of Quantitative Criminology*.

---

## Table of Contents

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
├── README.md                          # This file
├── LICENSE                            # MIT License
├── CITATION.cff                       # Citation metadata
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment specification
├── run_all.sh                         # Master script to reproduce all analyses
│
├── scripts/                           # R preprocessing scripts
│   ├── 01_importing_data.R           # Data loading from raw TSV files
│   ├── 02_preprocessing.R            # Feature engineering & cleaning
│   └── 03_EDA.R                      # Exploratory data analysis
│
├── src/                               # Python source modules
│   ├── __init__.py
│   ├── data_processing.py            # Data loading and preprocessing utilities
│   ├── modeling.py                   # Machine learning model implementations
│   ├── interpretability.py           # SHAP and feature importance analysis
│   └── visualization.py              # Plotting functions
│
├── notebooks/                         # Jupyter notebooks (analysis & visualization)
│   ├── 01_preprocessing_12.ipynb     # Python preprocessing (12th grade)
│   ├── 02_preprocessing_08_10.ipynb  # Python preprocessing (8th-10th grade)
│   ├── 03_modelling.ipynb            # Main ML pipeline & model comparison
│   ├── 04_regression.ipynb           # Regression analysis
│   └── 05_charts.ipynb               # Publication-ready visualizations
│
├── docs/                              # Documentation
│   ├── DATA_AVAILABILITY.md          # Data access instructions
│   ├── CODE_AVAILABILITY.md          # Code sharing statement
│   └── COMPUTATIONAL_NOTES.md        # Technical details
│
├── outputs/                           # Generated analysis outputs
│   ├── models/                       # Trained model artifacts (.joblib, .pkl)
│   ├── predictions/                  # Model predictions and metrics
│   └── tables/                       # Summary statistics tables
│
└── figures/                           # Publication figures
    ├── shap_summary_plot.png
    ├── aggregated_shap_importance.png
    └── top_20_aggregated_shap_importance.png
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

### Complete Analysis Pipeline (Automated)

To reproduce all results from raw data to final figures:

```bash
# Ensure data files are in ~/work/vaping_project_data/original_all_core/
# Then run:
bash run_all.sh
```

This master script executes the entire pipeline sequentially:
1. R preprocessing (data loading, feature engineering, cleaning)
2. Python modeling (6 ML classifiers with hyperparameter tuning)
3. Interpretability analysis (SHAP values, feature importance)
4. Visualization generation (publication figures)

**Expected runtime**: 2-6 hours depending on hardware (primarily hyperparameter tuning)

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

#### Step 2: Machine Learning Modeling (Python)

```bash
# Activate environment
conda activate vaping_env  # or: source venv/bin/activate

# Run notebooks in order
jupyter notebook notebooks/03_modelling.ipynb
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

(To be completed after manuscript acceptance)

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

**Last Updated**: 2025-11-05
**Repository Version**: 1.0.0
