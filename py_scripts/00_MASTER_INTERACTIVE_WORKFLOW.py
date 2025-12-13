#!/usr/bin/env python
# coding: utf-8

# # 🎯 Master Interactive Workflow: Adolescent Vaping Analysis
# 
# ## Project Overview
# 
# This master notebook provides an interactive guide to the complete analysis pipeline for:
# > **Machine Learning Prediction of Adolescent Nicotine Vaping from Population-Based Survey Data**
# 
# ### Research Question
# Can we identify early warning indicators of adolescent nicotine vaping using machine learning models trained on large-scale survey data?
# 
# ### Data Source
# - **Monitoring the Future (MTF)**: National survey of 8th, 10th, and 12th graders
# - **Time period**: 2017-2023
# - **Sample size**: ~50,000 adolescents
# - **Features**: ~100 covariates (substance use, demographics, attitudes, behaviors)
# 
# ---
# 
# ## 📚 Complete Analysis Pipeline
# 
# This workflow consists of **19 interactive notebooks** organized into 3 phases:
# 
# ### Phase 1: Core Analysis (Notebooks 01-05)
# 1. **01_preprocessing_12.ipynb**: Data preprocessing for 12th grade
# 2. **02_preprocessing_08_10.ipynb**: Data preprocessing for 8th-10th grades
# 3. **03_modelling.ipynb**: 🔥 **MAIN ANALYSIS** - 6-classifier ML pipeline with SHAP
# 4. **04_regression.ipynb**: Logistic regression analysis
# 5. **05_charts.ipynb**: Publication-ready visualizations
# 
# ### Phase 2: Validation & Robustness (Notebooks 06-09)
# 6. **06_weighted_regression.ipynb**: Survey-weighted regression
# 7. **07_temporal_validation.ipynb**: Train on 2017-2021, test on 2022-2023
# 8. **08_baseline_comparisons.ipynb**: Compare to alternative approaches
# 9. **09_robustness_checks.ipynb**: Sensitivity analyses
# 
# ### Phase 3: Advanced Extensions (Notebooks 10-19)
# 10. **10_interaction_regression.ipynb**: Test SHAP-identified interactions
# 11. **11_regression_with_mice.ipynb**: Multiple imputation analysis
# 12. **12_structural_break_analysis.ipynb**: COVID-19 threshold testing
# 13. **13_model_calibration.ipynb**: Calibration assessment
# 14. **14_gender_interactions.ipynb**: Gender-stratified analysis
# 15. **15_racial_disparities.ipynb**: Racial/ethnic trends
# 16. **16_regional_analysis.ipynb**: Geographic variation
# 17. **17_shap_stability.ipynb**: Bootstrap SHAP stability
# 18. **18_shap_dependence.ipynb**: Interaction visualization
# 19. **19_incremental_value.ipynb**: Nested model comparison
# 
# ---

# ## ⚙️ Setup: Prerequisites
# 
# ### 1. Data Location
# Ensure your processed data file is available at:
# ```
# ~/work/vaping_project_data/processed_data_g12n.csv
# ```
# 
# ### 2. R Preprocessing (Required First)
# Before running these notebooks, you must run the R preprocessing scripts:
# ```bash
# cd scripts/
# Rscript 01_importing_data.R
# Rscript 02_preprocessing.R
# ```
# 
# ### 3. Python Environment
# Install required packages:
# ```bash
# pip install -r requirements.txt
# ```
# 
# Or use conda:
# ```bash
# conda env create -f environment.yml
# conda activate vaping
# ```

# ## 🚀 Quick Start Guide
# 
# ### Option 1: Run Full Pipeline (3-5 hours)
# Execute all notebooks sequentially to reproduce the complete analysis.
# 
# ### Option 2: Core Analysis Only (1-2 hours)
# Run notebooks 01-05 for the main findings.
# 
# ### Option 3: Specific Analysis
# Jump to any notebook based on your research question:
# - Temporal trends? → Notebook 07
# - Gender differences? → Notebook 14
# - Model calibration? → Notebook 13
# - COVID-19 impact? → Notebook 12
# 
# ### Option 4: Interactive Exploration
# Open any notebook and run cell-by-cell to explore the analysis interactively.

# ## 📊 Key Outputs
# 
# After running the notebooks, you'll generate:
# 
# ### Tables (`outputs/tables/`)
# - Model performance metrics (AUC, F1, precision, recall)
# - Feature importance rankings
# - Regression coefficients
# - Temporal validation results
# - Subgroup analyses
# 
# ### Figures (`figures/`)
# - SHAP summary plots
# - SHAP dependence plots
# - ROC curves
# - Calibration curves
# - Temporal trends
# - Interaction heatmaps
# 
# ### Models (`outputs/models/`)
# - Trained classifiers (.joblib files)
# - Preprocessing pipelines
# - SHAP explainers

# ## 🔍 Workflow Status Checker
# 
# Run this cell to check which analyses have been completed:

# In[ ]:


import os
from pathlib import Path
from datetime import datetime

# Check data availability
data_path = os.path.expanduser('~/work/vaping_project_data/processed_data_g12n.csv')
print("=" * 70)
print(" WORKFLOW STATUS CHECK")
print("=" * 70)

print("\n[1] Data Availability")
if os.path.exists(data_path):
    file_size = os.path.getsize(data_path) / (1024**2)  # MB
    print(f"  ✓ Data file found: {data_path}")
    print(f"  ✓ Size: {file_size:.1f} MB")
else:
    print(f"  ✗ Data file NOT found: {data_path}")
    print("  → Run R preprocessing scripts first!")

# Check output directories
print("\n[2] Output Directories")
output_dirs = {
    'Tables': '../outputs/tables',
    'Models': '../outputs/models',
    'Figures': '../figures'
}

for name, path in output_dirs.items():
    if os.path.exists(path):
        n_files = len([f for f in os.listdir(path) if not f.startswith('.')])
        print(f"  ✓ {name:10s}: {path} ({n_files} files)")
    else:
        print(f"  ○ {name:10s}: {path} (will be created)")

# List available notebooks
print("\n[3] Available Notebooks")
notebooks = sorted([f for f in os.listdir('.') if f.endswith('.ipynb') and not f.startswith('00')])
print(f"  Found {len(notebooks)} analysis notebooks:")
for nb in notebooks[:5]:
    print(f"    • {nb}")
if len(notebooks) > 5:
    print(f"    ... and {len(notebooks)-5} more")

print("\n" + "=" * 70)
print(" Ready to start! Choose a notebook from the list above.")
print("=" * 70)


# ## 📖 Recommended Workflow Paths
# 
# ### For First-Time Users: Core Pipeline
# ```
# 1. Run R preprocessing (scripts/)
# 2. 03_modelling.ipynb          (Main ML analysis)
# 3. 04_regression.ipynb         (Regression analysis)
# 4. 05_charts.ipynb             (Visualizations)
# 5. 07_temporal_validation.ipynb (Temporal check)
# ```
# 
# ### For Reviewers: Validation Focus
# ```
# 1. 03_modelling.ipynb           (Main results)
# 2. 07_temporal_validation.ipynb (Time stability)
# 3. 08_baseline_comparisons.ipynb (Alternative approaches)
# 4. 09_robustness_checks.ipynb   (Sensitivity)
# 5. 13_model_calibration.ipynb   (Calibration)
# ```
# 
# ### For Policy Researchers: Applied Focus
# ```
# 1. 03_modelling.ipynb           (Main predictions)
# 2. 14_gender_interactions.ipynb (Gender differences)
# 3. 15_racial_disparities.ipynb  (Racial/ethnic trends)
# 4. 16_regional_analysis.ipynb   (Geographic patterns)
# 5. 12_structural_break_analysis.ipynb (COVID-19 impact)
# ```
# 
# ### For Methodologists: Technical Deep Dive
# ```
# 1. 03_modelling.ipynb           (ML framework)
# 2. 10_interaction_regression.ipynb (Interactions)
# 3. 11_regression_with_mice.ipynb (Multiple imputation)
# 4. 17_shap_stability.ipynb      (Feature stability)
# 5. 19_incremental_value.ipynb   (Model comparison)
# ```

# ## 💡 Tips for Interactive Analysis
# 
# ### Running Notebooks Interactively
# 1. **Cell-by-cell execution**: Use `Shift+Enter` to run each cell and inspect outputs
# 2. **Modify parameters**: Change hyperparameters, feature sets, or sample sizes
# 3. **Add visualizations**: Insert new cells to create custom plots
# 4. **Save checkpoints**: Use `File → Save Checkpoint` before major changes
# 
# ### Performance Tips
# - **Start small**: Test on subsamples before running full analyses
# - **Use subsets**: Filter data to specific years or grades for faster iteration
# - **Parallel processing**: Most models use `n_jobs=-1` for multi-core processing
# - **Memory management**: Close notebooks after completion to free RAM
# 
# ### Reproducibility
# - All random processes use `RANDOM_STATE = 42`
# - Results should be identical across runs
# - Save all outputs for publication/documentation

# ## 🎓 Learning Resources
# 
# ### Documentation
# - `README.md`: Project overview and setup instructions
# - `docs/CODE_AVAILABILITY.md`: Reproducibility statement
# - `docs/COMPUTATIONAL_NOTES.md`: Hardware requirements and benchmarks
# - `IMPLEMENTATION_SUMMARY.md`: Methodological enhancements
# 
# ### Key Methods
# - **SHAP (SHapley Additive exPlanations)**: Model interpretability
# - **Multi-model consensus**: Feature selection across 6 classifiers
# - **Stratified cross-validation**: Balanced training/testing
# - **Survey weights**: Population-representative estimates
# 
# ### Citation
# If you use this code, please cite:
# ```
# @article{vaping2025,
#   title={Machine Learning Prediction of Adolescent Nicotine Vaping},
#   journal={Journal of Quantitative Criminology},
#   year={2025}
# }
# ```

# ## 🐛 Troubleshooting
# 
# ### Common Issues
# 
# **Data file not found**
# ```
# Solution: Check that processed_data_g12n.csv exists in ~/work/vaping_project_data/
# Run R preprocessing scripts if needed
# ```
# 
# **Missing dependencies**
# ```
# Solution: pip install -r requirements.txt
# or: conda env create -f environment.yml
# ```
# 
# **Out of memory errors**
# ```
# Solution: 
# - Close other applications
# - Reduce sample size for testing
# - Use fewer bootstrap iterations
# - Run analyses sequentially rather than in parallel
# ```
# 
# **Slow SHAP computation**
# ```
# Solution:
# - SHAP TreeExplainer is fast for tree models
# - Use smaller test sets for initial exploration
# - Consider sampling for SHAP dependence plots
# ```
# 
# ### Getting Help
# - Check `docs/COMPUTATIONAL_NOTES.md` for detailed troubleshooting
# - Review individual notebook markdown cells for analysis-specific notes
# - Consult package documentation for method-specific questions

# ## ✅ Next Steps
# 
# 1. **Run the status checker** above to verify setup
# 2. **Choose your workflow path** based on research goals
# 3. **Open first notebook** and begin analysis
# 4. **Execute cells interactively** to understand each step
# 5. **Review outputs** in tables/ and figures/ directories
# 6. **Explore variations** by modifying parameters
# 
# ---
# 
# ## 🎉 Ready to Start!
# 
# You now have a complete interactive workflow for analyzing adolescent vaping behavior using machine learning. 
# 
# **Start with**: `03_modelling.ipynb` for the main analysis
# 
# **Or explore**: Any notebook based on your specific research question
# 
# ---
# 
# *Last updated: November 7, 2025*
