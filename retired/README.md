# Retired Scripts

This folder contains Python scripts that have been converted to interactive Jupyter notebooks.

## Why These Files Were Retired

These scripts (07-19) were originally designed for automated batch execution via shell scripts (`run_all.sh` and `run_revision_analyses.sh`). They have been converted to Jupyter notebooks to support:

- **Interactive exploration**: Run analyses cell-by-cell
- **Iterative development**: Modify parameters and re-run sections
- **Better documentation**: Inline markdown explanations
- **Easier teaching**: Step-by-step walkthroughs
- **Improved reproducibility**: Visual verification of outputs

## What's Here

| Script | Converted To | Purpose |
|--------|--------------|---------|
| `07_temporal_validation.py` | `notebooks/07_temporal_validation.ipynb` | Temporal validation (train 2017-2021, test 2022-2023) |
| `08_baseline_comparisons.py` | `notebooks/08_baseline_comparisons.ipynb` | Compare multi-model to baseline approaches |
| `09_robustness_checks.py` | `notebooks/09_robustness_checks.ipynb` | Sensitivity analyses |
| `10_interaction_regression.py` | `notebooks/10_interaction_regression.ipynb` | Test SHAP-identified interactions |
| `11_regression_with_mice.py` | `notebooks/11_regression_with_mice.ipynb` | Multiple imputation analysis |
| `12_structural_break_analysis.py` | `notebooks/12_structural_break_analysis.ipynb` | COVID-19 structural break testing |
| `13_model_calibration.py` | `notebooks/13_model_calibration.ipynb` | Calibration assessment |
| `14_gender_interactions.py` | `notebooks/14_gender_interactions.ipynb` | Gender-stratified analysis |
| `15_racial_disparities.py` | `notebooks/15_racial_disparities.ipynb` | Racial/ethnic trend analysis |
| `16_regional_analysis.py` | `notebooks/16_regional_analysis.ipynb` | Regional variation analysis |
| `17_shap_stability.py` | `notebooks/17_shap_stability.ipynb` | Bootstrap SHAP stability |
| `18_shap_dependence.py` | `notebooks/18_shap_dependence.ipynb` | SHAP dependence plots |
| `19_incremental_value.py` | `notebooks/19_incremental_value.ipynb` | Nested model comparison |

## When to Use These

**Use the Jupyter notebooks** (recommended) for:
- Interactive analysis
- Learning and exploration
- Teaching and presentations
- Step-by-step verification

**Keep these scripts** for reference:
- Comparing implementation approaches
- Understanding original workflow
- Historical documentation

## Note

These scripts are functionally identical to the notebooks but designed for command-line execution. The notebook versions provide the same analyses with enhanced interactivity and documentation.

---

*Last updated: November 7, 2025*
