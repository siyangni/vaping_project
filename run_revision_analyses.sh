#!/bin/bash

# Master script to run all revision analyses
# Implements all critical analyses recommended for JQC revision

echo "======================================================================"
echo " Running All Revision Analyses for JQC Submission"
echo "======================================================================"
echo ""
echo "This script will run:"
echo "  1. Preprocessing with survey weights preserved"
echo "  2. Weighted regression analysis"
echo "  3. Temporal validation (train 2017-2021, test 2022-2023)"
echo "  4. Baseline comparisons"
echo "  5. Robustness checks"
echo ""
echo "Estimated time: 2-3 hours"
echo "======================================================================"
echo ""

# Check prerequisites
echo "[STEP 0] Checking prerequisites..."

if ! command -v R &> /dev/null; then
    echo "ERROR: R not found. Please install R first."
    exit 1
fi

if ! command -v python &> /dev/null; then
    echo "ERROR: Python not found. Please install Python first."
    exit 1
fi

# Check data directory
DATA_DIR=~/work/vaping_project_data
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    echo "Please ensure MTF data is available."
    exit 1
fi

echo "✓ Prerequisites OK"
echo ""

# ============================================================================
# STEP 1: Preprocessing with Survey Weights
# ============================================================================

echo "======================================================================"
echo " STEP 1: Preprocessing with Survey Weights Preserved"
echo "======================================================================"
echo ""
echo "Running: scripts/02b_preprocessing_with_weights.R"
echo ""

# Check if original data files exist
if [ ! -f "$DATA_DIR/original_all_core/original_core_2017_0012.tsv" ]; then
    echo "WARNING: Original data files not found in $DATA_DIR/original_all_core/"
    echo "Skipping preprocessing step."
    echo "If processed data exists, this is OK."
else
    Rscript scripts/02b_preprocessing_with_weights.R
    if [ $? -ne 0 ]; then
        echo "ERROR: Preprocessing failed"
        exit 1
    fi
fi

echo "✓ STEP 1 Complete"
echo ""

# ============================================================================
# STEP 2: Weighted Regression Analysis
# ============================================================================

echo "======================================================================"
echo " STEP 2: Weighted Regression Analysis"
echo "======================================================================"
echo ""
echo "Running: notebooks/06_weighted_regression.ipynb"
echo ""

jupyter nbconvert --to notebook --execute notebooks/06_weighted_regression.ipynb \
    --output 06_weighted_regression_executed.ipynb 2>&1 | tee logs/weighted_regression.log

if [ $? -eq 0 ]; then
    echo "✓ STEP 2 Complete"
else
    echo "⚠ STEP 2 had errors (check logs/weighted_regression.log)"
fi
echo ""

# ============================================================================
# STEP 3: Temporal Validation
# ============================================================================

echo "======================================================================"
echo " STEP 3: Temporal Validation (Train 2017-2021, Test 2022-2023)"
echo "======================================================================"
echo ""
echo "Running: notebooks/07_temporal_validation.py"
echo ""

python notebooks/07_temporal_validation.py 2>&1 | tee logs/temporal_validation.log

if [ $? -eq 0 ]; then
    echo "✓ STEP 3 Complete"
else
    echo "⚠ STEP 3 had errors (check logs/temporal_validation.log)"
fi
echo ""

# ============================================================================
# STEP 4: Baseline Comparisons
# ============================================================================

echo "======================================================================"
echo " STEP 4: Baseline Comparisons"
echo "======================================================================"
echo ""
echo "Running: notebooks/08_baseline_comparisons.py"
echo ""

python notebooks/08_baseline_comparisons.py 2>&1 | tee logs/baseline_comparisons.log

if [ $? -eq 0 ]; then
    echo "✓ STEP 4 Complete"
else
    echo "⚠ STEP 4 had errors (check logs/baseline_comparisons.log)"
fi
echo ""

# ============================================================================
# STEP 5: Robustness Checks
# ============================================================================

echo "======================================================================"
echo " STEP 5: Robustness Checks"
echo "======================================================================"
echo ""
echo "Running: notebooks/09_robustness_checks.py"
echo ""

python notebooks/09_robustness_checks.py 2>&1 | tee logs/robustness_checks.log

if [ $? -eq 0 ]; then
    echo "✓ STEP 5 Complete"
else
    echo "⚠ STEP 5 had errors (check logs/robustness_checks.log)"
fi
echo ""

# ============================================================================
# Summary
# ============================================================================

echo "======================================================================"
echo " ALL REVISION ANALYSES COMPLETE"
echo "======================================================================"
echo ""
echo "Generated outputs:"
echo ""
echo "TABLES (outputs/tables/):"
echo "  • weighted_regression_full_model.csv"
echo "  • weighted_regression_with_effect_sizes.csv"
echo "  • model_comparison_nested.csv"
echo "  • likelihood_ratio_tests.csv"
echo "  • temporal_validation_results.csv"
echo "  • temporal_validation_feature_importance.csv"
echo "  • baseline_comparison.csv"
echo "  • robustness_imputation.csv"
echo "  • robustness_threshold.csv"
echo "  • robustness_splits.csv"
echo ""
echo "FIGURES (figures/):"
echo "  • weighted_regression_forest_plot.png"
echo "  • temporal_validation_roc_curves.png"
echo "  • baseline_comparison.png"
echo ""
echo "LOGS (logs/):"
echo "  • weighted_regression.log"
echo "  • temporal_validation.log"
echo "  • baseline_comparisons.log"
echo "  • robustness_checks.log"
echo ""
echo "======================================================================"
echo " Next Steps:"
echo "======================================================================"
echo ""
echo "1. Review all log files for any errors"
echo "2. Examine generated tables and figures"
echo "3. Incorporate results into revised manuscript (docs/main.tex)"
echo "4. Run: pdflatex docs/main.tex to generate PDF"
echo ""
echo "All analyses complete! Ready for manuscript revision."
echo ""
