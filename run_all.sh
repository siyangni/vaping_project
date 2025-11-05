#!/bin/bash

################################################################################
# Master Script: Complete Reproducible Analysis Pipeline
#
# This script executes the entire computational pipeline for the vaping
# project from raw data to final publication figures.
#
# Usage: bash run_all.sh
#
# Prerequisites:
# 1. MTF data files in ~/work/vaping_project_data/original_all_core/
# 2. R 4.0+ with required packages installed
# 3. Python 3.8+ with conda environment 'vaping_env' or venv activated
# 4. 16+ GB RAM, ~5 hours runtime
#
# Author: Siyang Ni
# Date: 2025-11-05
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Timer functions
start_time=$(date +%s)

print_elapsed() {
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    local hours=$((elapsed / 3600))
    local minutes=$(((elapsed % 3600) / 60))
    local seconds=$((elapsed % 60))
    log_info "Elapsed time: ${hours}h ${minutes}m ${seconds}s"
}

# Print header
echo "================================================================================"
echo "  VAPING PROJECT: REPRODUCIBLE ANALYSIS PIPELINE"
echo "================================================================================"
echo ""
log_info "Start time: $(date)"
echo ""

# Check prerequisites
log_info "Checking prerequisites..."

# Check if R is installed
if ! command -v Rscript &> /dev/null; then
    log_error "R is not installed. Please install R 4.0+."
    exit 1
fi
log_success "R found: $(Rscript --version 2>&1 | head -n 1)"

# Check if Python is installed
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    log_error "Python is not installed. Please install Python 3.8+."
    exit 1
fi
PYTHON_CMD=$(command -v python3 || command -v python)
log_success "Python found: $($PYTHON_CMD --version)"

# Check if data directory exists
DATA_DIR=~/work/vaping_project_data/original_all_core
if [ ! -d "$DATA_DIR" ]; then
    log_error "Data directory not found: $DATA_DIR"
    log_error "Please download MTF data and place TSV files in this directory."
    exit 1
fi

# Count data files
NUM_FILES=$(find "$DATA_DIR" -name "*.tsv" | wc -l)
if [ "$NUM_FILES" -lt 14 ]; then
    log_warning "Expected 14 TSV files, found $NUM_FILES"
    log_warning "Analysis may fail if data files are missing."
else
    log_success "Found $NUM_FILES data files"
fi

# Create output directories if they don't exist
log_info "Creating output directories..."
mkdir -p outputs/models outputs/predictions outputs/tables figures
log_success "Output directories ready"

echo ""
echo "================================================================================"
echo "  STAGE 1: R PREPROCESSING"
echo "================================================================================"
echo ""

# Stage 1.1: Data Import
log_info "Step 1/3: Importing raw data..."
if Rscript scripts/01_importing_data.R; then
    log_success "Data import completed"
else
    log_error "Data import failed"
    exit 1
fi
print_elapsed

# Stage 1.2: Preprocessing
log_info "Step 2/3: Preprocessing and feature engineering..."
if Rscript scripts/02_preprocessing.R; then
    log_success "Preprocessing completed"
else
    log_error "Preprocessing failed"
    exit 1
fi
print_elapsed

# Stage 1.3: Exploratory Data Analysis (optional)
log_info "Step 3/3: Exploratory data analysis (optional)..."
if Rscript scripts/03_EDA.R; then
    log_success "EDA completed"
else
    log_warning "EDA failed (continuing anyway...)"
fi
print_elapsed

echo ""
echo "================================================================================"
echo "  STAGE 2: PYTHON MACHINE LEARNING"
echo "================================================================================"
echo ""

# Check if Jupyter is installed
if ! $PYTHON_CMD -c "import jupyter" &> /dev/null; then
    log_error "Jupyter not found. Please install: pip install jupyter"
    exit 1
fi

# Stage 2.1: Main Modeling Notebook
log_info "Step 1/3: Training ML models (this will take 2-4 hours)..."
log_info "Running notebook: notebooks/03_modelling.ipynb"

# Execute notebook with jupyter nbconvert
if $PYTHON_CMD -m jupyter nbconvert --to notebook --execute \
    --output ../outputs/03_modelling_executed.ipynb \
    notebooks/03_modelling.ipynb; then
    log_success "Modeling completed"
else
    log_error "Modeling failed"
    exit 1
fi
print_elapsed

# Stage 2.2: Regression Analysis
log_info "Step 2/3: Running regression analysis..."
if $PYTHON_CMD -m jupyter nbconvert --to notebook --execute \
    --output ../outputs/04_regression_executed.ipynb \
    notebooks/04_regression.ipynb; then
    log_success "Regression analysis completed"
else
    log_warning "Regression analysis failed (continuing anyway...)"
fi
print_elapsed

# Stage 2.3: Visualization
log_info "Step 3/3: Generating publication figures..."
if $PYTHON_CMD -m jupyter nbconvert --to notebook --execute \
    --output ../outputs/05_charts_executed.ipynb \
    notebooks/05_charts.ipynb; then
    log_success "Figure generation completed"
else
    log_warning "Figure generation failed (continuing anyway...)"
fi
print_elapsed

echo ""
echo "================================================================================"
echo "  PIPELINE COMPLETED SUCCESSFULLY"
echo "================================================================================"
echo ""

# Summary
log_success "All stages completed!"
echo ""
log_info "Output locations:"
echo "  - Trained models:       outputs/models/"
echo "  - Predictions:          outputs/predictions/"
echo "  - Performance tables:   outputs/tables/"
echo "  - Publication figures:  figures/"
echo "  - Executed notebooks:   outputs/"
echo ""

# File counts
NUM_MODELS=$(find outputs/models -name "*.joblib" 2>/dev/null | wc -l)
NUM_FIGURES=$(find figures -name "*.png" 2>/dev/null | wc -l)

log_info "Generated files:"
echo "  - Models:    $NUM_MODELS"
echo "  - Figures:   $NUM_FIGURES"
echo ""

# Print total elapsed time
print_elapsed
log_info "End time: $(date)"

echo ""
echo "================================================================================"
echo "Next steps:"
echo "  1. Review outputs in outputs/ and figures/ directories"
echo "  2. Check executed notebooks in outputs/*.ipynb"
echo "  3. Compare results with manuscript"
echo "  4. Report any issues at: https://github.com/siyangni/vaping_project/issues"
echo "================================================================================"
echo ""

exit 0
