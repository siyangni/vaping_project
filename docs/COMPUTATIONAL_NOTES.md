# Computational and Technical Notes

## Hardware Requirements

### Minimum Specifications

- **CPU**: Quad-core processor (2.0 GHz or higher)
- **RAM**: 16 GB
- **Storage**: 10 GB free disk space
- **OS**: Linux, macOS, or Windows 10+ with WSL2

### Recommended Specifications

- **CPU**: 16+ cores with multi-threading support
- **RAM**: 32 GB or more
- **Storage**: 20 GB free disk space (SSD preferred)
- **GPU**: Optional (not utilized by current code)

## Software Dependencies

### Core Languages

- **Python**: 3.8, 3.9, 3.10, or 3.11 (3.10 recommended)
- **R**: 4.0+ (4.2 or 4.3 recommended)

### Critical Python Packages

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 1.24.3 | Numerical computing |
| pandas | 2.0.3 | Data manipulation |
| scikit-learn | 1.3.2 | ML algorithms |
| xgboost | 2.0.3 | Gradient boosting |
| catboost | 1.2.2 | Gradient boosting |
| shap | 0.44.1 | Model interpretation |
| matplotlib | 3.7.3 | Visualization |
| seaborn | 0.13.0 | Statistical visualization |

### Critical R Packages

| Package | Purpose |
|---------|---------|
| tidyverse | Data wrangling |
| caret | ML framework |
| glmnet | Regularized regression |
| randomForest | Random forest |
| doParallel | Parallel processing |

## Computational Complexity

### Time Complexity

**Data Preprocessing (R)**:
- Loading: O(n) where n = number of observations
- Feature engineering: O(n × p) where p = number of features
- Total: ~30 minutes for 50K observations

**Model Training (Python)**:
- Logistic Regression: O(n × p × i) where i = iterations (~5 min)
- Random Forest: O(n × log(n) × p × t) where t = trees (~30 min)
- Gradient Boosting: O(n × p × d × t) where d = depth (~45 min)
- XGBoost: Optimized O(n × p × d × t) (~45 min)
- CatBoost: Optimized O(n × p × d × t) (~30 min)

**SHAP Analysis**:
- Tree SHAP: O(TLD²) where T = trees, L = leaves, D = depth (~30 min)

### Space Complexity

**Memory Usage**:
- Raw data: ~3 GB (14 TSV files)
- Processed data: ~250 MB
- Trained models: ~500 MB total
- SHAP values: ~1 GB
- Peak RAM: ~8-12 GB during hyperparameter tuning

**Disk Storage**:
- Input data: ~3 GB
- Intermediate files: ~1 GB
- Outputs: ~2 GB
- Total: ~6 GB

## Parallelization

### R Scripts

R preprocessing uses `doParallel` for:
- Cross-validation (CV_FOLDS parallel processes)
- Feature importance computation

```r
library(doParallel)
registerDoParallel(cores = detectCores() - 1)
```

### Python Scripts

Python uses:
- **scikit-learn**: `n_jobs=-1` (all available cores)
- **XGBoost**: Multi-threaded by default
- **CatBoost**: Multi-threaded by default

## Numerical Precision

### Floating-Point Arithmetic

- Python: 64-bit floating point (double precision)
- R: 64-bit floating point
- Expected numerical precision: ~1e-15

### Reproducibility Considerations

1. **Random Seeds**: Set consistently across all code
2. **Parallel Processing**: May introduce minor numerical differences
3. **Cross-Platform**: Results reproducible within ±0.001

## Known Platform-Specific Issues

### macOS

- **XGBoost**: May require OpenMP: `brew install libomp`
- **CatBoost**: Works natively on M1/M2 chips

### Windows

- **Recommend WSL2**: For full compatibility
- **Native Windows**: May require Visual C++ Build Tools
- **Parallel Processing**: May be slower than Linux/macOS

### Linux

- **Recommended**: Best performance and compatibility
- **GPU Support**: CUDA-enabled GPUs not currently utilized

## Optimization Tips

### For Faster Execution

1. **Reduce Hyperparameter Grid**:
   - Decrease `n_iter` in RandomizedSearchCV
   - Use fewer CV folds (3 instead of 5)

2. **Subsample Data** (for testing):
   ```python
   X_train_sample = X_train[:10000]  # Use 10K samples
   ```

3. **Use Fewer Models**:
   - Comment out slower models (Gradient Boosting)
   - Focus on Random Forest + XGBoost

### For Lower Memory Usage

1. **Process Data in Chunks**:
   ```python
   for chunk in pd.read_csv('data.csv', chunksize=10000):
       process(chunk)
   ```

2. **Use Sparse Matrices**:
   ```python
   from scipy.sparse import csr_matrix
   X_sparse = csr_matrix(X)
   ```

3. **Delete Intermediate Objects**:
   ```python
   import gc
   del large_object
   gc.collect()
   ```

## Benchmarking Results

### Reference System

- **CPU**: Intel Xeon E5-2680 v4 (16 cores, 3.2 GHz)
- **RAM**: 32 GB DDR4
- **Storage**: NVMe SSD
- **OS**: Ubuntu 20.04 LTS

### Execution Times

| Stage | Time | Peak RAM |
|-------|------|----------|
| Data loading (R) | 5 min | 4 GB |
| Preprocessing (R) | 25 min | 8 GB |
| EDA (R) | 20 min | 6 GB |
| Logistic Reg (Python) | 5 min | 2 GB |
| Random Forest (Python) | 30 min | 6 GB |
| Gradient Boost (Python) | 45 min | 4 GB |
| XGBoost (Python) | 45 min | 5 GB |
| CatBoost (Python) | 30 min | 4 GB |
| SHAP Analysis (Python) | 30 min | 10 GB |
| Visualization (Python) | 10 min | 2 GB |
| **TOTAL** | **~4 hours** | **12 GB peak** |

## Troubleshooting

### Out of Memory Errors

**Symptom**: `MemoryError` or system freeze

**Solutions**:
1. Close other applications
2. Reduce batch size or use data subsampling
3. Use sparse matrices
4. Increase swap space (Linux)

### Slow Execution

**Symptom**: Code runs much slower than benchmarks

**Solutions**:
1. Check CPU usage (should be near 100% during training)
2. Verify multi-core parallelization is working
3. Check disk I/O (use SSD if possible)
4. Disable antivirus during execution

### Package Conflicts

**Symptom**: Import errors or version conflicts

**Solutions**:
1. Use isolated environment (conda or venv)
2. Install exact versions from `requirements.txt`
3. Clear package cache: `pip cache purge`

### Numerical Instability

**Symptom**: NaN values or divergence warnings

**Solutions**:
1. Check for missing data
2. Verify feature scaling
3. Adjust regularization parameters
4. Check for zero-variance features

## Validation Checklist

After running the pipeline, verify:

- [ ] All models trained successfully (6 models)
- [ ] ROC-AUC scores between 0.70-0.95 (reasonable range)
- [ ] No NaN values in predictions
- [ ] SHAP values sum to prediction differences
- [ ] Figures generated without errors
- [ ] Model comparison table saved
- [ ] Output files in correct directories

## Performance Metrics

Expected model performance ranges (test set):

| Metric | Minimum | Expected | Maximum |
|--------|---------|----------|---------|
| ROC-AUC | 0.70 | 0.80-0.85 | 0.95 |
| Accuracy | 0.75 | 0.82-0.88 | 0.95 |
| F1-Score | 0.60 | 0.70-0.80 | 0.90 |
| Precision | 0.65 | 0.75-0.85 | 0.95 |
| Recall | 0.60 | 0.70-0.80 | 0.90 |

If results fall outside these ranges:
1. Check data loading (correct files?)
2. Verify preprocessing (no errors?)
3. Check for data leakage
4. Review hyperparameter tuning results

## Code Profiling

To identify bottlenecks:

### Python

```python
import cProfile
cProfile.run('your_function()', 'output.prof')

# Analyze with snakeviz
pip install snakeviz
snakeviz output.prof
```

### R

```r
Rprof("output.prof")
your_function()
Rprof(NULL)
summaryRprof("output.prof")
```

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-11-05 | Initial release |
| 2.0.0 | 2025-11-07 | Added comprehensive revisions and enhancements |

---

**Last Updated**: 2025-11-07
**Document Version**: 2.0
**Contact**: [GitHub Issues](https://github.com/siyangni/vaping_project/issues)
