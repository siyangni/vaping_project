# Data Availability Statement

## Primary Data Source

This research uses data from the **Monitoring the Future (MTF)** study, a nationally representative survey of American adolescents conducted by the University of Michigan's Institute for Social Research.

## Data Access Information

### Public Availability

The MTF data are publicly available through the Inter-university Consortium for Political and Social Research (ICPSR) with the following access requirements:

1. **Registration**: Researchers must create a free account at [ICPSR](https://www.icpsr.umich.edu/)
2. **Data Use Agreement**: Users must agree to ICPSR's Terms of Use
3. **Citation Requirements**: Proper citation of the MTF study is required in all publications

### Data Location

- **Repository**: ICPSR Data Archive
- **URL**: https://www.icpsr.umich.edu/web/ICPSR/series/35
- **Series**: Monitoring the Future Series
- **Study Numbers**:
  - ICPSR 37841 (2017)
  - ICPSR 37842 (2018)
  - ICPSR 37843 (2019)
  - ICPSR 37416 (2020)
  - ICPSR 38502 (2021)
  - ICPSR 38882 (2022)
  - ICPSR 39243 (2023)

### Data Collection Details

- **Years**: 2017-2023 (7 waves)
- **Grade Levels**: 8th, 10th, and 12th grade
- **Sample Size**: ~50,000 students per year
- **Survey Mode**: Paper-and-pencil questionnaires administered in schools
- **Sampling**: Multi-stage random sampling of schools and students

### Data Files Required

To reproduce the analyses in this repository, download the following files from ICPSR:

```
Monitoring the Future: A Continuing Study of American Youth
- 12th-Grade Survey (Core Questionnaire), 2017-2023
- 8th and 10th-Grade Surveys (Core Questionnaire), 2017-2023
```

Expected file naming convention after download:
```
original_core_2017_0012.tsv  # 2017, 12th grade
original_core_2017_0810.tsv  # 2017, 8th-10th grade
original_core_2018_0012.tsv
original_core_2018_0810.tsv
...
original_core_2023_0012.tsv
original_core_2023_0810.tsv
```

## Data Use Restrictions

### Legal and Ethical Constraints

1. **No Redistribution**: Raw MTF data cannot be redistributed by secondary users
2. **Confidentiality**: Individual-level data contain no direct identifiers, but cell sizes <5 should not be reported
3. **Purpose Restriction**: Data should be used for research and statistical purposes only
4. **Citation Required**: All publications must cite both the MTF study and ICPSR

### Why Raw Data Are Not Included

The raw survey data files are **NOT included in this GitHub repository** due to:

1. ICPSR redistribution restrictions
2. File size constraints (~3 GB uncompressed)
3. Privacy protection requirements
4. Standard practice for secondary data analysis

## Processed Data

### Intermediate Datasets

The R preprocessing scripts generate intermediate datasets:

- `merged_data_g12.csv` (~500 MB): All waves merged
- `processed_data_g12.csv` (~300 MB): After feature engineering
- `processed_data_g12n.csv` (~250 MB): Final cleaned dataset (used for modeling)

**Note**: These processed datasets are also not included in the repository due to:
- File size (excluded via `.gitignore`)
- Derivative nature from restricted-use data
- ICPSR terms requiring independent data acquisition

### Data Characteristics

Final cleaned dataset used for modeling:

- **Observations**: ~45,000 students (after missing data handling)
- **Features**: ~100 survey variables (after feature selection)
- **Target Variable**: Binary nicotine vaping indicator (`nicotine12d`)
- **Missing Data**: <10% per feature (imputed using median/mode)
- **Class Balance**: ~15% positive class (nicotine vapers)

## Replication Package

### Complete Replication Steps

1. **Obtain Data** (1-3 business days):
   - Register at ICPSR
   - Request MTF data for 2017-2023
   - Download TSV files
   - Place in `~/work/vaping_project_data/original_all_core/`

2. **Run Preprocessing** (~30 minutes):
   ```bash
   Rscript scripts/01_importing_data.R
   Rscript scripts/02_preprocessing.R
   ```

3. **Run Modeling** (~2-4 hours):
   ```bash
   jupyter notebook notebooks/03_modelling.ipynb
   ```

### Alternative: Use Synthetic Data (for Testing)

For code testing without MTF data access, you can generate synthetic data matching the structure:

```python
# Generate synthetic test data (not for publication!)
python src/generate_synthetic_data.py
```

**Warning**: Synthetic data should NOT be used for scientific inference, only for code testing and development.

## Data Citation

When using MTF data, cite as:

```
Miech, R. A., Johnston, L. D., Bachman, J. G., O'Malley, P. M., Schulenberg, J. E., & Patrick, M. E. (2023).
Monitoring the Future: A Continuing Study of American Youth (12th-Grade Survey), 2023.
Inter-university Consortium for Political and Social Research [distributor].
https://doi.org/10.3886/ICPSR39243.v1
```

## Contact for Data Access Issues

- **MTF Website**: https://monitoringthefuture.org/
- **ICPSR Support**: https://www.icpsr.umich.edu/web/pages/support/
- **MTF Data Support**: mtfdata@umich.edu

## Compliance with Journal Policies

This data availability statement complies with:

- **Journal of Quantitative Criminology** data sharing policy (Type 2)
- **Springer Nature** research data policy
- **ICPSR** data use agreement terms
- **Federal** regulations on human subjects research (data are de-identified)

---

**Last Updated**: 2025-11-05
**Document Version**: 1.0
**Repository**: https://github.com/siyangni/vaping_project
