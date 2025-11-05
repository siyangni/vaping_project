# IMPLEMENTATION PLAN: Enhancements to Main.tex and Analytical Process

**Date**: 2025-11-05
**Purpose**: Structured plan for implementing suggestions from comprehensive repository analysis

---

## OVERVIEW

This document outlines a phased implementation plan for 13 major enhancements to strengthen both the methodological framework and substantive criminological contributions of the vaping project.

**Estimated Total Time**: 40-60 hours over 2-3 weeks
**Priority Levels**: ⭐⭐⭐ = Critical | ⭐⭐ = Important | ⭐ = Nice-to-have

---

## PHASE 1: CRITICAL METHODOLOGICAL FIXES (Week 1)

### Priority 1A: Interaction Terms in Regression ⭐⭐⭐
**Time**: 8-12 hours
**Files to Create/Modify**:
- `notebooks/10_interaction_regression.py` (NEW)
- `docs/main.tex` (Section 2.3.4 - add new subsection, Section 3.3 - update results)

**Steps**:
1. Create interaction variables: wave×marijuana, wave×alcohol, wave×cigarettes
2. Create post-2020 dummy and post2020×marijuana interaction
3. Fit 3 regression models: (A) main effects only, (B) + 2-way interactions, (C) + threshold
4. Conduct likelihood ratio tests comparing models A vs B vs C
5. Create interaction plots showing how marijuana effect changes across waves
6. Update Table 4 in main.tex to show Model B results
7. Add interpretation to Results section

**Expected Findings**:
- Wave × Marijuana interaction: p < 0.001
- Marijuana protective effect stronger pre-2020, weakens/reverses post-2020
- Demonstrates value of ML discovery → regression testing pipeline

**Success Criteria**:
- [ ] Interaction model fits successfully
- [ ] LR test shows significant improvement (p < 0.001)
- [ ] Interaction plot clearly shows changing marijuana effect
- [ ] main.tex Section 2.3.4 added with equations
- [ ] main.tex Section 3.3 updated with interaction results

---

### Priority 1B: MICE Multiple Imputation ⭐⭐⭐
**Time**: 6-8 hours
**Files to Create/Modify**:
- `scripts/03_mice_imputation.R` (NEW)
- `notebooks/11_regression_with_mice.py` (NEW)
- `docs/main.tex` (Section 2.3.2 - verify implementation matches description)

**Steps**:
1. Implement MICE in R with m=5, maxit=20, PMM method
2. Check convergence plots
3. Export 5 imputed datasets to CSV
4. Modify Python regression code to:
   - Load all 5 imputed datasets
   - Fit regression on each
   - Pool estimates using Rubin's rules (beta_bar, within-variance, between-variance)
   - Compute pooled SEs and p-values
5. Compare pooled results vs single-imputation results
6. Update all regression tables in main.tex with MICE-based estimates

**Expected Changes**:
- Confidence intervals 5-10% wider (more conservative)
- Some marginal effects (p=0.04) may become non-significant (p=0.06)
- More honest uncertainty quantification

**Success Criteria**:
- [ ] MICE converges (trace plots stable)
- [ ] Pooled estimates computed correctly
- [ ] Variance formula matches equation in main.tex (line 289-291)
- [ ] All regression tables updated

---

### Priority 1C: Structural Break Analysis ⭐⭐⭐
**Time**: 4-6 hours
**Files to Create/Modify**:
- `notebooks/12_structural_break_analysis.py` (NEW)
- `docs/main.tex` (Section 3.2.4 - add formal test subsection after line 476)

**Steps**:
1. Fit 3 models: pooled, pre-2021 only, post-2020 only
2. Compute Chow test (LR version for GLM)
3. Extract and compare coefficients pre vs post
4. Implement interrupted time series model with:
   - wave_centered (wave - 20)
   - post_2020 (dummy for wave > 20)
   - wave_post (interaction: slope change post-2020)
5. Create coefficient comparison table
6. Test whether specific variables (marijuana, alcohol) show different effects pre vs post
7. Add formal statistical test to main.tex with F-statistic and p-value

**Expected Findings**:
- Massive structural break: χ² > 1000, p < 0.001
- Marijuana effect changes from OR=0.85 (pre) to OR=1.05 (post)
- Baseline vaping probability jumps ~35 percentage points in 2021

**Success Criteria**:
- [ ] Chow test computed correctly
- [ ] ITS model fits and shows significant break
- [ ] Coefficient table shows clear pre/post differences
- [ ] main.tex updated with formal test section

---

## PHASE 2: SUBSTANTIVE CRIMINOLOGICAL DEEPENING (Week 2)

### Priority 2A: Gender × Substance Interactions ⭐⭐
**Time**: 4-5 hours
**Files to Create/Modify**:
- `notebooks/14_gender_interactions.py` (NEW)
- `docs/main.tex` (Section 3.2.4 - add new paragraph)

**Steps**:
1. Create female×marijuana, female×alcohol interaction terms
2. Fit interaction model and test significance
3. Fit stratified models (males only, females only)
4. Compare marijuana ORs: expect OR_male ≈ 0.98, OR_female ≈ 0.83
5. Create interaction plot showing differential effects
6. Add theoretical interpretation citing Akers (2009) on peer networks
7. Add policy implications paragraph

**Success Criteria**:
- [ ] Gender×Marijuana interaction p < 0.001
- [ ] Stratified results show clear gender difference
- [ ] Interaction plot created
- [ ] main.tex paragraph added with criminological theory integration

---

### Priority 2B: Racial/Ethnic Disparity Analysis ⭐⭐
**Time**: 5-7 hours
**Files to Create/Modify**:
- `notebooks/15_racial_disparities.py` (NEW)
- `docs/main.tex` (Section 3.2.4 - add new paragraph on racial disparities)

**Steps**:
1. Create race/ethnicity dummies (white=ref, black, hispanic, asian, other)
2. Fit model with race dummies + SES controls (parent education, income)
3. Test race × wave interactions (are disparities widening/narrowing?)
4. Create prevalence trends by race/ethnicity plot
5. Calculate disparity ratios: Black/White, Hispanic/White prevalence
6. Add interpretation with 4 theoretical mechanisms:
   - Differential access
   - Cultural norms
   - Enforcement salience
   - Protective cultural factors
7. Add critical discussion of historical parallels (crack, menthol)

**Success Criteria**:
- [ ] Racial disparities quantified with ORs and CIs
- [ ] Temporal trend analysis shows if gaps widening/narrowing
- [ ] Main.tex paragraph connects to Sampson & Laub, Hirschi theories
- [ ] Figure showing racial/ethnic trends created

---

### Priority 2C: Regional Variation Analysis ⭐⭐
**Time**: 4-5 hours
**Files to Create/Modify**:
- `notebooks/16_regional_analysis.py` (NEW)
- `docs/main.tex` (Section 3.2.4 - expand region discussion)

**Steps**:
1. Create region dummies (northeast, midwest, south, west=ref)
2. Fit model with region effects
3. Test region × wave interactions
4. Calculate regional prevalence by year
5. Create plot showing diverging regional trends
6. Link to policy contexts:
   - Western flavor bans
   - Regional marijuana legalization patterns
   - Socioeconomic composition
7. Suggest future state-level policy analysis

**Success Criteria**:
- [ ] Regional effects quantified
- [ ] Trends plot shows West highest, diverging over time
- [ ] Policy context interpretation added
- [ ] Future research direction specified

---

## PHASE 3: ADVANCED ANALYTICAL EXTENSIONS (Week 3)

### Priority 3A: Model Calibration Analysis ⭐⭐
**Time**: 3-4 hours
**Files to Create/Modify**:
- `notebooks/13_model_calibration.py` (NEW)
- `docs/main.tex` (Section 3.1.1 - add calibration subsection)

**Steps**:
1. Compute Brier scores for all 6 models
2. Generate calibration plots (predicted vs observed in deciles)
3. Identify that tree models are overconfident
4. Recommend isotonic calibration for risk scoring applications
5. Add to manuscript distinguishing: discrimination (AUC) vs calibration (Brier)

**Success Criteria**:
- [ ] Calibration plots generated for all models
- [ ] Brier scores computed
- [ ] Manuscript section explains when calibration matters
- [ ] Recommendation provided for applied use

---

### Priority 3B: Feature Importance Stability ⭐⭐
**Time**: 4-5 hours
**Files to Create/Modify**:
- `notebooks/17_shap_stability.py` (NEW)
- `docs/main.tex` (Section 3.1.2 - add stability analysis)

**Steps**:
1. Implement bootstrap SHAP (50 bootstrap samples)
2. Compute mean SHAP importance + 95% CIs
3. Calculate coefficient of variation (CV) for each feature
4. Identify unstable features (CV > 50%)
5. Create error bar plot showing SHAP mean ± CI
6. Discuss: stable features are most trustworthy for consensus

**Success Criteria**:
- [ ] Bootstrap SHAP implemented correctly
- [ ] Stability metrics computed
- [ ] Figure with error bars created
- [ ] Manuscript discusses implication for consensus approach

---

### Priority 3C: SHAP Dependence Plots ⭐⭐
**Time**: 3-4 hours
**Files to Create/Modify**:
- `notebooks/18_shap_dependence.py` (NEW)
- Add 2-3 figures to main.tex

**Steps**:
1. Create SHAP dependence plot: wave (colored by marijuana)
2. Create SHAP dependence plot: marijuana (colored by wave)
3. Create SHAP interaction heatmap for top 15 features
4. Add interpretation showing how marijuana effect changes across waves visually
5. Link to regression interaction results (Priority 1A)

**Success Criteria**:
- [ ] 3 high-quality dependence plots created
- [ ] Plots clearly show wave×marijuana interaction
- [ ] Figures integrated into main.tex Section 3.2

---

### Priority 3D: Incremental Predictive Value ⭐⭐
**Time**: 4-5 hours
**Files to Create/Modify**:
- `notebooks/19_incremental_value.py` (NEW)
- `docs/main.tex` (Section 3.3.2 - add subsection on tier comparison)

**Steps**:
1. Fit all 6 nested models on same train/test split
2. Compute AUC for each tier
3. Conduct DeLong tests comparing successive tiers
4. Determine optimal tier (best parsimony-performance trade-off)
5. Create plot: AUC vs number of features (diminishing returns curve)
6. Add guidance: "For most applications, Tier 1-2 features suffice"

**Success Criteria**:
- [ ] DeLong tests show Tier 2 significant, Tiers 3-6 not significant
- [ ] Optimal tier identified
- [ ] Diminishing returns plot created
- [ ] Practical guidance added to manuscript

---

## PHASE 4: MANUSCRIPT POLISHING (Final 2-3 days)

### Priority 4A: Revise Abstract ⭐⭐⭐
**Time**: 2-3 hours
**Files to Modify**:
- `docs/main.tex` (lines 26-34)

**Steps**:
1. Replace abstract with improved version (see detailed suggestion above)
2. Ensure COVID-19 finding prominently featured
3. Quantify performance gains over baselines
4. State counterintuitive marijuana/alcohol findings
5. Balance methods (55%) vs findings (45%)
6. Verify word count < 350 words

**Success Criteria**:
- [ ] Abstract clearly states all 6 ML models
- [ ] COVID-19 threshold effect quantified
- [ ] Baseline comparisons mentioned
- [ ] More balanced criminology-methods framing

---

### Priority 4B: Enhance Limitations Section ⭐⭐
**Time**: 2-3 hours
**Files to Modify**:
- `docs/main.tex` (Section 4.3, lines 918-927)

**Steps**:
1. Replace generic limitations with specific ones tied to your data
2. Add 3 concrete remedies:
   - Fixed-effects models for causality
   - State-level DiD for COVID mechanism
   - Heckman selection for dropout bias
3. Make each limitation actionable with equations/designs

**Success Criteria**:
- [ ] Limitations are specific, not generic
- [ ] Each limitation has concrete remedy
- [ ] Equations provided for future designs
- [ ] Demonstrates methodological sophistication

---

### Priority 4C: Create Framework Decision Tree ⭐
**Time**: 3-4 hours
**Files to Create/Modify**:
- `figures/framework_decision_tree.pdf` (NEW - create in R/Python or Adobe)
- `docs/main.tex` (Section 2.4 - add new section)

**Steps**:
1. Create flowchart with 5 decision points:
   - Outcome type
   - p vs n ratio
   - Primary goal
   - Computational resources
   - Consensus threshold
2. Export as high-res PDF
3. Add LaTeX section guiding researchers through decisions
4. Make framework more accessible to non-experts

**Success Criteria**:
- [ ] Decision tree is clear and professional
- [ ] Covers key decision points
- [ ] Integrated into manuscript
- [ ] Makes framework more usable

---

## VERIFICATION CHECKLIST

After completing all phases, verify:

### Code Verification
- [ ] All new Python scripts run without errors
- [ ] All new R scripts source successfully
- [ ] Results are reproducible (set random seeds)
- [ ] Outputs saved to correct directories (`outputs/`, `figures/`)

### Manuscript Verification
- [ ] All new sections compile in LaTeX
- [ ] All cross-references work
- [ ] All figures are referenced in text
- [ ] All tables have clear captions
- [ ] Equations are numbered correctly
- [ ] Citations are formatted properly

### Substantive Verification
- [ ] Interaction findings align with theoretical predictions
- [ ] Structural break analysis confirms COVID-19 threshold
- [ ] Gender/race analyses connect to criminological theories
- [ ] Limitations are honest and specific
- [ ] Policy implications are concrete and actionable

### Methodological Verification
- [ ] MICE implementation matches manuscript description
- [ ] Rubin's rules applied correctly
- [ ] Calibration metrics appropriate for use case
- [ ] SHAP stability analysis validates consensus approach
- [ ] All hypothesis tests use appropriate methods

---

## TIMELINE

**Week 1** (Days 1-5):
- Day 1-2: Priority 1A (Interactions)
- Day 3: Priority 1B (MICE)
- Day 4: Priority 1C (Structural break)
- Day 5: Review and debugging

**Week 2** (Days 6-10):
- Day 6: Priority 2A (Gender)
- Day 7-8: Priority 2B (Race)
- Day 9: Priority 2C (Region)
- Day 10: Integration and review

**Week 3** (Days 11-15):
- Day 11: Priority 3A (Calibration)
- Day 12: Priority 3B (Stability)
- Day 13: Priority 3C (SHAP dependence)
- Day 14: Priority 3D (Incremental value)
- Day 15: Final manuscript polishing

**Final 2-3 Days**:
- Abstract revision
- Limitations enhancement
- Decision tree creation
- Final LaTeX compilation
- Proofreading

---

## RISK MITIGATION

### Potential Challenges

**Challenge 1**: MICE may not converge
**Mitigation**: Try different methods (PMM, RF imputation), reduce maxit, or exclude highly sparse variables

**Challenge 2**: Interaction models may be too complex (convergence issues)
**Mitigation**: Standardize all variables before creating interactions, use regularization

**Challenge 3**: Racial categories may have small cell sizes
**Mitigation**: Combine small categories (Other/Mixed), use exact logistic regression if needed

**Challenge 4**: Regional variation may be confounded with urbanicity
**Mitigation**: Add urbanicity as control, test region effects within urban/rural strata

**Challenge 5**: Bootstrap SHAP may be computationally expensive
**Mitigation**: Reduce to 30 bootstrap samples, subsample data to n=10,000

---

## DEPENDENCIES

**Software Requirements**:
- R >= 4.0 with `mice`, `tidyverse`, `glmnet`, `survey` packages
- Python >= 3.8 with `statsmodels`, `scikit-learn`, `shap`, `xgboost`, `catboost`
- LaTeX with `amsmath`, `booktabs`, `graphicx`, `natbib`

**Data Requirements**:
- `processed_data_g12n_weighted.csv` with survey weights preserved
- All 5 MICE-imputed datasets
- Trained ML models in `outputs/models/`

**Compute Resources**:
- RAM: 16GB minimum (32GB recommended for SHAP bootstrap)
- CPU: 4+ cores for parallel processing
- Storage: 5GB for imputed datasets and figures
- Runtime: ~20-30 hours total compute time

---

## SUCCESS METRICS

### Quantitative Metrics
1. **Coverage**: All 13 priorities addressed
2. **Reproducibility**: All scripts run with seed=42 and produce same results
3. **Significance**: Interaction model LR test p < 0.001
4. **Stability**: Top 10 SHAP features have CV < 30%
5. **Calibration**: Brier score for regression model < 0.18

### Qualitative Metrics
1. **Clarity**: Non-expert can follow decision tree to apply framework
2. **Rigor**: Limitations section demonstrates methodological sophistication
3. **Integration**: ML findings seamlessly inform regression specifications
4. **Contribution**: Manuscript balances methodological innovation with substantive criminological insights
5. **Policy Relevance**: Recommendations are concrete and actionable

---

## NEXT STEPS AFTER COMPLETION

1. **Compile full manuscript**: `pdflatex main.tex`
2. **Generate supplementary materials**: Full regression tables, additional figures
3. **Prepare replication package**: Anonymized data + complete code archive
4. **Submit to Journal of Quantitative Criminology**
5. **Prepare response to reviewers** (using implementation plan to address methodological concerns)

---

**Document Version**: 1.0
**Last Updated**: 2025-11-05
**Estimated Total Effort**: 40-60 hours
**Expected Completion**: 3 weeks from start
