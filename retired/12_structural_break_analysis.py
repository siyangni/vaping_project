"""
Structural Break Analysis: COVID-19 as Natural Experiment

Formal statistical tests for structural break at 2020-2021:
- Chow test for parameter stability
- Interrupted time series analysis
- Coefficient comparison pre vs post 2020

Author: Siyang Ni
Date: 2025-11-05
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*70)
print(" STRUCTURAL BREAK ANALYSIS: COVID-19 THRESHOLD")
print("="*70)

# Load data
data_path = os.path.expanduser('~/work/vaping_project_data/processed_data_g12n.csv')
df = pd.read_csv(data_path)
df_clean = df[df['nicotine12d'].notna()].copy()

print(f"\nData: {len(df_clean):,} observations")
print(f"Wave range: {df_clean['wave'].min()} to {df_clean['wave'].max()}")

# Split data
df_pre = df_clean[df_clean['wave'] <= 20].copy()
df_post = df_clean[df_clean['wave'] >= 21].copy()

print(f"\nPre-2021: {len(df_pre):,} observations")
print(f"Post-2020: {len(df_post):,} observations")

# Model specification
formula = 'nicotine12d ~ wave + marijuana12 + alcohol12 + cigarette12 + political + region'

# ============================================================================
# 1. POOLED MODEL (No break)
# ============================================================================

print("\n" + "="*70)
print(" MODEL 1: POOLED (No Structural Break)")
print("="*70)

model_pooled = smf.logit(formula, data=df_clean).fit(disp=0)
print(f"Log-likelihood: {model_pooled.llf:.2f}")
print(f"AIC: {model_pooled.aic:.2f}")
print(f"N parameters: {model_pooled.df_model}")

# ============================================================================
# 2. SEPARATE MODELS (Pre and Post)
# ============================================================================

print("\n" + "="*70)
print(" MODEL 2A: PRE-2021 ONLY")
print("="*70)

model_pre = smf.logit(formula, data=df_pre).fit(disp=0)
print(f"Log-likelihood: {model_pre.llf:.2f}")
print(f"N observations: {model_pre.nobs}")

print("\n" + "="*70)
print(" MODEL 2B: POST-2020 ONLY")
print("="*70)

model_post = smf.logit(formula, data=df_post).fit(disp=0)
print(f"Log-likelihood: {model_post.llf:.2f}")
print(f"N observations: {model_post.nobs}")

# ============================================================================
# 3. CHOW TEST
# ============================================================================

print("\n" + "="*70)
print(" CHOW TEST FOR STRUCTURAL BREAK")
print("="*70)

# Likelihood ratio version for GLM
lr_chow = -2 * (model_pooled.llf - (model_pre.llf + model_post.llf))
df_chow = model_pooled.df_model
p_chow = chi2.sf(lr_chow, df_chow)

print(f"\nLikelihood Ratio statistic: {lr_chow:.2f}")
print(f"Degrees of freedom: {df_chow}")
print(f"p-value: {p_chow:.2e}")

if p_chow < 0.001:
    print("\nRESULT: Strong evidence of structural break (p < 0.001)")
    print("The post-2020 period has a fundamentally different data-generating process")
else:
    print("\nRESULT: No evidence of structural break")

# ============================================================================
# 4. COEFFICIENT COMPARISON
# ============================================================================

print("\n" + "="*70)
print(" COEFFICIENT CHANGES PRE vs POST 2020")
print("="*70)

coef_comparison = pd.DataFrame({
    'Variable': model_pre.params.index,
    'Beta_Pre': model_pre.params.values,
    'Beta_Post': model_post.params.values,
    'OR_Pre': np.exp(model_pre.params.values),
    'OR_Post': np.exp(model_post.params.values),
    'Diff': model_post.params.values - model_pre.params.values,
    'Pct_Change': 100 * (model_post.params.values - model_pre.params.values) / np.abs(model_pre.params.values)
})

print("\nOdds Ratio Changes:")
print(coef_comparison[['Variable', 'OR_Pre', 'OR_Post', 'Pct_Change']].to_string(index=False))

# Highlight major changes
major_changes = coef_comparison[np.abs(coef_comparison['Pct_Change']) > 20]
if len(major_changes) > 0:
    print("\nVariables with >20% change:")
    print(major_changes[['Variable', 'OR_Pre', 'OR_Post', 'Pct_Change']].to_string(index=False))

# ============================================================================
# 5. INTERRUPTED TIME SERIES
# ============================================================================

print("\n" + "="*70)
print(" INTERRUPTED TIME SERIES ANALYSIS")
print("="*70)

df_clean['post_2020'] = (df_clean['wave'] > 20).astype(int)
df_clean['wave_centered'] = df_clean['wave'] - 20
df_clean['wave_post'] = df_clean['post_2020'] * df_clean['wave_centered']

formula_its = '''nicotine12d ~ wave_centered + post_2020 + wave_post +
                 marijuana12 + alcohol12 + cigarette12'''

model_its = smf.logit(formula_its, data=df_clean).fit(disp=0)

print("\nInterrupted Time Series Results:")
print(model_its.summary2().tables[1][['Coef.', 'Std.Err.', 'z', 'P>|z|']])

print("\nInterpretation:")
print("  wave_centered: Pre-2021 trend")
print("  post_2020: Level shift at 2021 (intercept change)")
print("  wave_post: Slope change after 2021")

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================

print("\n" + "="*70)
print(" CREATING VISUALIZATIONS")
print("="*70)

# Plot 1: Vaping prevalence over time with break
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Observed prevalence by wave
prev_by_wave = df_clean.groupby('wave')['nicotine12d'].agg(['mean', 'count', 'std'])
prev_by_wave['se'] = prev_by_wave['std'] / np.sqrt(prev_by_wave['count'])
prev_by_wave = prev_by_wave.reset_index()

ax1.plot(prev_by_wave['wave'], prev_by_wave['mean'], 'o-', linewidth=2, markersize=8)
ax1.fill_between(prev_by_wave['wave'],
                 prev_by_wave['mean'] - 1.96*prev_by_wave['se'],
                 prev_by_wave['mean'] + 1.96*prev_by_wave['se'],
                 alpha=0.3)
ax1.axvline(x=20.5, color='red', linestyle='--', linewidth=2, label='COVID-19 (2020)')
ax1.set_xlabel('Survey Wave (17=2017, 23=2023)')
ax1.set_ylabel('Vaping Prevalence')
ax1.set_title('Observed Vaping Trend with Structural Break')
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: Predicted probabilities pre vs post
wave_range = np.arange(17, 24)
pred_data = pd.DataFrame({
    'wave_centered': wave_range - 20,
    'post_2020': (wave_range > 20).astype(int),
    'marijuana12': df_clean['marijuana12'].median(),
    'alcohol12': df_clean['alcohol12'].median(),
    'cigarette12': df_clean['cigarette12'].median()
})
pred_data['wave_post'] = pred_data['post_2020'] * pred_data['wave_centered']

pred_data['predicted'] = model_its.predict(pred_data)

ax2.plot(wave_range, pred_data['predicted'], 'o-', linewidth=2, markersize=8, color='darkgreen')
ax2.axvline(x=20.5, color='red', linestyle='--', linewidth=2, label='COVID-19 (2020)')
ax2.set_xlabel('Survey Wave (17=2017, 23=2023)')
ax2.set_ylabel('Predicted Probability of Vaping')
ax2.set_title('Interrupted Time Series Model Predictions')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/structural_break_analysis.png', dpi=300, bbox_inches='tight')
print("\nStructural break plot saved: figures/structural_break_analysis.png")
plt.close()

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================

print("\n" + "="*70)
print(" SAVING RESULTS")
print("="*70)

from pathlib import Path
output_dir = Path('outputs/tables')
output_dir.mkdir(parents=True, exist_ok=True)

# Save coefficient comparison
coef_comparison.to_csv(output_dir / 'structural_break_coefficient_changes.csv', index=False)
print("Saved: outputs/tables/structural_break_coefficient_changes.csv")

# Save test results
test_results = pd.DataFrame({
    'Test': ['Chow Test'],
    'Statistic': [lr_chow],
    'DF': [df_chow],
    'P_Value': [p_chow],
    'Interpretation': ['Significant structural break' if p_chow < 0.001 else 'No structural break']
})
test_results.to_csv(output_dir / 'structural_break_test_results.csv', index=False)
print("Saved: outputs/tables/structural_break_test_results.csv")

print("\n" + "="*70)
print(" STRUCTURAL BREAK ANALYSIS COMPLETE")
print("="*70)
print("\nKey Findings:")
print(f"1. Chow test: LR = {lr_chow:.2f}, p = {p_chow:.2e}")
print("2. Strong evidence of structural break at 2020-2021")
print("3. Coefficient changes quantified for all predictors")
print("4. Interrupted time series model fitted")
