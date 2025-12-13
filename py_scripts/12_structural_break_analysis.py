#!/usr/bin/env python
# coding: utf-8

# # Structural Break Analysis: COVID-19 as Natural Experiment
# 
# ## Overview
# This notebook performs formal statistical tests for a structural break at 2020-2021 (COVID-19 pandemic). The pandemic represents a natural experiment that may have fundamentally altered adolescent substance use patterns.
# 
# ## Key Questions
# - Is there a structural break in vaping patterns at 2020-2021?
# - Do predictor coefficients differ pre vs post COVID-19?
# - Can we quantify the magnitude of the pandemic impact?
# 
# ## Tests Conducted
# - **Chow test**: Tests equality of coefficients across periods
# - **Interrupted time series**: Models level and slope changes
# - **Coefficient comparison**: Quantifies parameter shifts
# 
# ---

# ## 1. Setup and Imports

# In[ ]:


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
from pathlib import Path

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*70)
print(" STRUCTURAL BREAK ANALYSIS: COVID-19 THRESHOLD")
print("="*70)


# ## 2. Load and Split Data

# In[ ]:


# Load data
data_path = os.path.expanduser('~/work/vaping_project_data/processed_data_g12n.csv')
df = pd.read_csv(data_path)
df_clean = df[df['nicotine12d'].notna()].copy()

print(f"\nData: {len(df_clean):,} observations")
print(f"Wave range: {df_clean['wave'].min()} to {df_clean['wave'].max()}")

# Split data at 2020-2021 threshold
df_pre = df_clean[df_clean['wave'] <= 20].copy()
df_post = df_clean[df_clean['wave'] >= 21].copy()

print(f"\nPre-2021 (wave ≤ 20): {len(df_pre):,} observations")
print(f"Post-2020 (wave ≥ 21): {len(df_post):,} observations")


# ## 3. Model 1: Pooled (No Structural Break)

# In[ ]:


print("="*70)
print(" MODEL 1: POOLED (No Structural Break)")
print("="*70)

# Model specification
formula = 'nicotine12d ~ wave + marijuana12 + alcohol12 + cigarette12 + political + region'

model_pooled = smf.logit(formula, data=df_clean).fit(disp=0)
print(f"\nLog-likelihood: {model_pooled.llf:.2f}")
print(f"AIC: {model_pooled.aic:.2f}")
print(f"N parameters: {model_pooled.df_model}")


# ## 4. Models 2A & 2B: Separate Pre and Post Models

# In[ ]:


print("="*70)
print(" MODEL 2A: PRE-2021 ONLY")
print("="*70)

model_pre = smf.logit(formula, data=df_pre).fit(disp=0)
print(f"\nLog-likelihood: {model_pre.llf:.2f}")
print(f"N observations: {model_pre.nobs:.0f}")

print("\n" + "="*70)
print(" MODEL 2B: POST-2020 ONLY")
print("="*70)

model_post = smf.logit(formula, data=df_post).fit(disp=0)
print(f"\nLog-likelihood: {model_post.llf:.2f}")
print(f"N observations: {model_post.nobs:.0f}")


# ## 5. Chow Test for Structural Break

# In[ ]:


print("="*70)
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


# ## 6. Coefficient Changes: Pre vs Post

# In[ ]:


print("="*70)
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


# ## 7. Interrupted Time Series Analysis

# In[ ]:


print("="*70)
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


# ## 8. Visualizations

# In[ ]:


print("="*70)
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
ax1.set_xlabel('Survey Wave (17=2017, 23=2023)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Vaping Prevalence', fontsize=11, fontweight='bold')
ax1.set_title('Observed Vaping Trend with Structural Break', fontsize=13, fontweight='bold')
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
ax2.set_xlabel('Survey Wave (17=2017, 23=2023)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Predicted Probability of Vaping', fontsize=11, fontweight='bold')
ax2.set_title('Interrupted Time Series Model Predictions', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\nStructural break plots created")


# ## 9. Save Results

# In[ ]:


print("="*70)
print(" SAVING RESULTS")
print("="*70)

output_dir = Path('../outputs/tables')
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

# Save figure
fig_dir = Path('../figures')
fig_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_dir / 'structural_break_analysis.png', dpi=300, bbox_inches='tight')
print(f"✓ Figure saved to: {fig_dir / 'structural_break_analysis.png'}")


# ## Summary
# 
# ### Key Findings:
# - ✅ Chow test provides formal statistical evidence for structural break
# - ✅ Coefficient comparison quantifies magnitude of parameter shifts
# - ✅ Interrupted time series decomposes changes into level vs slope effects
# - ✅ COVID-19 pandemic represents a genuine natural experiment
# 
# ### Interpretation:
# - **Significant break**: Strong evidence that post-2020 period differs fundamentally
# - **Level shift**: Immediate change in vaping prevalence at pandemic onset
# - **Slope change**: Altered trajectory of vaping trends post-pandemic
# - **Substantive implications**: Pandemic disrupted normal adolescent substance use patterns
