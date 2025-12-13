#!/usr/bin/env python
# coding: utf-8

# # Regional Variation Analysis
# 
# ## Overview
# This notebook examines regional differences in vaping prevalence and tests whether regional trends are diverging over time.
# 
# ## Key Questions
# - Are there significant regional differences in vaping?
# - Are regional trends diverging or converging?
# - What drives regional variation?
# 
# ---

# ## 1. Setup and Imports

# In[ ]:


import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import statsmodels.formula.api as smf
from scipy.stats import chi2
import matplotlib.pyplot as plt

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*70)
print(" REGIONAL VARIATION ANALYSIS")
print("="*70)


# ## 2. Load Data and Check Region Variable

# In[ ]:


data_path = os.path.expanduser('~/work/vaping_project_data/processed_data_g12n.csv')
df = pd.read_csv(data_path)
df_clean = df[df['nicotine12d'].notna()].copy()

print(f"\nData: {len(df_clean):,} observations")

# Check for region variable
if 'region' in df_clean.columns:
    print("\nRegion distribution:")
    print(df_clean['region'].value_counts().sort_index())
else:
    print("\nWARNING: Region variable not found")


# ## 3. Create Region Dummies

# In[ ]:


if 'region' in df_clean.columns:
    # Create region dummies (1=NE, 2=MW, 3=S, 4=W typically)
    df_clean['northeast'] = (df_clean['region'] == 1).astype(int)
    df_clean['midwest'] = (df_clean['region'] == 2).astype(int)
    df_clean['south'] = (df_clean['region'] == 3).astype(int)
    # West as reference
    
    print("\nRegion dummies created (West as reference)")


# ## 4. Model 1: Regional Effects

# In[ ]:


if 'region' in df_clean.columns:
    print("="*70)
    print(" MODEL 1: REGIONAL EFFECTS (West as reference)")
    print("="*70)

    formula_region = 'nicotine12d ~ northeast + midwest + south + wave + marijuana12 + alcohol12'
    model_region = smf.logit(formula_region, data=df_clean).fit(disp=0)

    print("\nRegression Results:")
    print(model_region.summary2().tables[1][['Coef.', 'Std.Err.', 'z', 'P>|z|']])


# ## 5. Model 2: Test Region × Wave Interactions

# In[ ]:


if 'region' in df_clean.columns:
    # Test region x wave interactions
    df_clean['northeast_x_wave'] = df_clean['northeast'] * df_clean['wave']
    df_clean['midwest_x_wave'] = df_clean['midwest'] * df_clean['wave']
    df_clean['south_x_wave'] = df_clean['south'] * df_clean['wave']

    formula_trend = formula_region + ' + northeast_x_wave + midwest_x_wave + south_x_wave'
    model_trend = smf.logit(formula_trend, data=df_clean).fit(disp=0)

    lr_stat = -2 * (model_region.llf - model_trend.llf)
    p_value = chi2.sf(lr_stat, 3)

    print(f"\nLR test for region x wave: LR = {lr_stat:.2f}, p = {p_value:.4f}")
    
    if p_value < 0.05:
        print("Regional trends are diverging over time")
    else:
        print("Regional trends are parallel")


# ## 6. Visualization: Regional Trends

# In[ ]:


if 'region' in df_clean.columns:
    # Regional trends plot
    region_labels = {1: 'Northeast', 2: 'Midwest', 3: 'South', 4: 'West'}
    prev_by_region_wave = df_clean.groupby(['wave', 'region'])['nicotine12d'].agg(['mean', 'count', 'std']).reset_index()
    prev_by_region_wave['se'] = prev_by_region_wave['std'] / np.sqrt(prev_by_region_wave['count'])

    fig, ax = plt.subplots(figsize=(12, 6))

    for region_code, region_name in region_labels.items():
        subset = prev_by_region_wave[prev_by_region_wave['region'] == region_code]
        if len(subset) > 0:
            ax.plot(subset['wave'], subset['mean'], marker='o', label=region_name, linewidth=2)
            ax.fill_between(subset['wave'],
                           subset['mean'] - 1.96*subset['se'],
                           subset['mean'] + 1.96*subset['se'],
                           alpha=0.2)

    ax.axvline(x=20, color='red', linestyle='--', alpha=0.7, label='COVID-19')
    ax.set_xlabel('Survey Wave', fontsize=11, fontweight='bold')
    ax.set_ylabel('Vaping Prevalence', fontsize=11, fontweight='bold')
    ax.set_title('Regional Vaping Trends 2017-2023', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    print("\nRegional trends plot created")


# ## 7. Save Results

# In[ ]:


if 'region' in df_clean.columns:
    output_dir = Path('../outputs/tables')
    output_dir.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame({
        'Region': ['Northeast', 'Midwest', 'South', 'West (ref)'],
        'OR': [np.exp(model_region.params.get('northeast', 0)),
               np.exp(model_region.params.get('midwest', 0)),
               np.exp(model_region.params.get('south', 0)),
               1.0]
    })

    results.to_csv(output_dir / 'regional_effects.csv', index=False)
    print("\n✓ Results saved to: outputs/tables/regional_effects.csv")
    
    # Save figure
    fig_dir = Path('../figures')
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / 'regional_trends.png', dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved to: {fig_dir / 'regional_trends.png'}")

else:
    print("\nWARNING: Region variable not found")


# ## Summary
# 
# ### Key Findings:
# - ✅ Quantified regional variation in vaping prevalence
# - ✅ Tested whether regional trends are diverging
# - ✅ Visualized geographic patterns over time
# - ✅ Results inform region-specific prevention strategies
# 
# ### Interpretation:
# - **Regional differences**: Odds ratios vs West (reference)
# - **Temporal stability**: Tests parallel vs diverging trends
# - **Policy implications**: Supports geographically-targeted interventions
# - **Contextual factors**: Regional variation may reflect policy/cultural differences
