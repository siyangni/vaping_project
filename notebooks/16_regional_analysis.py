"""
Regional Variation Analysis

Examines regional differences in vaping prevalence and tests whether
regional trends are diverging over time.

Author: Siyang Ni
Date: 2025-11-05
"""

import pandas as pd
import numpy as np
import os
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

data_path = os.path.expanduser('~/work/vaping_project_data/processed_data_g12n.csv')
df = pd.read_csv(data_path)
df_clean = df[df['nicotine12d'].notna()].copy()

print(f"\nData: {len(df_clean):,} observations")

# Check for region variable
if 'region' in df_clean.columns:
    print("\nRegion distribution:")
    print(df_clean['region'].value_counts().sort_index())

    # Create region dummies (1=NE, 2=MW, 3=S, 4=W typically)
    df_clean['northeast'] = (df_clean['region'] == 1).astype(int)
    df_clean['midwest'] = (df_clean['region'] == 2).astype(int)
    df_clean['south'] = (df_clean['region'] == 3).astype(int)
    # West as reference

    # Model with regional effects
    print("\n" + "="*70)
    print(" MODEL 1: REGIONAL EFFECTS (West as reference)")
    print("="*70)

    formula_region = 'nicotine12d ~ northeast + midwest + south + wave + marijuana12 + alcohol12'
    model_region = smf.logit(formula_region, data=df_clean).fit(disp=0)

    print(model_region.summary2().tables[1][['Coef.', 'Std.Err.', 'z', 'P>|z|']])

    # Test region x wave interactions
    df_clean['northeast_x_wave'] = df_clean['northeast'] * df_clean['wave']
    df_clean['midwest_x_wave'] = df_clean['midwest'] * df_clean['wave']
    df_clean['south_x_wave'] = df_clean['south'] * df_clean['wave']

    formula_trend = formula_region + ' + northeast_x_wave + midwest_x_wave + south_x_wave'
    model_trend = smf.logit(formula_trend, data=df_clean).fit(disp=0)

    lr_stat = -2 * (model_region.llf - model_trend.llf)
    p_value = chi2.sf(lr_stat, 3)

    print(f"\nLR test for region x wave: LR = {lr_stat:.2f}, p = {p_value:.4f}")

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
    ax.set_xlabel('Survey Wave')
    ax.set_ylabel('Vaping Prevalence')
    ax.set_title('Regional Vaping Trends 2017-2023')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/regional_trends.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved: figures/regional_trends.png")
    plt.close()

    # Save results
    from pathlib import Path
    output_dir = Path('outputs/tables')
    output_dir.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame({
        'Region': ['Northeast', 'Midwest', 'South', 'West (ref)'],
        'OR': [np.exp(model_region.params.get('northeast', 0)),
               np.exp(model_region.params.get('midwest', 0)),
               np.exp(model_region.params.get('south', 0)),
               1.0]
    })

    results.to_csv(output_dir / 'regional_effects.csv', index=False)
    print("Saved: outputs/tables/regional_effects.csv")

else:
    print("\nWARNING: Region variable not found")

print("\n" + "="*70)
print(" REGIONAL ANALYSIS COMPLETE")
print("="*70)
