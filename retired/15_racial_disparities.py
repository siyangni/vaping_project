"""
Racial/Ethnic Disparity Analysis

Quantifies racial and ethnic disparities in vaping prevalence and tests
whether disparities are widening or narrowing over time.

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
import seaborn as sns

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*70)
print(" RACIAL/ETHNIC DISPARITY ANALYSIS")
print("="*70)

# Load data
data_path = os.path.expanduser('~/work/vaping_project_data/processed_data_g12n.csv')
df = pd.read_csv(data_path)
df_clean = df[df['nicotine12d'].notna()].copy()

print(f"\nData: {len(df_clean):,} observations")

# Identify race variable
race_var = None
for col in ['race', 'race_ethnicity', 'V2150']:
    if col in df_clean.columns:
        race_var = col
        break

if race_var:
    print(f"\nRace/ethnicity distribution ({race_var}):")
    print(df_clean[race_var].value_counts())

    # Create race dummies (adjust based on actual coding)
    # This is a template - adjust based on actual data
    if df_clean[race_var].dtype == 'object':
        df_clean['white'] = (df_clean[race_var].str.contains('White', case=False, na=False)).astype(int)
        df_clean['black'] = (df_clean[race_var].str.contains('Black', case=False, na=False)).astype(int)
        df_clean['hispanic'] = (df_clean[race_var].str.contains('Hispanic', case=False, na=False)).astype(int)
        df_clean['asian'] = (df_clean[race_var].str.contains('Asian', case=False, na=False)).astype(int)
    else:
        # Numeric coding - adjust based on codebook
        df_clean['white'] = (df_clean[race_var] == 1).astype(int)
        df_clean['black'] = (df_clean[race_var] == 2).astype(int)
        df_clean['hispanic'] = (df_clean[race_var] == 3).astype(int)
        df_clean['asian'] = (df_clean[race_var] == 4).astype(int)

    # Model with race effects
    print("\n" + "="*70)
    print(" MODEL 1: RACIAL DISPARITIES (White as reference)")
    print("="*70)

    formula_race = 'nicotine12d ~ black + hispanic + asian + marijuana12 + alcohol12 + wave'
    model_race = smf.logit(formula_race, data=df_clean).fit(disp=0)

    print("\nResults:")
    print(model_race.summary2().tables[1][['Coef.', 'Std.Err.', 'z', 'P>|z|']])

    print("\nOdds Ratios (vs White):")
    for var in ['black', 'hispanic', 'asian']:
        if var in model_race.params.index:
            or_val = np.exp(model_race.params[var])
            print(f"  {var.capitalize()}: OR = {or_val:.3f}")

    # Test race x wave interactions
    print("\n" + "="*70)
    print(" MODEL 2: TESTING TEMPORAL TRENDS IN DISPARITIES")
    print("="*70)

    df_clean['black_x_wave'] = df_clean['black'] * df_clean['wave']
    df_clean['hispanic_x_wave'] = df_clean['hispanic'] * df_clean['wave']
    df_clean['asian_x_wave'] = df_clean['asian'] * df_clean['wave']

    formula_trend = formula_race + ' + black_x_wave + hispanic_x_wave + asian_x_wave'
    model_trend = smf.logit(formula_trend, data=df_clean).fit(disp=0)

    lr_stat = -2 * (model_race.llf - model_trend.llf)
    df_diff = 3
    p_value = chi2.sf(lr_stat, df_diff)

    print(f"\nLR test for race x wave interactions:")
    print(f"  LR = {lr_stat:.2f}, df = {df_diff}, p = {p_value:.4f}")

    if p_value < 0.05:
        print("  Racial disparities are changing over time")
    else:
        print("  Racial disparities are stable over time")

    # Prevalence by race and wave
    print("\n" + "="*70)
    print(" VAPING PREVALENCE BY RACE/ETHNICITY AND WAVE")
    print("="*70)

    # Create race categories
    df_clean['race_category'] = 'Other'
    df_clean.loc[df_clean['white'] == 1, 'race_category'] = 'White'
    df_clean.loc[df_clean['black'] == 1, 'race_category'] = 'Black'
    df_clean.loc[df_clean['hispanic'] == 1, 'race_category'] = 'Hispanic'
    df_clean.loc[df_clean['asian'] == 1, 'race_category'] = 'Asian'

    prev_by_race_wave = df_clean.groupby(['wave', 'race_category'])['nicotine12d'].agg(['mean', 'count', 'std']).reset_index()
    prev_by_race_wave['se'] = prev_by_race_wave['std'] / np.sqrt(prev_by_race_wave['count'])

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))

    for race in ['White', 'Black', 'Hispanic', 'Asian']:
        subset = prev_by_race_wave[prev_by_race_wave['race_category'] == race]
        if len(subset) > 0:
            ax.plot(subset['wave'], subset['mean'], marker='o', label=race, linewidth=2)
            ax.fill_between(subset['wave'],
                           subset['mean'] - 1.96*subset['se'],
                           subset['mean'] + 1.96*subset['se'],
                           alpha=0.2)

    ax.axvline(x=20, color='red', linestyle='--', alpha=0.7, label='COVID-19 (2020)')
    ax.set_xlabel('Survey Wave (17=2017, 23=2023)')
    ax.set_ylabel('Vaping Prevalence')
    ax.set_title('Racial/Ethnic Disparities in Vaping Over Time')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/racial_disparities_trends.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved: figures/racial_disparities_trends.png")
    plt.close()

    # Save results
    from pathlib import Path
    output_dir = Path('outputs/tables')
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame({
        'Race_Ethnicity': ['Black', 'Hispanic', 'Asian'],
        'OR_vs_White': [np.exp(model_race.params.get(r, np.nan)) for r in ['black', 'hispanic', 'asian']],
        'p_value': [model_race.pvalues.get(r, np.nan) for r in ['black', 'hispanic', 'asian']]
    })

    results_df.to_csv(output_dir / 'racial_disparities_results.csv', index=False)
    print("\nSaved: outputs/tables/racial_disparities_results.csv")

else:
    print("\nWARNING: Race/ethnicity variable not found in dataset")

print("\n" + "="*70)
print(" RACIAL DISPARITY ANALYSIS COMPLETE")
print("="*70)
