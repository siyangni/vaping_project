"""
Gender Interaction Analysis

Tests whether marijuana and alcohol effects on vaping vary by gender,
addressing core criminological questions about gender-specific substance
use pathways.

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
print(" GENDER INTERACTION ANALYSIS")
print("="*70)

# Load data
data_path = os.path.expanduser('~/work/vaping_project_data/processed_data_g12n.csv')
df = pd.read_csv(data_path)
df_clean = df[df['nicotine12d'].notna()].copy()

print(f"\nData: {len(df_clean):,} observations")

# Check gender variable
if 'female' in df_clean.columns:
    gender_var = 'female'
elif 'sex' in df_clean.columns:
    df_clean['female'] = (df_clean['sex'] == 2).astype(int)
    gender_var = 'female'
else:
    print("WARNING: Gender variable not found")
    gender_var = None

if gender_var:
    print(f"\nGender distribution:")
    print(df_clean[gender_var].value_counts())

    # Create interaction terms
    df_clean['female_x_marijuana'] = df_clean[gender_var] * df_clean['marijuana12']
    df_clean['female_x_alcohol'] = df_clean[gender_var] * df_clean['alcohol12']

    # Main effects model
    print("\n" + "="*70)
    print(" MODEL A: MAIN EFFECTS ONLY")
    print("="*70)

    formula_main = f'nicotine12d ~ {gender_var} + marijuana12 + alcohol12 + wave + cigarette12'
    model_main = smf.logit(formula_main, data=df_clean).fit(disp=0)
    print(f"Log-likelihood: {model_main.llf:.2f}")
    print(f"AIC: {model_main.aic:.2f}")

    # Interaction model
    print("\n" + "="*70)
    print(" MODEL B: WITH GENDER INTERACTIONS")
    print("="*70)

    formula_int = f'nicotine12d ~ {gender_var} + marijuana12 + alcohol12 + female_x_marijuana + female_x_alcohol + wave + cigarette12'
    model_int = smf.logit(formula_int, data=df_clean).fit(disp=0)
    print(f"Log-likelihood: {model_int.llf:.2f}")
    print(f"AIC: {model_int.aic:.2f}")

    print("\nCoefficients:")
    print(model_int.summary2().tables[1][['Coef.', 'Std.Err.', 'z', 'P>|z|']])

    # LR test
    lr_stat = -2 * (model_main.llf - model_int.llf)
    df_diff = model_int.df_model - model_main.df_model
    p_value = chi2.sf(lr_stat, df_diff)

    print(f"\nLikelihood Ratio Test: LR = {lr_stat:.2f}, df = {df_diff}, p = {p_value:.4f}")

    if p_value < 0.001:
        print("Gender interactions are highly significant (p < 0.001)")

    # Stratified analysis
    print("\n" + "="*70)
    print(" STRATIFIED ANALYSIS BY GENDER")
    print("="*70)

    df_male = df_clean[df_clean[gender_var] == 0]
    df_female = df_clean[df_clean[gender_var] == 1]

    formula_strat = 'nicotine12d ~ marijuana12 + alcohol12 + wave + cigarette12'

    model_male = smf.logit(formula_strat, data=df_male).fit(disp=0)
    model_female = smf.logit(formula_strat, data=df_female).fit(disp=0)

    print("\nMales:")
    print(f"  N = {len(df_male):,}")
    print(f"  Marijuana OR = {np.exp(model_male.params['marijuana12']):.3f}")
    print(f"  Alcohol OR = {np.exp(model_male.params['alcohol12']):.3f}")

    print("\nFemales:")
    print(f"  N = {len(df_female):,}")
    print(f"  Marijuana OR = {np.exp(model_female.params['marijuana12']):.3f}")
    print(f"  Alcohol OR = {np.exp(model_female.params['alcohol12']):.3f}")

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))

    vars_compare = ['marijuana12', 'alcohol12', 'cigarette12', 'wave']
    x_pos = np.arange(len(vars_compare))

    or_male = [np.exp(model_male.params[v]) for v in vars_compare]
    or_female = [np.exp(model_female.params[v]) for v in vars_compare]

    width = 0.35
    ax.bar(x_pos - width/2, or_male, width, label='Males', alpha=0.8)
    ax.bar(x_pos + width/2, or_female, width, label='Females', alpha=0.8)

    ax.axhline(y=1, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Variable')
    ax.set_ylabel('Odds Ratio')
    ax.set_title('Gender-Stratified Effects on Vaping')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(vars_compare)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/gender_stratified_effects.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved: figures/gender_stratified_effects.png")
    plt.close()

    # Save results
    from pathlib import Path
    output_dir = Path('outputs/tables')
    output_dir.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame({
        'Gender': ['Male', 'Female'],
        'N': [len(df_male), len(df_female)],
        'Marijuana_OR': [np.exp(model_male.params['marijuana12']), np.exp(model_female.params['marijuana12'])],
        'Alcohol_OR': [np.exp(model_male.params['alcohol12']), np.exp(model_female.params['alcohol12'])]
    })

    results.to_csv(output_dir / 'gender_stratified_results.csv', index=False)
    print("\nSaved: outputs/tables/gender_stratified_results.csv")

print("\n" + "="*70)
print(" GENDER INTERACTION ANALYSIS COMPLETE")
print("="*70)
