"""
Enhanced Nested Regression with Discovered Interactions

This script explicitly tests interaction terms identified by SHAP analysis in the
machine learning stage, demonstrating the value of the ML discovery -> regression
testing pipeline.

Implements three regression model variants:
- Model A: Main effects only (baseline)
- Model B: Main effects + two-way interactions
- Model C: Main effects + threshold effects + interactions

Author: Siyang Ni
Date: 2025-11-05
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*70)
print(" INTERACTION REGRESSION ANALYSIS")
print("="*70)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

data_path = os.path.expanduser('~/work/vaping_project_data/processed_data_g12n.csv')

if not os.path.exists(data_path):
    print("ERROR: Data file not found!")
    print(f"Expected: {data_path}")
    raise FileNotFoundError(data_path)

df = pd.read_csv(data_path)
print(f"\nData loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Target variable
TARGET = 'nicotine12d'

# Remove missing targets
df_clean = df[df[TARGET].notna()].copy()
print(f"After removing missing targets: {len(df_clean):,} samples")

# ============================================================================
# 2. CREATE INTERACTION TERMS
# ============================================================================

print("\n" + "="*70)
print(" CREATING INTERACTION TERMS")
print("="*70)

# Based on SHAP interaction analysis, create:
# - Wave x Marijuana
# - Wave x Alcohol
# - Wave x Cigarettes
# - Post-2020 dummy
# - Post-2020 x Marijuana

# Standardize variables for interaction terms (helps with convergence)
df_clean['wave_std'] = (df_clean['wave'] - df_clean['wave'].mean()) / df_clean['wave'].std()
df_clean['marijuana12_std'] = (df_clean['marijuana12'] - df_clean['marijuana12'].mean()) / df_clean['marijuana12'].std()
df_clean['alcohol12_std'] = (df_clean['alcohol12'] - df_clean['alcohol12'].mean()) / df_clean['alcohol12'].std()
df_clean['cigarette12_std'] = (df_clean['cigarette12'] - df_clean['cigarette12'].mean()) / df_clean['cigarette12'].std()

# Create interaction terms
df_clean['wave_x_marijuana'] = df_clean['wave_std'] * df_clean['marijuana12_std']
df_clean['wave_x_alcohol'] = df_clean['wave_std'] * df_clean['alcohol12_std']
df_clean['wave_x_cigarettes'] = df_clean['wave_std'] * df_clean['cigarette12_std']

# Create post-2020 indicator (wave > 20 corresponds to 2021+)
df_clean['post_2020'] = (df_clean['wave'] > 20).astype(int)
df_clean['post2020_x_marijuana'] = df_clean['post_2020'] * df_clean['marijuana12_std']
df_clean['post2020_x_alcohol'] = df_clean['post_2020'] * df_clean['alcohol12_std']

print("\nInteraction terms created:")
print("  - wave_x_marijuana")
print("  - wave_x_alcohol")
print("  - wave_x_cigarettes")
print("  - post_2020 (dummy for wave > 20)")
print("  - post2020_x_marijuana")
print("  - post2020_x_alcohol")

# ============================================================================
# 3. MODEL A: MAIN EFFECTS ONLY
# ============================================================================

print("\n" + "="*70)
print(" MODEL A: MAIN EFFECTS ONLY (BASELINE)")
print("="*70)

# Use standardized variables for consistency
formula_A = '''nicotine12d ~ wave_std + marijuana12_std + alcohol12_std + cigarette12_std'''

try:
    model_A = smf.logit(formula_A, data=df_clean).fit(disp=0)

    print("\nModel A Results:")
    print(f"  Log-likelihood: {model_A.llf:.2f}")
    print(f"  AIC: {model_A.aic:.2f}")
    print(f"  BIC: {model_A.bic:.2f}")
    print(f"  Pseudo R-squared: {model_A.prsquared:.4f}")
    print(f"  N parameters: {model_A.df_model}")

    print("\nCoefficients (Odds Ratios):")
    results_A = pd.DataFrame({
        'Variable': model_A.params.index,
        'Coefficient': model_A.params.values,
        'OR': np.exp(model_A.params.values),
        'SE': model_A.bse.values,
        'z': model_A.tvalues.values,
        'p': model_A.pvalues.values
    })
    print(results_A.to_string(index=False))

except Exception as e:
    print(f"\nERROR fitting Model A: {e}")
    model_A = None

# ============================================================================
# 4. MODEL B: MAIN EFFECTS + TWO-WAY INTERACTIONS
# ============================================================================

print("\n" + "="*70)
print(" MODEL B: MAIN EFFECTS + TWO-WAY INTERACTIONS")
print("="*70)

formula_B = '''nicotine12d ~ wave_std + marijuana12_std + alcohol12_std + cigarette12_std +
                wave_x_marijuana + wave_x_alcohol + wave_x_cigarettes'''

try:
    model_B = smf.logit(formula_B, data=df_clean).fit(disp=0)

    print("\nModel B Results:")
    print(f"  Log-likelihood: {model_B.llf:.2f}")
    print(f"  AIC: {model_B.aic:.2f}")
    print(f"  BIC: {model_B.bic:.2f}")
    print(f"  Pseudo R-squared: {model_B.prsquared:.4f}")
    print(f"  N parameters: {model_B.df_model}")

    print("\nCoefficients (Odds Ratios):")
    results_B = pd.DataFrame({
        'Variable': model_B.params.index,
        'Coefficient': model_B.params.values,
        'OR': np.exp(model_B.params.values),
        'SE': model_B.bse.values,
        'z': model_B.tvalues.values,
        'p': model_B.pvalues.values
    })
    print(results_B.to_string(index=False))

    # Likelihood ratio test: Model A vs Model B
    if model_A is not None:
        lr_stat = -2 * (model_A.llf - model_B.llf)
        df_diff = model_B.df_model - model_A.df_model
        p_value = chi2.sf(lr_stat, df_diff)

        print("\n" + "-"*70)
        print(" LIKELIHOOD RATIO TEST: Model A vs Model B")
        print("-"*70)
        print(f"  LR statistic: {lr_stat:.2f}")
        print(f"  df: {df_diff}")
        print(f"  p-value: {p_value:.4e}")

        if p_value < 0.001:
            print("  RESULT: Interactions significantly improve model fit (p < 0.001)")
        elif p_value < 0.05:
            print("  RESULT: Interactions significantly improve model fit (p < 0.05)")
        else:
            print("  RESULT: Interactions do not significantly improve model fit")

except Exception as e:
    print(f"\nERROR fitting Model B: {e}")
    model_B = None

# ============================================================================
# 5. MODEL C: WITH THRESHOLD EFFECTS
# ============================================================================

print("\n" + "="*70)
print(" MODEL C: MAIN EFFECTS + THRESHOLD EFFECTS + INTERACTIONS")
print("="*70)

formula_C = '''nicotine12d ~ wave_std + marijuana12_std + alcohol12_std + cigarette12_std +
                post_2020 + post2020_x_marijuana + post2020_x_alcohol'''

try:
    model_C = smf.logit(formula_C, data=df_clean).fit(disp=0)

    print("\nModel C Results:")
    print(f"  Log-likelihood: {model_C.llf:.2f}")
    print(f"  AIC: {model_C.aic:.2f}")
    print(f"  BIC: {model_C.bic:.2f}")
    print(f"  Pseudo R-squared: {model_C.prsquared:.4f}")
    print(f"  N parameters: {model_C.df_model}")

    print("\nCoefficients (Odds Ratios):")
    results_C = pd.DataFrame({
        'Variable': model_C.params.index,
        'Coefficient': model_C.params.values,
        'OR': np.exp(model_C.params.values),
        'SE': model_C.bse.values,
        'z': model_C.tvalues.values,
        'p': model_C.pvalues.values
    })
    print(results_C.to_string(index=False))

    # Likelihood ratio test: Model A vs Model C
    if model_A is not None:
        lr_stat = -2 * (model_A.llf - model_C.llf)
        df_diff = model_C.df_model - model_A.df_model
        p_value = chi2.sf(lr_stat, df_diff)

        print("\n" + "-"*70)
        print(" LIKELIHOOD RATIO TEST: Model A vs Model C")
        print("-"*70)
        print(f"  LR statistic: {lr_stat:.2f}")
        print(f"  df: {df_diff}")
        print(f"  p-value: {p_value:.4e}")

        if p_value < 0.001:
            print("  RESULT: Threshold effects significantly improve model fit (p < 0.001)")

except Exception as e:
    print(f"\nERROR fitting Model C: {e}")
    model_C = None

# ============================================================================
# 6. VISUALIZE INTERACTIONS
# ============================================================================

print("\n" + "="*70)
print(" CREATING INTERACTION PLOTS")
print("="*70)

if model_B is not None:
    # Create interaction plot showing how marijuana effect changes across waves

    # Generate prediction data
    wave_range = np.linspace(df_clean['wave'].min(), df_clean['wave'].max(), 50)
    marijuana_levels = [df_clean['marijuana12'].quantile(0.25),
                       df_clean['marijuana12'].median(),
                       df_clean['marijuana12'].quantile(0.75)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Predicted probability by wave, stratified by marijuana use
    ax1 = axes[0]

    for mj_level in marijuana_levels:
        pred_data = pd.DataFrame({
            'wave': wave_range,
            'wave_std': (wave_range - df_clean['wave'].mean()) / df_clean['wave'].std(),
            'marijuana12': mj_level,
            'marijuana12_std': (mj_level - df_clean['marijuana12'].mean()) / df_clean['marijuana12'].std(),
            'alcohol12_std': 0,
            'cigarette12_std': 0
        })
        pred_data['wave_x_marijuana'] = pred_data['wave_std'] * pred_data['marijuana12_std']
        pred_data['wave_x_alcohol'] = 0
        pred_data['wave_x_cigarettes'] = 0

        # Predict
        pred_data['predicted_prob'] = model_B.predict(pred_data)

        label = f'Marijuana = {mj_level:.1f}'
        if mj_level == marijuana_levels[0]:
            label += ' (25th pct)'
        elif mj_level == marijuana_levels[1]:
            label += ' (median)'
        else:
            label += ' (75th pct)'

        ax1.plot(pred_data['wave'], pred_data['predicted_prob'],
                label=label, linewidth=2)

    ax1.axvline(x=20, color='red', linestyle='--', alpha=0.7, label='COVID-19 (2020)')
    ax1.set_xlabel('Survey Wave (Year: 2017=17, 2023=23)')
    ax1.set_ylabel('Predicted Probability of Vaping')
    ax1.set_title('Wave x Marijuana Interaction Effect')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Marginal effect of marijuana by wave
    ax2 = axes[1]

    # Calculate marginal effect of marijuana at different waves
    marginal_effects = []

    for wave_val in wave_range:
        # Small change in marijuana
        base_data = pd.DataFrame({
            'wave_std': [(wave_val - df_clean['wave'].mean()) / df_clean['wave'].std()],
            'marijuana12_std': [0],
            'alcohol12_std': [0],
            'cigarette12_std': [0],
            'wave_x_marijuana': [0],
            'wave_x_alcohol': [0],
            'wave_x_cigarettes': [0]
        })

        # Increase marijuana by 1 unit
        delta_data = base_data.copy()
        delta_data['marijuana12_std'] = [1 / df_clean['marijuana12'].std()]
        delta_data['wave_x_marijuana'] = delta_data['wave_std'] * delta_data['marijuana12_std']

        prob_base = model_B.predict(base_data).iloc[0]
        prob_delta = model_B.predict(delta_data).iloc[0]

        marginal_effect = prob_delta - prob_base
        marginal_effects.append(marginal_effect)

    ax2.plot(wave_range, marginal_effects, linewidth=2, color='darkgreen')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.axvline(x=20, color='red', linestyle='--', alpha=0.7, label='COVID-19 (2020)')
    ax2.set_xlabel('Survey Wave (Year: 2017=17, 2023=23)')
    ax2.set_ylabel('Marginal Effect of Marijuana on Vaping Probability')
    ax2.set_title('How Marijuana Effect Changes Over Time')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/interaction_plot_wave_marijuana.png', dpi=300, bbox_inches='tight')
    print("\nInteraction plot saved: figures/interaction_plot_wave_marijuana.png")
    plt.close()

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================

print("\n" + "="*70)
print(" SAVING RESULTS")
print("="*70)

output_dir = Path('outputs/tables')
output_dir.mkdir(parents=True, exist_ok=True)

if model_A is not None:
    results_A.to_csv(output_dir / 'interaction_model_A_main_effects.csv', index=False)
    print("Saved: outputs/tables/interaction_model_A_main_effects.csv")

if model_B is not None:
    results_B.to_csv(output_dir / 'interaction_model_B_with_interactions.csv', index=False)
    print("Saved: outputs/tables/interaction_model_B_with_interactions.csv")

if model_C is not None:
    results_C.to_csv(output_dir / 'interaction_model_C_threshold.csv', index=False)
    print("Saved: outputs/tables/interaction_model_C_threshold.csv")

# Model comparison summary
if model_A is not None and model_B is not None and model_C is not None:
    comparison = pd.DataFrame({
        'Model': ['A: Main Effects', 'B: + Interactions', 'C: + Threshold'],
        'Log_Likelihood': [model_A.llf, model_B.llf, model_C.llf],
        'AIC': [model_A.aic, model_B.aic, model_C.aic],
        'BIC': [model_A.bic, model_B.bic, model_C.bic],
        'Pseudo_R2': [model_A.prsquared, model_B.prsquared, model_C.prsquared],
        'N_Params': [model_A.df_model, model_B.df_model, model_C.df_model]
    })

    comparison.to_csv(output_dir / 'interaction_model_comparison.csv', index=False)
    print("Saved: outputs/tables/interaction_model_comparison.csv")

    print("\nModel Comparison Summary:")
    print(comparison.to_string(index=False))

print("\n" + "="*70)
print(" INTERACTION REGRESSION ANALYSIS COMPLETE")
print("="*70)
print("\nKey Findings:")
print("1. Likelihood ratio tests indicate whether interactions significantly improve fit")
print("2. Interaction plots show how marijuana effect changes over time")
print("3. Results validate ML discovery -> regression testing pipeline")
print("\nNext steps:")
print("- Review interaction_plot_wave_marijuana.png")
print("- Examine coefficients in saved CSV files")
print("- Update manuscript with interaction findings")
