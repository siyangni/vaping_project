"""
Incremental Predictive Value Analysis

Tests whether adding lower-consensus features significantly improves
predictive performance using DeLong test for AUC comparison.

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
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import norm
import matplotlib.pyplot as plt

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*70)
print(" INCREMENTAL PREDICTIVE VALUE ANALYSIS")
print("="*70)

# Load data
data_path = os.path.expanduser('~/work/vaping_project_data/processed_data_g12n.csv')
df = pd.read_csv(data_path)

TARGET = 'nicotine12d'
df_clean = df[df[TARGET].notna()].copy()

print(f"\nData: {len(df_clean):,} observations")

# Define tier features (adjust based on your actual consensus tiers)
tier_features = {
    1: ['wave', 'marijuana12', 'alcohol12', 'cigarette12', 'political', 'region', 'avg_grade'],
    2: ['wave', 'marijuana12', 'alcohol12', 'cigarette12', 'political', 'region', 'avg_grade',
        'female', 'school_ability', 'fun_evenings'],
    # Add more tiers if available
}

# Train/test split
from sklearn.model_selection import train_test_split

X_full = df_clean[[c for c in df_clean.columns if c != TARGET]]
y = df_clean[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ============================================================================
# FIT NESTED MODELS
# ============================================================================

print("\n" + "="*70)
print(" FITTING NESTED MODELS BY TIER")
print("="*70)

results = []

for tier, features in sorted(tier_features.items()):
    # Filter to available features
    available_features = [f for f in features if f in X_train.columns]

    if len(available_features) == 0:
        print(f"\nTier {tier}: No features available, skipping")
        continue

    print(f"\nTier {tier}: {len(available_features)} features")

    # Prepare data
    X_train_tier = X_train[available_features].fillna(X_train[available_features].median())
    X_test_tier = X_test[available_features].fillna(X_test[available_features].median())

    # Fit logistic regression
    model = sm.GLM(y_train, sm.add_constant(X_train_tier),
                   family=sm.families.Binomial()).fit()

    # Predictions
    y_pred_prob = model.predict(sm.add_constant(X_test_tier))

    # Metrics
    auc = roc_auc_score(y_test, y_pred_prob)

    results.append({
        'Tier': tier,
        'N_Features': len(available_features),
        'AUC': auc,
        'LogLik': model.llf,
        'AIC': model.aic,
        'y_pred_prob': y_pred_prob
    })

    print(f"  AUC: {auc:.4f}")
    print(f"  Log-likelihood: {model.llf:.2f}")

# ============================================================================
# COMPUTE AUC GAINS AND SIGNIFICANCE
# ============================================================================

print("\n" + "="*70)
print(" INCREMENTAL VALUE ANALYSIS")
print("="*70)

results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'y_pred_prob'} for r in results])
results_df['AUC_Gain'] = results_df['AUC'] - results_df.iloc[0]['AUC']

# Simple DeLong test approximation
# For rigorous implementation, use pROC package or dedicated library
# This is a simplified version

def delong_test_simple(y_true, pred1, pred2):
    """Simplified DeLong test for correlated ROC curves"""
    from scipy.stats import norm

    auc1 = roc_auc_score(y_true, pred1)
    auc2 = roc_auc_score(y_true, pred2)

    n = len(y_true)
    # Simplified variance estimation
    se = np.sqrt((auc1 * (1-auc1) + auc2 * (1-auc2)) / n)

    z = (auc2 - auc1) / se
    p_value = 2 * (1 - norm.cdf(abs(z)))

    return p_value

results_df['P_vs_Tier1'] = np.nan

for i in range(1, len(results)):
    if 'y_pred_prob' in results[0] and 'y_pred_prob' in results[i]:
        p_val = delong_test_simple(y_test, results[0]['y_pred_prob'], results[i]['y_pred_prob'])
        results_df.loc[i, 'P_vs_Tier1'] = p_val

print("\nIncremental Value Summary:")
print(results_df[['Tier', 'N_Features', 'AUC', 'AUC_Gain', 'P_vs_Tier1']].to_string(index=False))

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "="*70)
print(" CREATING VISUALIZATIONS")
print("="*70)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: AUC by tier
ax1.plot(results_df['Tier'], results_df['AUC'], 'o-', linewidth=2, markersize=10)
ax1.set_xlabel('Consensus Tier')
ax1.set_ylabel('ROC AUC')
ax1.set_title('Predictive Performance by Consensus Tier')
ax1.grid(alpha=0.3)
ax1.set_xticks(results_df['Tier'])

# Plot 2: AUC gain vs features
ax2.scatter(results_df['N_Features'], results_df['AUC_Gain'], s=100)
ax2.set_xlabel('Number of Features')
ax2.set_ylabel('AUC Gain over Tier 1')
ax2.set_title('Diminishing Returns: Features vs Performance Gain')
ax2.grid(alpha=0.3)
ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)

# Annotate points
for _, row in results_df.iterrows():
    ax2.annotate(f"Tier {int(row['Tier'])}", (row['N_Features'], row['AUC_Gain']),
                xytext=(5, 5), textcoords='offset points', fontsize=10)

plt.tight_layout()
plt.savefig('figures/incremental_value.png', dpi=300, bbox_inches='tight')
print("Plot saved: figures/incremental_value.png")
plt.close()

# ============================================================================
# SAVE RESULTS
# ============================================================================

output_dir = Path('outputs/tables')
output_dir.mkdir(parents=True, exist_ok=True)

results_df.to_csv(output_dir / 'incremental_value_results.csv', index=False)
print("Saved: outputs/tables/incremental_value_results.csv")

print("\n" + "="*70)
print(" INCREMENTAL VALUE ANALYSIS COMPLETE")
print("="*70)

if len(results_df) > 1:
    tier2_gain = results_df.iloc[1]['AUC_Gain']
    tier2_p = results_df.iloc[1]['P_vs_Tier1']

    print(f"\nKey Finding:")
    print(f"Tier 2 provides AUC gain of {tier2_gain:.4f} (p = {tier2_p:.4f})")

    if tier2_p < 0.05:
        print("Tier 2 features significantly improve prediction")
    else:
        print("Tier 2 features do not significantly improve prediction")

    print("\nRecommendation: Use features from tiers with significant gains")
