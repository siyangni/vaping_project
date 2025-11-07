"""
SHAP Dependence Plots for Top Interactions

Visualizes HOW interactions manifest, showing how effect of one variable
changes across values of another variable.

Author: Siyang Ni
Date: 2025-11-05
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

import shap
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*70)
print(" SHAP DEPENDENCE PLOTS")
print("="*70)

# Load data
data_path = os.path.expanduser('~/work/vaping_project_data/processed_data_g12n.csv')
df = pd.read_csv(data_path)

TARGET = 'nicotine12d'
df_clean = df[df[TARGET].notna()].copy()

exclude_cols = [TARGET, 'V1'] if 'V1' in df.columns else [TARGET]
features = [c for c in df.columns if c not in exclude_cols]

X = df_clean[features].fillna(df_clean[features].median())

# Subsample for computational efficiency
if len(X) > 5000:
    X_sample = X.sample(5000, random_state=RANDOM_STATE)
    print(f"\nUsing sample of {len(X_sample):,} observations")
else:
    X_sample = X

# Load model
models_dir = Path('outputs/models')
model_path = models_dir / 'xgboost.joblib'

if not model_path.exists():
    print(f"\nERROR: XGBoost model not found")
    exit(1)

model = joblib.load(model_path)
print("Loaded: XGBoost model")

# Compute SHAP values
print("\nComputing SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)
print("SHAP values computed")

# ============================================================================
# DEPENDENCE PLOTS
# ============================================================================

print("\n" + "="*70)
print(" CREATING DEPENDENCE PLOTS")
print("="*70)

# Plot 1: Wave x Marijuana
if 'wave' in X_sample.columns and 'marijuana12' in X_sample.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.dependence_plot(
        "wave",
        shap_values,
        X_sample,
        interaction_index="marijuana12",
        show=False,
        ax=ax
    )
    plt.title("SHAP Interaction: Survey Wave x Marijuana Use", fontsize=14)
    plt.xlabel("Survey Wave (Year)", fontsize=12)
    plt.ylabel("SHAP Value for Wave\n(Impact on Vaping Prediction)", fontsize=12)
    plt.tight_layout()
    plt.savefig('figures/shap_wave_x_marijuana.png', dpi=300, bbox_inches='tight')
    print("Saved: figures/shap_wave_x_marijuana.png")
    plt.close()

# Plot 2: Marijuana x Wave
if 'marijuana12' in X_sample.columns and 'wave' in X_sample.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.dependence_plot(
        "marijuana12",
        shap_values,
        X_sample,
        interaction_index="wave",
        show=False,
        ax=ax
    )
    plt.title("SHAP Interaction: Marijuana Use x Survey Wave", fontsize=14)
    plt.xlabel("Marijuana Use Level", fontsize=12)
    plt.ylabel("SHAP Value for Marijuana\n(Impact on Vaping Prediction)", fontsize=12)
    plt.tight_layout()
    plt.savefig('figures/shap_marijuana_x_wave.png', dpi=300, bbox_inches='tight')
    print("Saved: figures/shap_marijuana_x_wave.png")
    plt.close()

# Plot 3: Interaction heatmap for top features
print("\nComputing SHAP interaction values (this may take a few minutes)...")
shap_interaction = explainer.shap_interaction_values(X_sample.sample(min(1000, len(X_sample)), random_state=42))
shap_interaction_mean = np.abs(shap_interaction).mean(axis=0)

# Get top features
top_features_idx = np.argsort(np.abs(shap_values).mean(axis=0))[-15:]
interaction_subset = shap_interaction_mean[top_features_idx][:, top_features_idx]

fig, ax = plt.subplots(figsize=(12, 10))
top_feature_names = [features[i] for i in top_features_idx]

sns.heatmap(interaction_subset,
            xticklabels=top_feature_names,
            yticklabels=top_feature_names,
            annot=True, fmt='.3f', cmap='RdYlBu_r', center=0,
            cbar_kws={'label': 'Mean |SHAP Interaction|'},
            ax=ax)

plt.title('SHAP Interaction Heatmap: Top 15 Features', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('figures/shap_interaction_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: figures/shap_interaction_heatmap.png")
plt.close()

print("\n" + "="*70)
print(" SHAP DEPENDENCE ANALYSIS COMPLETE")
print("="*70)
print("\nGenerated 3 plots:")
print("1. shap_wave_x_marijuana.png - Shows how wave effect varies by marijuana use")
print("2. shap_marijuana_x_wave.png - Shows how marijuana effect varies over time")
print("3. shap_interaction_heatmap.png - Shows all pairwise interactions for top 15 features")
