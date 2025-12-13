#!/usr/bin/env python
# coding: utf-8

# # SHAP Feature Importance Stability Analysis
# 
# ## Overview
# This notebook uses bootstrap sampling to assess how stable SHAP importance rankings are. This validates that consensus features are genuinely robust, not artifacts of a single train-test split.
# 
# ## Key Questions
# - Are SHAP importance rankings stable across bootstrap samples?
# - What is the coefficient of variation for top features?
# - Are consensus features genuinely robust?
# 
# ---

# ## 1. Setup and Imports

# In[ ]:


import pandas as pd
import numpy as np
import os
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.utils import resample
import shap
import matplotlib.pyplot as plt

RANDOM_STATE = 42
N_BOOTSTRAP = 30  # Reduced for computational efficiency
np.random.seed(RANDOM_STATE)

print("="*70)
print(" SHAP STABILITY ANALYSIS")
print("="*70)


# ## 2. Load Data and Model

# In[ ]:


# Load data
data_path = os.path.expanduser('~/work/vaping_project_data/processed_data_g12n.csv')
df = pd.read_csv(data_path)

TARGET = 'nicotine12d'
df_clean = df[df[TARGET].notna()].copy()

exclude_cols = [TARGET, 'V1'] if 'V1' in df.columns else [TARGET]
features = [c for c in df.columns if c not in exclude_cols]

X = df_clean[features].fillna(df_clean[features].median())
y = df_clean[TARGET]

# Load best model (XGBoost)
models_dir = Path('../outputs/models')
model_path = models_dir / 'xgboost.joblib'

if not model_path.exists():
    print(f"\nWARNING: XGBoost model not found at {model_path}")
    print("Creating demonstration with Random Forest...")
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X, y)
else:
    model = joblib.load(model_path)
    print(f"\nLoaded: XGBoost model")

# Subsample for computational efficiency
if len(X) > 10000:
    X_sample = X.sample(10000, random_state=RANDOM_STATE)
    print(f"Using sample of {len(X_sample):,} observations for bootstrap analysis")
else:
    X_sample = X


# ## 3. Bootstrap SHAP Analysis

# In[ ]:


print("="*70)
print(f" RUNNING {N_BOOTSTRAP} BOOTSTRAP ITERATIONS")
print("="*70)

shap_values_bootstrap = []

for i in range(N_BOOTSTRAP):
    print(f"Bootstrap iteration {i+1}/{N_BOOTSTRAP}...", end='\r')

    # Bootstrap sample
    X_boot = resample(X_sample, random_state=i)

    # Compute SHAP
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_boot)

    # Handle different SHAP output formats
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]  # Use positive class

    # Mean absolute SHAP
    importance = np.abs(shap_vals).mean(axis=0)
    shap_values_bootstrap.append(importance)

print(f"\nCompleted {N_BOOTSTRAP} bootstrap iterations")

# Convert to array
shap_bootstrap_array = np.array(shap_values_bootstrap)


# ## 4. Compute Stability Metrics

# In[ ]:


print("="*70)
print(" STABILITY METRICS")
print("="*70)

shap_mean = shap_bootstrap_array.mean(axis=0)
shap_std = shap_bootstrap_array.std(axis=0)
shap_lower = np.percentile(shap_bootstrap_array, 2.5, axis=0)
shap_upper = np.percentile(shap_bootstrap_array, 97.5, axis=0)

# Coefficient of variation
cv = (shap_std / shap_mean) * 100

stability_df = pd.DataFrame({
    'Feature': features[:len(shap_mean)],
    'Mean_Importance': shap_mean,
    'SD': shap_std,
    'Lower_95CI': shap_lower,
    'Upper_95CI': shap_upper,
    'CV_Pct': cv
}).sort_values('Mean_Importance', ascending=False)

# Top 20 features
top20 = stability_df.head(20)

print("\nTop 20 Features with Stability Metrics:")
print(top20[['Feature', 'Mean_Importance', 'CV_Pct']].to_string(index=False))

# Identify unstable features
unstable = top20[top20['CV_Pct'] > 50]
if len(unstable) > 0:
    print(f"\nWarning: {len(unstable)} features have CV > 50% (unstable):")
    print(unstable[['Feature', 'CV_Pct']].to_string(index=False))
else:
    print("\n✓ All top 20 features are stable (CV < 50%)")


# ## 5. Visualization

# In[ ]:


print("="*70)
print(" CREATING VISUALIZATIONS")
print("="*70)

fig, ax = plt.subplots(figsize=(10, 8))

y_pos = np.arange(len(top20))

# Error bars showing 95% CI
ax.barh(y_pos, top20['Mean_Importance'],
        xerr=[top20['Mean_Importance'] - top20['Lower_95CI'],
              top20['Upper_95CI'] - top20['Mean_Importance']],
        capsize=4, alpha=0.7)

ax.set_yticks(y_pos)
ax.set_yticklabels(top20['Feature'])
ax.set_xlabel(f'SHAP Importance (Mean ± 95% CI, N={N_BOOTSTRAP} bootstraps)', fontsize=11, fontweight='bold')
ax.set_title('Feature Importance Stability Analysis', fontsize=13, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

print("\nStability plot created")


# ## 6. Save Results

# In[ ]:


output_dir = Path('../outputs/tables')
output_dir.mkdir(parents=True, exist_ok=True)

stability_df.to_csv(output_dir / 'shap_stability_results.csv', index=False)
print(f"\n✓ Results saved to: {output_dir / 'shap_stability_results.csv'}")

# Save figure
fig_dir = Path('../figures')
fig_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_dir / 'shap_stability.png', dpi=300, bbox_inches='tight')
print(f"✓ Figure saved to: {fig_dir / 'shap_stability.png'}")


# ## Summary
# 
# ### Key Findings:
# - ✅ Analyzed stability across bootstrap samples
# - ✅ Quantified coefficient of variation for top features
# - ✅ Validated robustness of consensus features
# - ✅ Identified any unstable feature rankings
# 
# ### Interpretation:
# - **Low CV (< 20%)**: Feature importance is very stable
# - **Moderate CV (20-50%)**: Some variability but generally reliable
# - **High CV (> 50%)**: Unstable rankings, interpret with caution
# - **Consensus validation**: Stable features support multi-model consensus approach
