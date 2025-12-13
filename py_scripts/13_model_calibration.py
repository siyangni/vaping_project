#!/usr/bin/env python
# coding: utf-8

# # Model Calibration Analysis
# 
# ## Overview
# This notebook evaluates whether predicted probabilities match observed frequencies. High AUC doesn't guarantee good calibration, which is critical for risk stratification and intervention targeting.
# 
# ## Key Questions
# - Are predicted probabilities well-calibrated to observed frequencies?
# - Which models show better calibration?
# - How does calibration affect practical applications?
# 
# ## Methods
# - **Brier score**: Overall calibration metric
# - **Calibration curves**: Visual assessment of agreement
# - **Comparison**: Evaluate multiple models
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

from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*70)
print(" MODEL CALIBRATION ANALYSIS")
print("="*70)


# ## 2. Load Data and Models

# In[ ]:


# Load data
data_path = os.path.expanduser('~/work/vaping_project_data/processed_data_g12n.csv')
df = pd.read_csv(data_path)

TARGET = 'nicotine12d'
df_clean = df[df[TARGET].notna()].copy()

# Prepare features
exclude_cols = [TARGET, 'V1'] if 'V1' in df.columns else [TARGET]
features = [c for c in df.columns if c not in exclude_cols]

X = df_clean[features].fillna(df_clean[features].median())
y = df_clean[TARGET]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"\nTest set: {len(X_test):,} observations")


# In[ ]:


# Load trained models
models_dir = Path('../outputs/models')

model_files = {
    'Random Forest': 'random_forest.joblib',
    'Gradient Boosting': 'gradient_boosting.joblib',
    'XGBoost': 'xgboost.joblib',
    'CatBoost': 'catboost.joblib'
}

models = {}
for name, filename in model_files.items():
    path = models_dir / filename
    if path.exists():
        models[name] = joblib.load(path)
        print(f"Loaded: {name}")
    else:
        print(f"WARNING: {name} not found at {path}")

if len(models) == 0:
    print("\nERROR: No trained models found. Run modeling pipeline first.")
    print("Continuing with demonstration using dummy models...")


# ## 3. Compute Calibration Metrics

# In[ ]:


print("="*70)
print(" COMPUTING CALIBRATION METRICS")
print("="*70)

calibration_results = []

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (name, model) in enumerate(models.items()):
    ax = axes[idx] if idx < 4 else None

    # Get predicted probabilities
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test)

    # Brier score
    brier = brier_score_loss(y_test, y_prob)

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy='quantile')

    calibration_results.append({
        'Model': name,
        'Brier_Score': brier
    })

    print(f"\n{name}:")
    print(f"  Brier Score: {brier:.4f}")

    # Plot if axis available
    if ax is not None:
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax.plot(prob_pred, prob_true, 's-', markersize=8, linewidth=2,
                label=f'{name}\n(Brier={brier:.4f})')
        ax.set_xlabel('Predicted Probability', fontsize=11)
        ax.set_ylabel('Observed Frequency', fontsize=11)
        ax.set_title(f'{name} Calibration', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

plt.tight_layout()
plt.show()

print("\nCalibration plots created")


# ## 4. Summary and Comparison

# In[ ]:


print("="*70)
print(" CALIBRATION RESULTS SUMMARY")
print("="*70)

calibration_df = pd.DataFrame(calibration_results)
calibration_df = calibration_df.sort_values('Brier_Score')
print(calibration_df.to_string(index=False))

print("\nInterpretation:")
print("- Brier score ranges from 0 (perfect) to 1 (worst)")
print("- Lower is better")
print("- Tree models typically show moderate miscalibration")
print("- For risk scoring applications, consider isotonic recalibration")


# ## 5. Save Results

# In[ ]:


# Save results
output_dir = Path('../outputs/tables')
output_dir.mkdir(parents=True, exist_ok=True)

calibration_df.to_csv(output_dir / 'model_calibration_results.csv', index=False)
print(f"\n✓ Results saved to: {output_dir / 'model_calibration_results.csv'}")

# Save figure
fig_dir = Path('../figures')
fig_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_dir / 'model_calibration.png', dpi=300, bbox_inches='tight')
print(f"✓ Figure saved to: {fig_dir / 'model_calibration.png'}")


# ## Summary
# 
# ### Key Findings:
# - ✅ Evaluated calibration of multiple ML models
# - ✅ Brier score quantifies overall calibration quality
# - ✅ Calibration curves reveal systematic over/under-prediction
# - ✅ Tree-based models often need recalibration for probability estimates
# 
# ### Interpretation:
# - **Good calibration**: Predicted probabilities match observed frequencies
# - **Poor calibration**: Systematic bias in probability estimates
# - **Practical implications**: Important for risk stratification and intervention targeting
# - **Recalibration options**: Platt scaling or isotonic regression can improve calibration
