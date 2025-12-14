# %%
# ============================
#  Title:  Model Analysis & Visualization (Interactive)
#  Author: Siyang Ni
#  Notes:  This script loads pre-trained models and performs interactive analysis
#          including ROC curves, feature importance plots, SHAP analysis, and
#          partial dependence plots. Designed for Jupyter/IPython environments.
#          
#          PREREQUISITE: Run 03a_model_training.py first to train and save models.
# ============================

# %%
# ================
# 1. IMPORTS
# ================

import os
import logging
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
import shap
import math

# Import shared configuration
from model_config import (
    MODELS_DIR, RANDOM_STATE,
    get_model_path, get_preprocessed_data_path, get_shap_output_dir,
    TARGET_COL
)

# %%
# ================
# 2. CONFIGURATION
# ================

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

logging.info(f"Loading models from: {MODELS_DIR}")

# %%
# ================
# 3. LOAD PREPROCESSED DATA
# ================

logging.info("\n" + "="*70)
logging.info("LOADING PREPROCESSED DATA")
logging.info("="*70)

data_path = get_preprocessed_data_path()
if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"Preprocessed data not found at {data_path}.\n"
        "Please run 03a_model_training.py first to generate preprocessed data and models."
    )

loaded_data = joblib.load(data_path)
X_train_with_indicators = loaded_data['X_train_with_indicators']
X_test_with_indicators = loaded_data['X_test_with_indicators']
y_train = loaded_data['y_train']
y_test = loaded_data['y_test']
categorical_features = loaded_data['categorical_features']
numeric_features = loaded_data.get('numeric_features', [])
preprocessor = loaded_data['preprocessor']

logging.info(f"Training set shape: {X_train_with_indicators.shape}")
logging.info(f"Test set shape: {X_test_with_indicators.shape}")
logging.info(f"Numeric features: {len(numeric_features)}")
logging.info(f"Categorical features: {len(categorical_features)}")

# %%
# ================
# 4. HELPER FUNCTIONS FOR VISUALIZATION
# ================

def load_model_safe(model_name: str):
    """Load a model with error handling."""
    model_path = get_model_path(model_name)
    if not os.path.exists(model_path):
        logging.warning(f"Model '{model_name}' not found at {model_path}")
        return None
    try:
        model = joblib.load(model_path)
        logging.info(f"✓ Loaded {model_name} from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading {model_name}: {e}")
        return None


def plot_roc_curve(model, X_test, y_test, model_name: str):
    """Plot ROC curve for a given model."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{model_name} ROC Curve on Test Data', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return auc_score


def aggregate_feature_importance(importances: np.ndarray, encoded_feature_names: list) -> pd.DataFrame:
    """
    Aggregates feature importance of one-hot-encoded features back to original feature names.
    """
    def _base_feature(name: str) -> str:
        if '__' in name:
            name = name.split('__', 1)[1]
        return name.rsplit('_', 1)[0] if '_' in name else name

    original_features = list({_base_feature(feat) for feat in encoded_feature_names})
    original_feature_importance = {feature: 0 for feature in original_features}
    for i, encoded_feature in enumerate(encoded_feature_names):
        base_feature = _base_feature(encoded_feature)
        original_feature_importance[base_feature] += importances[i]
    
    importance_df = pd.DataFrame(
        list(original_feature_importance.items()), 
        columns=['Feature', 'Importance']
    )
    importance_df = importance_df.sort_values('Importance', ascending=False)
    return importance_df


def plot_aggregated_feature_importance(
    importance_df: pd.DataFrame, 
    top_n: int = 20, 
    title: str = "Aggregated Feature Importance"
):
    """
    Plots aggregated feature importances after grouping by base feature.
    """
    top_n_df = importance_df.head(top_n).sort_values(by='Importance', ascending=True)
    plt.figure(figsize=(10, 8))
    plt.barh(y=top_n_df['Feature'], width=top_n_df['Importance'], color='steelblue')
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_tree_feature_importance(model, model_name: str, top_n: int = 20):
    """
    Plot feature importance for tree-based models.
    Aggregates one-hot encoded features back to original names.
    """
    if hasattr(model, 'named_steps'):
        # It's a pipeline
        if 'classifier' in model.named_steps:
            clf = model.named_steps['classifier']
        else:
            logging.warning(f"No 'classifier' step found in {model_name} pipeline")
            return
        
        if 'preprocessor' in model.named_steps:
            prep = model.named_steps['preprocessor']
            if hasattr(prep, 'get_feature_names_out'):
                feature_names = prep.get_feature_names_out()
            else:
                feature_names = [f"Feature_{i}" for i in range(len(clf.feature_importances_))]
        else:
            feature_names = [f"Feature_{i}" for i in range(len(clf.feature_importances_))]
    else:
        clf = model
        feature_names = [f"Feature_{i}" for i in range(len(clf.feature_importances_))]
    
    if not hasattr(clf, 'feature_importances_'):
        logging.warning(f"{model_name} does not have feature_importances_ attribute")
        return
    
    importances = clf.feature_importances_
    
    # Aggregate by base feature name
    importance_df = aggregate_feature_importance(importances, feature_names)
    plot_aggregated_feature_importance(importance_df, top_n=top_n, title=f"{model_name} - Top {top_n} Features")


def plot_coefficient_importance(model, model_name: str, top_n: int = 20):
    """
    Plot coefficient-based importance for linear models (Lasso).
    Aggregates one-hot encoded features.
    """
    if not hasattr(model, 'named_steps'):
        logging.warning(f"{model_name} is not a pipeline")
        return
    
    if 'classifier' not in model.named_steps:
        logging.warning(f"No classifier in {model_name}")
        return
    
    lr = model.named_steps['classifier']
    if not hasattr(lr, 'coef_'):
        logging.warning(f"{model_name} classifier has no coef_ attribute")
        return
    
    coefficients = lr.coef_[0] if lr.coef_.ndim > 1 else lr.coef_
    
    # Get feature names from preprocessor
    if 'preprocessor' in model.named_steps:
        prep = model.named_steps['preprocessor']
        if hasattr(prep, 'get_feature_names_out'):
            feature_names = prep.get_feature_names_out()
        else:
            feature_names = [f"Feature_{i}" for i in range(len(coefficients))]
    elif 'poly' in model.named_steps:
        # For polynomial features, create placeholder names
        feature_names = [f"Poly_Feature_{i}" for i in range(len(coefficients))]
    else:
        feature_names = [f"Feature_{i}" for i in range(len(coefficients))]
    
    # Use absolute coefficients for importance
    abs_coefs = np.abs(coefficients)
    
    #  Aggregate by base feature
    if 'poly' not in model.named_steps:
        # Only aggregate if not polynomial (poly features are too complex to aggregate meaningfully)
        importance_df = aggregate_feature_importance(abs_coefs, feature_names)
        plot_aggregated_feature_importance(
            importance_df, top_n=top_n, 
            title=f"{model_name} - Top {top_n} Features (|Coefficient|)"
        )
    else:
        # For polynomial models, show top raw coefficients
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': abs_coefs
        }).sort_values('Coefficient', ascending=False).head(top_n)
        
        coef_df_sorted = coef_df.sort_values('Coefficient', ascending=True)
        plt.figure(figsize=(10, 8))
        plt.barh(y=coef_df_sorted['Feature'], width=coef_df_sorted['Coefficient'], color='coral')
        plt.xlabel('|Coefficient|', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(f"{model_name} - Top {top_n} Features (|Coefficient|)", fontsize=14)
        plt.tight_layout()
        plt.show()


def plot_partial_dependence(model, X_data, feature_names: list = None, top_n: int = 10, model_name: str = "Model"):
    """
    Plot partial dependence for top features.
    """
    if feature_names is None:
        # Use all features excluding missing indicators and respondent_sex
        feature_names = [
            feat for feat in X_data.columns
            if feat.lower() != 'respondent_sex' and not feat.lower().startswith("missing_")
        ][:top_n]
    
    n_features = len(feature_names)
    n_cols = 3
    n_rows = math.ceil(n_features / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), squeeze=False)
    
    for ax, feature in zip(axes.flat, feature_names):
        try:
            PartialDependenceDisplay.from_estimator(model, X_data, [feature], ax=ax)
            ax.grid(False)
            ax.set_title(f'PDP: {feature}')
        except Exception as e:
            logging.warning(f"Could not plot PDP for {feature}: {e}")
            ax.set_visible(False)
    
    # Hide unused subplots
    for ax in axes.flat[n_features:]:
        ax.set_visible(False)
    
    fig.suptitle(f"{model_name} - Partial Dependence Plots", fontsize=16, y=1.0)
    plt.tight_layout()
    plt.show()


def compute_shap_for_linear_model(model, X_data, model_name: str, n_samples: int = 2000):
    """
    Compute and visualize SHAP values for linear models.
    """
    logging.info(f"Computing SHAP values for {model_name}...")
    
    # Transform data to match classifier's feature space
    if 'poly' in model.named_steps:
        # For polynomial models, transform through preprocessor and poly
        X_pre = model.named_steps['preprocessor'].transform(X_data)
        X_transformed = model.named_steps['poly'].transform(X_pre)
    elif 'preprocessor' in model.named_steps:
        X_transformed = model.named_steps['preprocessor'].transform(X_data)
    else:
        X_transformed = X_data
    
    # Sample data for performance
    n_samples = min(n_samples, X_transformed.shape[0])
    sample_idx = np.random.choice(X_transformed.shape[0], n_samples, replace=False)
    X_sample = X_transformed[sample_idx]
    
    # Background data (smaller sample)
    bg_n = min(500, X_transformed.shape[0])
    bg_idx = np.random.choice(X_transformed.shape[0], bg_n, replace=False)
    X_bg = X_transformed[bg_idx]
    
    # Get classifier
    if 'classifier' in model.named_steps:
        clf = model.named_steps['classifier']
    else:
        clf = model
    
    # Compute SHAP
    try:
        explainer = shap.LinearExplainer(clf, X_bg)
        shap_values = explainer.shap_values(X_sample)
        
        # Handle list output (binary classification)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.title(f"{model_name} - SHAP Feature Importance", fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # Beeswarm plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title(f"{model_name} - SHAP Summary (Beeswarm)", fontsize=14)
        plt.tight_layout()
        plt.show()
        
        logging.info(f"SHAP analysis complete for {model_name}")
        
    except Exception as e:
        logging.error(f"SHAP computation failed for {model_name}: {e}")


def compute_shap_for_tree_model(model, X_data, model_name: str, n_samples: int = 500):
    """
    Compute and visualize SHAP values for tree-based models using TreeExplainer.
    """
    logging.info(f"Computing SHAP values for {model_name}...")
    
    # Get classifier from pipeline
    if 'classifier' in model.named_steps:
        clf = model.named_steps['classifier']
        # Transform data through preprocessor
        if 'preprocessor' in model.named_steps:
            X_transformed = model.named_steps['preprocessor'].transform(X_data)
        else:
            X_transformed = X_data
    else:
        clf = model
        X_transformed = X_data
    
    # Sample for performance
    n_samples = min(n_samples, X_transformed.shape[0])
    sample_idx = np.random.choice(X_transformed.shape[0], n_samples, replace=False)
    X_sample = X_transformed[sample_idx]
    
    try:
        # Use TreeExplainer for tree-based models
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_sample)
        
        # Handle different output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        # Bar plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.title(f"{model_name} - SHAP Feature Importance", fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # Beeswarm plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title(f"{model_name} - SHAP Summary (Beeswarm)", fontsize=14)
        plt.tight_layout()
        plt.show()
        
        logging.info(f"SHAP analysis complete for {model_name}")
        
    except Exception as e:
        logging.error(f"SHAP computation failed for {model_name}: {e}")


# %%
# ================
# 5. ELASTIC NET MODEL ANALYSIS
# ================

logging.info("\n" + "="*70)
logging.info("ELASTIC NET MODEL ANALYSIS")
logging.info("="*70)

elasticnet_model = load_model_safe("elasticnet")
if elasticnet_model is not None:
    # ROC Curve
    auc = plot_roc_curve(elasticnet_model, X_test_with_indicators, y_test, "Elastic Net Logistic Regression")
    logging.info(f"Elastic Net Test ROC-AUC: {auc:.4f}")
    
    # Coefficient importance
    plot_coefficient_importance(elasticnet_model, "Elastic Net", top_n=20)
    
    # SHAP analysis
    compute_shap_for_linear_model(elasticnet_model, X_train_with_indicators, "Elastic Net", n_samples=2000)
    
    # Partial dependence (top 10 features by coefficient)
    logging.info("Plotting partial dependence for Elastic Net...")
    top_features = [
        feat for feat in X_train_with_indicators.columns
        if not feat.lower().startswith("missing_") and feat.lower() != "respondent_sex"
    ][:10]
    plot_partial_dependence(elasticnet_model, X_train_with_indicators, top_features, model_name="Elastic Net")

# %%
# ================
# 6. ELASTIC NET 2-WAY INTERACTION MODEL ANALYSIS
# ================

logging.info("\n" + "="*70)
logging.info("ELASTIC NET 2-WAY INTERACTION MODEL ANALYSIS")
logging.info("="*70)

elasticnet_2way_model = load_model_safe("elasticnet_2way")
if elasticnet_2way_model is not None:
    # ROC Curve
    auc = plot_roc_curve(elasticnet_2way_model, X_test_with_indicators, y_test, "Elastic Net (2-way Interactions)")
    logging.info(f"Elastic Net 2-way Test ROC-AUC: {auc:.4f}")
    
    # Coefficient importance (polynomial features - will show many interaction terms)
    plot_coefficient_importance(elasticnet_2way_model, "Elastic Net 2-way", top_n=30)
    
    # SHAP analysis (warning: may be slow due to high dimensionality)
    logging.info("Note: SHAP for 2-way interactions may take several minutes due to high feature count...")
    # Uncomment if you want to run SHAP on 2-way model (can be slow)
    # compute_shap_for_linear_model(elasticnet_2way_model, X_train_with_indicators, "Elastic Net 2-way", n_samples=500)

# %%
# ================
# 7. RANDOM FOREST ANALYSIS
# ================

logging.info("\n" + "="*70)
logging.info("RANDOM FOREST ANALYSIS")
logging.info("="*70)

rf_model = load_model_safe("rf")
if rf_model is not None:
    # ROC Curve
    auc = plot_roc_curve(rf_model, X_test_with_indicators, y_test, "Random Forest")
    logging.info(f"Random Forest Test ROC-AUC: {auc:.4f}")
    
    # Feature importance
    plot_tree_feature_importance(rf_model, "Random Forest", top_n=20)
    
    # SHAP analysis
    compute_shap_for_tree_model(rf_model, X_train_with_indicators, "Random Forest", n_samples=500)
    
    # Partial dependence
    logging.info("Plotting partial dependence for Random Forest...")
    plot_partial_dependence(rf_model, X_train_with_indicators, top_n=10, model_name="Random Forest")

# %%
# ================
# 8. GRADIENT BOOSTING ANALYSIS
# ================

logging.info("\n" + "="*70)
logging.info("GRADIENT BOOSTING ANALYSIS")
logging.info("="*70)

gbt_model = load_model_safe("gbt")
if gbt_model is not None:
    # ROC Curve
    auc = plot_roc_curve(gbt_model, X_test_with_indicators, y_test, "Gradient Boosting")
    logging.info(f"GBT Test ROC-AUC: {auc:.4f}")
    
    # Feature importance
    plot_tree_feature_importance(gbt_model, "Gradient Boosting", top_n=20)
    
    # SHAP analysis
    compute_shap_for_tree_model(gbt_model, X_train_with_indicators, "Gradient Boosting", n_samples=500)
    
    # Partial dependence
    logging.info("Plotting partial dependence for GBT...")
    plot_partial_dependence(gbt_model, X_train_with_indicators, top_n=10, model_name="Gradient Boosting")

# %%
# ================
# 9. HIST GRADIENT BOOSTING ANALYSIS
# ================

logging.info("\n" + "="*70)
logging.info("HIST GRADIENT BOOSTING ANALYSIS")
logging.info("="*70)

hgbt_model = load_model_safe("hgbt")
if hgbt_model is not None:
    # ROC Curve
    auc = plot_roc_curve(hgbt_model, X_test_with_indicators, y_test, "Hist Gradient Boosting")
    logging.info(f"HGBT Test ROC-AUC: {auc:.4f}")
    
    # Feature importance
    plot_tree_feature_importance(hgbt_model, "Hist Gradient Boosting", top_n=20)
    
    # SHAP analysis
    compute_shap_for_tree_model(hgbt_model, X_train_with_indicators, "Hist GBT", n_samples=500)
    
    # Partial dependence
    logging.info("Plotting partial dependence for HGBT...")
    plot_partial_dependence(hgbt_model, X_train_with_indicators, top_n=10, model_name="Hist GBT")

# %%
# ================
# 10. XGBOOST ANALYSIS
# ================

logging.info("\n" + "="*70)
logging.info("XGBOOST ANALYSIS")
logging.info("="*70)

xgb_model = load_model_safe("xgb")
if xgb_model is not None:
    # ROC Curve
    auc = plot_roc_curve(xgb_model, X_test_with_indicators, y_test, "XGBoost")
    logging.info(f"XGBoost Test ROC-AUC: {auc:.4f}")
    
    # Feature importance
    plot_tree_feature_importance(xgb_model, "XGBoost", top_n=20)
    
    # SHAP analysis
    compute_shap_for_tree_model(xgb_model, X_train_with_indicators, "XGBoost", n_samples=500)
    
    # Partial dependence
    logging.info("Plotting partial dependence for XGBoost...")
    plot_partial_dependence(xgb_model, X_train_with_indicators, top_n=10, model_name="XGBoost")

# %%
# ================
# 11. CATBOOST ANALYSIS
# ================

logging.info("\n" + "="*70)
logging.info("CATBOOST ANALYSIS")
logging.info("="*70)

cb_model = load_model_safe("cb")
if cb_model is not None:
    # ROC Curve
    auc = plot_roc_curve(cb_model, X_test_with_indicators, y_test, "CatBoost")
    logging.info(f"CatBoost Test ROC-AUC: {auc:.4f}")
    
    # Feature importance
    plot_tree_feature_importance(cb_model, "CatBoost", top_n=20)
    
    # SHAP analysis
    compute_shap_for_tree_model(cb_model, X_train_with_indicators, "CatBoost", n_samples=500)
    
    # Partial dependence
    logging.info("Plotting partial dependence for CatBoost...")
    plot_partial_dependence(cb_model, X_train_with_indicators, top_n=10, model_name="CatBoost")

# %%
# ================
# 12. MODEL COMPARISON
# ================

logging.info("\n" + "="*70)
logging.info("MODEL COMPARISON")
logging.info("="*70)

# Compare all models on ROC curves in one plot
model_names = ["elasticnet", "elasticnet_2way", "rf", "gbt", "hgbt", "xgb", "cb"]
model_labels = ["Elastic Net", "Elastic Net 2-way", "Random Forest", "GBT", "Hist GBT", "XGBoost", "CatBoost"]
colors = ['blue', 'cyan', 'green', 'orange', 'red', 'purple', 'brown']

plt.figure(figsize=(10, 8))

auc_scores = {}
for name, label, color in zip(model_names, model_labels, colors):
    model = load_model_safe(name)
    if model is not None:
        try:
            y_pred_proba = model.predict_proba(X_test_with_indicators)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)
            auc_scores[label] = auc
            plt.plot(fpr, tpr, label=f'{label} (AUC={auc:.4f})', color=color, linewidth=2)
        except Exception as e:
            logging.warning(f"Could not plot ROC for {label}: {e}")

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Model Comparison: ROC Curves', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print AUC scores summary
print("\n" + "="*70)
print("TEST SET ROC-AUC SCORES SUMMARY")
print("="*70)
for label in sorted(auc_scores.keys(), key=lambda x: auc_scores[x], reverse=True):
    print(f"{label:20s}: {auc_scores[label]:.4f}")
print("="*70)

# %%
print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"All visualizations generated from models in: {MODELS_DIR}")
