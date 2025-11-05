"""
Model Interpretability Module

Functions for interpreting ML models using SHAP values, feature importance,
and partial dependence plots.

Key functions:
    - compute_shap_values: Calculate SHAP values for model
    - plot_shap_summary: Generate SHAP summary plots
    - compute_feature_importance: Extract feature importance scores
    - compute_permutation_importance: Permutation-based importance
    - plot_partial_dependence: Generate PDP plots
    - analyze_feature_interactions: Detect feature interactions

Author: Siyang Ni
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

import shap
from sklearn.inspection import permutation_importance, partial_dependence
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def compute_shap_values(
    model: object,
    X: np.ndarray,
    X_background: Optional[np.ndarray] = None,
    n_background: int = 100,
    model_type: str = 'tree'
) -> Tuple[shap.Explanation, shap.Explainer]:
    """
    Compute SHAP values for model predictions.

    Args:
        model: Trained model
        X: Data to explain
        X_background: Background dataset for SHAP (if None, sample from X)
        n_background: Number of background samples
        model_type: Type of explainer ('tree', 'linear', 'kernel')

    Returns:
        Tuple of (shap_values, explainer)
    """
    print(f"Computing SHAP values using {model_type} explainer...")

    # Select background data
    if X_background is None:
        n_background = min(n_background, X.shape[0])
        background_idx = np.random.choice(X.shape[0], n_background, replace=False)
        X_background = X[background_idx]

    # Create explainer based on model type
    if model_type == 'tree':
        explainer = shap.TreeExplainer(model, X_background)
    elif model_type == 'linear':
        explainer = shap.LinearExplainer(model, X_background)
    elif model_type == 'kernel':
        explainer = shap.KernelExplainer(model.predict_proba, X_background)
    else:
        # Auto-detect
        explainer = shap.Explainer(model, X_background)

    # Compute SHAP values
    shap_values = explainer(X)

    print(f"SHAP values computed. Shape: {shap_values.values.shape}")

    return shap_values, explainer


def plot_shap_summary(
    shap_values: shap.Explanation,
    feature_names: Optional[List[str]] = None,
    max_display: int = 20,
    plot_type: str = 'dot',
    output_path: Optional[str] = None
) -> None:
    """
    Generate SHAP summary plot.

    Args:
        shap_values: SHAP values from compute_shap_values()
        feature_names: List of feature names
        max_display: Maximum features to display
        plot_type: 'dot', 'bar', or 'violin'
        output_path: Path to save figure
    """
    plt.figure(figsize=(12, 8))

    if plot_type == 'bar':
        shap.summary_plot(
            shap_values,
            plot_type='bar',
            max_display=max_display,
            feature_names=feature_names,
            show=False
        )
    else:
        shap.summary_plot(
            shap_values,
            plot_type=plot_type,
            max_display=max_display,
            feature_names=feature_names,
            show=False
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"SHAP summary plot saved to: {output_path}")

    plt.show()


def aggregate_shap_importance(
    shap_values: shap.Explanation,
    feature_names: List[str],
    top_n: int = 20
) -> pd.DataFrame:
    """
    Aggregate SHAP values to get feature importance.

    Args:
        shap_values: SHAP values
        feature_names: Feature names
        top_n: Number of top features to return

    Returns:
        DataFrame with feature importance scores
    """
    # Mean absolute SHAP value for each feature
    shap_importance = np.abs(shap_values.values).mean(axis=0)

    # Handle multi-output (multi-class)
    if len(shap_importance.shape) > 1:
        shap_importance = shap_importance.mean(axis=1)

    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': shap_importance
    })

    importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)

    return importance_df


def plot_shap_feature_importance(
    importance_df: pd.DataFrame,
    output_path: Optional[str] = None
) -> None:
    """
    Plot aggregated SHAP feature importance.

    Args:
        importance_df: DataFrame from aggregate_shap_importance()
        output_path: Path to save figure
    """
    plt.figure(figsize=(10, 8))

    plt.barh(
        importance_df['feature'],
        importance_df['importance'],
        color='steelblue'
    )

    plt.xlabel('Mean |SHAP value| (Average impact on model output)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Feature Importance (SHAP)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to: {output_path}")

    plt.show()


def compute_permutation_importance(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 10,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compute permutation-based feature importance.

    Args:
        model: Trained model
        X: Features
        y: Labels
        feature_names: Feature names
        n_repeats: Number of permutation repeats
        random_state: Random seed

    Returns:
        DataFrame with permutation importance scores
    """
    print("Computing permutation importance...")

    perm_importance = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    })

    importance_df = importance_df.sort_values('importance_mean', ascending=False)

    print("Permutation importance computed.")

    return importance_df


def plot_partial_dependence(
    model: object,
    X: np.ndarray,
    feature_names: List[str],
    features_to_plot: List[Union[int, str]],
    output_path: Optional[str] = None
) -> None:
    """
    Generate partial dependence plots.

    Args:
        model: Trained model
        X: Features
        feature_names: Feature names
        features_to_plot: List of feature indices or names to plot
        output_path: Path to save figure
    """
    from sklearn.inspection import PartialDependenceDisplay

    # Convert feature names to indices if needed
    feature_indices = []
    for feat in features_to_plot:
        if isinstance(feat, str):
            feature_indices.append(feature_names.index(feat))
        else:
            feature_indices.append(feat)

    fig, ax = plt.subplots(figsize=(14, 4))

    display = PartialDependenceDisplay.from_estimator(
        model,
        X,
        features=feature_indices,
        feature_names=feature_names,
        ax=ax,
        n_cols=len(feature_indices)
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Partial dependence plot saved to: {output_path}")

    plt.show()


def analyze_feature_interactions(
    shap_values: shap.Explanation,
    feature_names: List[str],
    top_n: int = 10
) -> pd.DataFrame:
    """
    Identify top feature interactions using SHAP interaction values.

    Args:
        shap_values: SHAP values
        feature_names: Feature names
        top_n: Number of top interactions to return

    Returns:
        DataFrame with interaction scores
    """
    print("Analyzing feature interactions...")

    # This requires SHAP interaction values (only for tree models)
    # For now, return placeholder
    print("Note: Interaction analysis requires TreeExplainer with interactions enabled.")

    return pd.DataFrame()


def create_shap_waterfall_plot(
    shap_values: shap.Explanation,
    instance_idx: int = 0,
    output_path: Optional[str] = None
) -> None:
    """
    Create SHAP waterfall plot for a single prediction.

    Args:
        shap_values: SHAP values
        instance_idx: Index of instance to explain
        output_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))

    shap.plots.waterfall(shap_values[instance_idx], show=False)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Waterfall plot saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    print("Interpretability module loaded successfully.")
