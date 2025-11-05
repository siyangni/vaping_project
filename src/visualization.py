"""
Visualization Module

Functions for creating publication-quality figures and charts.

Key functions:
    - plot_confusion_matrix: Confusion matrix heatmap
    - plot_roc_curves: ROC curves for multiple models
    - plot_model_comparison: Bar chart comparing model performance
    - plot_feature_distributions: Distribution plots for features
    - plot_correlation_heatmap: Feature correlation matrix

Author: Siyang Ni
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Set publication-quality style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    normalize: bool = False,
    title: str = 'Confusion Matrix',
    cmap: str = 'Blues',
    output_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix as heatmap.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        normalize: Whether to normalize by row (True) or show counts (False)
        title: Plot title
        cmap: Color map
        output_path: Path to save figure
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap=cmap,
        xticklabels=labels if labels else 'auto',
        yticklabels=labels if labels else 'auto',
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )

    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {output_path}")

    plt.show()


def plot_roc_curves(
    models_dict: Dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_path: Optional[str] = None
) -> None:
    """
    Plot ROC curves for multiple models.

    Args:
        models_dict: Dictionary of {model_name: trained_model}
        X_test: Test features
        y_test: Test labels
        output_path: Path to save figure
    """
    plt.figure(figsize=(10, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(models_dict)))

    for (model_name, model), color in zip(models_dict.items(), colors):
        # Get predictions
        y_pred_proba = model.predict_proba(X_test)

        # For binary classification
        if y_pred_proba.shape[1] == 2:
            y_score = y_pred_proba[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)

            plt.plot(
                fpr,
                tpr,
                color=color,
                lw=2,
                label=f'{model_name} (AUC = {roc_auc:.3f})'
            )

    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to: {output_path}")

    plt.show()


def plot_model_comparison(
    results_df: pd.DataFrame,
    metrics: List[str] = ['roc_auc', 'f1_score', 'accuracy'],
    output_path: Optional[str] = None
) -> None:
    """
    Create bar chart comparing model performance across metrics.

    Args:
        results_df: DataFrame with model results
        metrics: List of metrics to plot
        output_path: Path to save figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))

    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        data = results_df.sort_values(metric, ascending=False)

        ax.barh(
            data['model_name'],
            data[metric],
            color='steelblue',
            alpha=0.8
        )

        ax.set_xlabel(metric.upper().replace('_', ' '), fontsize=12)
        ax.set_ylabel('Model', fontsize=12)
        ax.set_title(f'{metric.upper().replace("_", " ")}', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.set_xlim([0, 1])

        # Add value labels
        for i, v in enumerate(data[metric]):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison saved to: {output_path}")

    plt.show()


def plot_feature_distributions(
    df: pd.DataFrame,
    features: List[str],
    target_col: Optional[str] = None,
    n_cols: int = 3,
    output_path: Optional[str] = None
) -> None:
    """
    Plot distributions of multiple features.

    Args:
        df: DataFrame with features
        features: List of feature names to plot
        target_col: Target column for color coding
        n_cols: Number of columns in subplot grid
        output_path: Path to save figure
    """
    n_features = len(features)
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    for idx, feature in enumerate(features):
        ax = axes[idx]

        if target_col:
            for label in df[target_col].unique():
                subset = df[df[target_col] == label]
                ax.hist(
                    subset[feature].dropna(),
                    alpha=0.6,
                    label=f'{target_col}={label}',
                    bins=30
                )
            ax.legend()
        else:
            ax.hist(df[feature].dropna(), bins=30, color='steelblue', alpha=0.7)

        ax.set_xlabel(feature, fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'Distribution: {feature}', fontsize=12)

    # Hide empty subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Feature distributions saved to: {output_path}")

    plt.show()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    method: str = 'pearson',
    threshold: Optional[float] = None,
    output_path: Optional[str] = None
) -> None:
    """
    Plot correlation heatmap for features.

    Args:
        df: DataFrame with features
        features: List of features (if None, use all numeric)
        method: Correlation method ('pearson', 'spearman', 'kendall')
        threshold: Only show correlations above threshold
        output_path: Path to save figure
    """
    # Select features
    if features:
        data = df[features]
    else:
        data = df.select_dtypes(include=[np.number])

    # Compute correlation
    corr = data.corr(method=method)

    # Apply threshold
    if threshold:
        mask = np.abs(corr) < threshold
        corr = corr.mask(mask)

    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=False,
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': f'{method.capitalize()} Correlation'}
    )

    plt.title(f'Feature Correlation Matrix ({method.capitalize()})', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Correlation heatmap saved to: {output_path}")

    plt.show()


def plot_learning_curve(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    train_sizes: np.ndarray,
    title: str = 'Learning Curve',
    output_path: Optional[str] = None
) -> None:
    """
    Plot learning curve showing training and validation scores.

    Args:
        train_scores: Training scores for different sample sizes
        test_scores: Validation scores for different sample sizes
        train_sizes: Sample sizes
        title: Plot title
        output_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))

    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', color='g', label='Validation score')

    # Shaded std regions
    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.1,
        color='r'
    )
    plt.fill_between(
        train_sizes,
        test_mean - test_std,
        test_mean + test_std,
        alpha=0.1,
        color='g'
    )

    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Learning curve saved to: {output_path}")

    plt.show()


def save_figure_with_metadata(
    fig: plt.Figure,
    output_path: str,
    title: str,
    description: str
) -> None:
    """
    Save figure with embedded metadata.

    Args:
        fig: Matplotlib figure
        output_path: Output filepath
        title: Figure title
        description: Figure description
    """
    metadata = {
        'Title': title,
        'Description': description,
        'Software': 'Python/Matplotlib',
        'Creator': 'Vaping Project Analysis'
    }

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches='tight',
        metadata=metadata
    )

    print(f"Figure saved with metadata to: {output_path}")


if __name__ == "__main__":
    print("Visualization module loaded successfully.")
