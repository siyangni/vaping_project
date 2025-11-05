"""
Machine Learning Modeling Module

Functions for training, evaluating, and comparing ML classifiers.

Key functions:
    - get_classifier_configs: Define model hyperparameter grids
    - train_single_model: Train individual classifier with CV
    - train_all_models: Train suite of 6 classifiers
    - evaluate_model: Compute performance metrics
    - compare_models: Generate comparison table

Classifiers implemented:
    1. Logistic Regression (Lasso)
    2. Random Forest
    3. Gradient Boosting
    4. Histogram-based Gradient Boosting
    5. XGBoost
    6. CatBoost

Author: Siyang Ni
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
import xgboost as xgb
from catboost import CatBoostClassifier
import joblib


# Constants
RANDOM_STATE = 42
CV_FOLDS = 5


def get_classifier_configs() -> Dict:
    """
    Get classifier configurations with hyperparameter grids.

    Returns:
        Dictionary mapping model names to (estimator, param_grid) tuples
    """
    configs = {
        'LogisticRegression': {
            'estimator': LogisticRegression(
                penalty='l1',
                solver='liblinear',
                max_iter=1000,
                random_state=RANDOM_STATE
            ),
            'param_grid': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100]
            },
            'search_type': 'grid'
        },

        'RandomForest': {
            'estimator': RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            'param_grid': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            },
            'search_type': 'random',
            'n_iter': 50
        },

        'GradientBoosting': {
            'estimator': GradientBoostingClassifier(
                random_state=RANDOM_STATE
            ),
            'param_grid': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'subsample': [0.8, 1.0]
            },
            'search_type': 'random',
            'n_iter': 50
        },

        'HistGradientBoosting': {
            'estimator': HistGradientBoostingClassifier(
                random_state=RANDOM_STATE
            ),
            'param_grid': {
                'max_iter': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [5, 10, 15, None],
                'min_samples_leaf': [20, 50, 100],
                'l2_regularization': [0.0, 0.1, 1.0]
            },
            'search_type': 'random',
            'n_iter': 30
        },

        'XGBoost': {
            'estimator': xgb.XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric='logloss',
                use_label_encoder=False
            ),
            'param_grid': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'gamma': [0, 0.1, 0.2]
            },
            'search_type': 'random',
            'n_iter': 50
        },

        'CatBoost': {
            'estimator': CatBoostClassifier(
                random_state=RANDOM_STATE,
                verbose=0
            ),
            'param_grid': {
                'iterations': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'depth': [4, 6, 8],
                'l2_leaf_reg': [1, 3, 5, 7],
                'border_count': [32, 64, 128]
            },
            'search_type': 'random',
            'n_iter': 30
        }
    }

    return configs


def train_single_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = CV_FOLDS,
    scoring: str = 'roc_auc',
    verbose: int = 1
) -> Tuple[object, Dict]:
    """
    Train a single classifier with hyperparameter tuning.

    Args:
        model_name: Name of model from get_classifier_configs()
        X_train: Training features
        y_train: Training labels
        cv_folds: Number of CV folds
        scoring: Scoring metric for CV
        verbose: Verbosity level

    Returns:
        Tuple of (best_model, training_info)
    """
    configs = get_classifier_configs()

    if model_name not in configs:
        raise ValueError(f"Model {model_name} not found. Available: {list(configs.keys())}")

    config = configs[model_name]
    estimator = config['estimator']
    param_grid = config['param_grid']
    search_type = config['search_type']

    print(f"\n{'='*60}")
    print(f"Training {model_name}...")
    print(f"{'='*60}")

    # Choose search strategy
    if search_type == 'grid':
        searcher = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=verbose,
            return_train_score=True
        )
    else:  # random search
        searcher = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            n_iter=config.get('n_iter', 30),
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=verbose,
            random_state=RANDOM_STATE,
            return_train_score=True
        )

    # Train
    searcher.fit(X_train, y_train)

    # Extract results
    training_info = {
        'best_params': searcher.best_params_,
        'best_cv_score': searcher.best_score_,
        'mean_train_score': searcher.cv_results_['mean_train_score'][searcher.best_index_],
        'std_cv_score': searcher.cv_results_['std_test_score'][searcher.best_index_],
        'model_name': model_name
    }

    print(f"\nBest parameters: {searcher.best_params_}")
    print(f"Best CV {scoring}: {searcher.best_score_:.4f} (+/- {training_info['std_cv_score']:.4f})")

    return searcher.best_estimator_, training_info


def evaluate_model(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    label_mapping: Optional[Dict] = None
) -> Dict:
    """
    Evaluate trained model on test set.

    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test labels
        model_name: Model name
        label_mapping: Dictionary mapping class indices to labels

    Returns:
        Dictionary of evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Calculate metrics
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'roc_auc': roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    }

    # Print results
    print(f"\n{'='*60}")
    print(f"{model_name} - Test Set Performance")
    print(f"{'='*60}")
    for metric, value in metrics.items():
        if metric != 'model_name':
            print(f"{metric.capitalize():20s}: {value:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_mapping.values() if label_mapping else None))

    return metrics


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_mapping: Optional[Dict] = None,
    models_to_train: Optional[List[str]] = None
) -> Tuple[Dict, pd.DataFrame]:
    """
    Train and evaluate all classifiers.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        label_mapping: Class label mapping
        models_to_train: List of model names (default: all)

    Returns:
        Tuple of (trained_models_dict, results_dataframe)
    """
    configs = get_classifier_configs()

    if models_to_train is None:
        models_to_train = list(configs.keys())

    trained_models = {}
    all_results = []

    for model_name in models_to_train:
        # Train model
        model, train_info = train_single_model(model_name, X_train, y_train)

        # Evaluate model
        test_metrics = evaluate_model(model, X_test, y_test, model_name, label_mapping)

        # Combine results
        results = {**train_info, **test_metrics}
        all_results.append(results)

        # Store model
        trained_models[model_name] = model

    # Create comparison DataFrame
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('roc_auc', ascending=False)

    print(f"\n{'='*80}")
    print("MODEL COMPARISON (sorted by ROC-AUC)")
    print(f"{'='*80}")
    print(results_df[['model_name', 'roc_auc', 'f1_score', 'accuracy', 'precision', 'recall']].to_string(index=False))

    return trained_models, results_df


def save_model(model: object, filepath: str) -> None:
    """
    Save trained model to disk.

    Args:
        model: Trained model
        filepath: Output filepath (.joblib)
    """
    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")


def load_model(filepath: str) -> object:
    """
    Load trained model from disk.

    Args:
        filepath: Path to model file

    Returns:
        Loaded model
    """
    model = joblib.load(filepath)
    print(f"Model loaded from: {filepath}")
    return model


if __name__ == "__main__":
    print("Modeling module loaded successfully.")
    print(f"Available models: {list(get_classifier_configs().keys())}")
