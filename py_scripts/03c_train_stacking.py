# %%
# ============================
#  Title:  Stacking Classifier Training (Interactive)
#  Author: Siyang Ni
#  Notes:  This script loads pre-trained base models and creates an ensemble
#          stacking classifier. Can be run interactively or as a batch script.
# ============================

# %%
# ================
# 1. IMPORTS
# ================

import os
import sys
import time
import logging
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# %%
# Import shared configuration
from model_config import (
    MODELS_DIR, 
    RANDOM_STATE, 
    N_SPLITS_CV,
    get_model_path,
    get_preprocessed_data_path
)


# %%
# ================
# 2. CUSTOM TRANSFORMERS
# ================

from sklearn.base import BaseEstimator, TransformerMixin

class DenseTransformer(BaseEstimator, TransformerMixin):
    """
    Converts sparse matrices to dense arrays.
    Required for models that don't support sparse input (e.g., HistGradientBoostingClassifier).
    
    This class must be defined here to allow proper deserialization of saved HGBT models
    that include this transformer in their pipeline.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if hasattr(X, 'toarray'):
            return X.toarray()
        return X


# %%
# ================
# 3. LOGGING SETUP
# ================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(MODELS_DIR, 'stacking_training.log')),
        logging.StreamHandler(sys.stdout)
    ]
)

# %%
# ================
# 4. HELPER FUNCTIONS
# ================

def load_model_with_fallback(model_name: str, fallback_names: list = None):
    """
    Load a model with fallback naming options.
    
    Parameters
    ----------
    model_name : str
        Primary model name to try
    fallback_names : list, optional
        Alternative names to try if primary fails
        
    Returns
    -------
    model or None
        Loaded model or None if not found
    """
    import warnings
    
    # Suppress version warnings during model loading
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        
        # Try primary name
        path = get_model_path(model_name)
        if os.path.exists(path):
            logging.info(f"Loading {model_name} from {path}")
            try:
                return joblib.load(path)
            except (ValueError, AttributeError) as e:
                logging.warning(f"Failed to load {model_name}: {e}")
                return None
    
    # Try fallbacks
    if fallback_names:
        for fallback in fallback_names:
            path = get_model_path(fallback)
            if os.path.exists(path):
                logging.info(f"Loading {model_name} (fallback: {fallback}) from {path}")
                try:
                    return joblib.load(path)
                except (ValueError, AttributeError) as e:
                    logging.warning(f"Failed to load {fallback}: {e}")
                    continue
    
    logging.warning(f"Model {model_name} not found (tried: {model_name}, {fallback_names})")
    return None


def train_and_save_model(model, model_name, X_train, y_train, X_test, y_test):
    """
    Train a model and save it with performance metrics.
    """
    logging.info(f"\n{'='*70}")
    logging.info(f"Training {model_name}...")
    logging.info(f"{'='*70}")
    
    start_time = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start_time
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    logging.info(f"=== {model_name} Training Complete ({elapsed/60:.2f} minutes) ===")
    logging.info("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))
    logging.info("\nClassification Report:\n" + str(classification_report(y_test, y_pred)))
    logging.info(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}\n")
    
    # Save model
    model_path = get_model_path(model_name)
    joblib.dump(model, model_path)
    logging.info(f"{model_name} saved to '{model_path}'.")
    
    return model


# %%
# ================
# 5. DATA LOADING
# ================

if __name__ == "__main__":
    logging.info("="*70)
    logging.info("STACKING CLASSIFIER TRAINING")
    logging.info("="*70)
    
    # Load preprocessed data
    logging.info("\nLoading preprocessed data...")
    preprocessed_path = get_preprocessed_data_path()
    
    if not os.path.exists(preprocessed_path):
        logging.error(f"Preprocessed data not found at {preprocessed_path}")
        logging.error("Please run sections 1-4 of 03a_model_training.py first to create preprocessed data.")
        sys.exit(1)
    
    data = joblib.load(preprocessed_path)
    X_train_with_indicators = data['X_train_with_indicators']
    X_test_with_indicators = data['X_test_with_indicators']
    y_train = data['y_train']
    y_test = data['y_test']
    
    logging.info(f"Loaded data shapes:")
    logging.info(f"  X_train: {X_train_with_indicators.shape}")
    logging.info(f"  X_test: {X_test_with_indicators.shape}")
    logging.info(f"  y_train: {y_train.shape}")
    logging.info(f"  y_test: {y_test.shape}")
    
    # %%
    # ================
    # 6. LOAD BASE MODELS
    # ================
    
    logging.info("\n" + "="*70)
    logging.info("Loading base models...")
    logging.info("="*70)
    
    model_mapping = {
        'elasticnet': ['lasso'],  # Try 'elasticnet', fallback to 'lasso'
        'elasticnet_2way': ['lasso_2way'],
        'rf': [],
        'gbt': [],
        'hgbt': ['hist_gbt'],  # Try 'hgbt', fallback to 'hist_gbt'
        'xgb': [],
        'cb': [],
    }
    
    loaded_models = {}
    for model_name, fallbacks in model_mapping.items():
        model = load_model_with_fallback(model_name, fallbacks)
        if model is not None:
            loaded_models[model_name] = model
    
    # Check if we have enough models
    if len(loaded_models) < 3:
        logging.error(f"\nInsufficient models loaded ({len(loaded_models)}/7)")
        logging.error("Need at least 3 base models to create a stacking classifier.")
        logging.error("\nAvailable models:")
        for name in loaded_models.keys():
            logging.error(f"  ✓ {name}")
        logging.error("\nMissing models:")
        for name in model_mapping.keys():
            if name not in loaded_models:
                logging.error(f"  ✗ {name}")
        logging.error("\nPlease run sections 5.1-5.7 of 03a_model_training.py to train the base models.")
        sys.exit(1)
    
    logging.info(f"\nSuccessfully loaded {len(loaded_models)}/7 base models:")
    for name in loaded_models.keys():
        logging.info(f"  ✓ {name}")
    
    # %%
    # ================
    # 7. TRAIN STACKING CLASSIFIER
    # ================
    
    # Create estimators list for stacking
    estimators_list = [(name, model) for name, model in loaded_models.items()]
    
    logging.info("\n" + "="*70)
    logging.info("TRAINING STACKING CLASSIFIER")
    logging.info("="*70)
    logging.info(f"Base models: {[name for name, _ in estimators_list]}")
    logging.info(f"Final estimator: LogisticRegression")
    logging.info(f"Cross-validation folds: {N_SPLITS_CV}")
    logging.info(f"n_jobs: 1 (to avoid oversubscription with base models)")
    
    stacking_clf = StackingClassifier(
        estimators=estimators_list,
        final_estimator=LogisticRegression(random_state=RANDOM_STATE),
        cv=N_SPLITS_CV, 
        n_jobs=1,  # Keep sequential to avoid memory/CPU oversubscription
        passthrough=False
    )
    
    train_and_save_model(
        model=stacking_clf,
        model_name="stacking",
        X_train=X_train_with_indicators,
        y_train=y_train,
        X_test=X_test_with_indicators,
        y_test=y_test
    )
    
    logging.info("\n" + "="*70)
    logging.info("STACKING CLASSIFIER TRAINING COMPLETE")
    logging.info("="*70)
    logging.info(f"Model saved to: {get_model_path('stacking')}")
    logging.info(f"Log saved to: {os.path.join(MODELS_DIR, 'stacking_training.log')}")
    
    print("\n" + "="*70)
    print("STACKING CLASSIFIER TRAINING FINISHED SUCCESSFULLY")
    print("="*70)
