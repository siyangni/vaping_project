# %%
# ============================
#  Title:  Model Training Pipeline (Cluster-Friendly)
#  Author: Siyang Ni
#  Notes:  This script handles data loading, preprocessing, and model training
#          for multiple classifiers. Designed for non-interactive cluster runs.
#          All interactive plotting has been moved to 03b_model_analysis.py.
# ============================

# %%
# ================
# 1. IMPORTS
# ================

import os
import logging
import warnings
import joblib
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import sys
import time

from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score, 
    RandomizedSearchCV, RepeatedStratifiedKFold
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    HistGradientBoostingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.inspection import permutation_importance
import optuna
from scipy.stats import randint, uniform, loguniform
import inspect

# Import shared configuration
from model_config import (
    DATA_PATH, MODELS_DIR, RANDOM_STATE, TEST_SIZE, N_SPLITS_CV,
    SCORING_METRIC, VERBOSE, CPU_COUNT, TUNING_PRESET,
    get_tuning_iters, get_model_path, get_preprocessed_data_path,
    FEATURE_RENAME_MAP, TARGET_COL
)

# %%
# ================
# 2. CONFIGURATION
# ================

# Get tuning iterations based on preset
_N_ITER = get_tuning_iters()
logging.info(f"Tuning preset: {TUNING_PRESET} (n_iter={_N_ITER})")

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(MODELS_DIR, 'training.log')),
        logging.StreamHandler()
    ]
)

logging.info(f"Data path: {DATA_PATH}")
logging.info(f"Models directory: {MODELS_DIR}")

# %%
# ================
# 3. HELPER FUNCTIONS
# ================

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads data from a CSV file into a Pandas DataFrame.
    
    Parameters
    ----------
    filepath : str
        Full path to the CSV file.
    
    Returns
    -------
    pd.DataFrame or None
        Loaded DataFrame if successful, None if file not found.
    """
    try:
        df = pd.read_csv(os.path.expanduser(filepath))
        logging.info("Data loaded successfully.")
        return df
    except FileNotFoundError:
        logging.error(f"File not found at {filepath}")
        return None


def identify_categorical_columns(df: pd.DataFrame) -> list:
    """
    Identifies columns of type object or category in a DataFrame.
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    logging.info(f"Identified categorical columns: {categorical_cols}")
    return categorical_cols


def convert_to_categorical(df: pd.DataFrame, columns: list) -> None:
    """
    Converts specified columns in a DataFrame to categorical type in-place.
    """
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype('category')
        else:
            logging.warning(f"Column '{col}' not found in DataFrame.")
    logging.info("Categorical conversion complete.")


def create_train_test_split(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = TEST_SIZE, 
    random_state: int = RANDOM_STATE
) -> tuple:
    """
    Splits data into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=y, shuffle=True
    )
    logging.info(f"Training set shape: {X_train.shape}")
    logging.info(f"Testing set shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def create_missing_indicators(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame
) -> tuple:
    """
    Creates binary indicators for missing values in features.
    """
    missing_indicator = MissingIndicator(features='all')
    missing_indicator.fit(X_train)
    X_train_flags = missing_indicator.transform(X_train)
    X_test_flags = missing_indicator.transform(X_test)
    
    missing_columns = [f'missing_{col}' for col in X_train.columns]
    X_train_with_indicators = pd.concat(
        [X_train.reset_index(drop=True),
         pd.DataFrame(X_train_flags, columns=missing_columns)],
        axis=1
    )
    X_test_with_indicators = pd.concat(
        [X_test.reset_index(drop=True),
         pd.DataFrame(X_test_flags, columns=missing_columns)],
        axis=1
    )
    logging.info("Missing indicators created.")
    return X_train_with_indicators, X_test_with_indicators


def infer_feature_types(
    X: pd.DataFrame,
    categorical_unique_threshold: int = 20,
) -> tuple[list, list]:
    """
    Infer numeric vs categorical feature lists.

    Many survey datasets encode categorical variables as integers. Treating those as
    continuous can materially degrade model quality. This heuristic treats:
      - object/category/bool as categorical
      - numeric columns with <= threshold unique values as categorical
      - remaining numeric columns as numeric
    """
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in X.columns:
        s = X[col]
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s):
            categorical_cols.append(col)
            continue
        if pd.api.types.is_numeric_dtype(s):
            nunique = int(s.nunique(dropna=True))
            if nunique <= categorical_unique_threshold:
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
            continue
        # Fallback: be conservative and treat unknown dtypes as categorical
        categorical_cols.append(col)

    return numeric_cols, categorical_cols


def build_preprocessor(
    numeric_features: list,
    categorical_features: list,
    *,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Create a robust preprocessor for mixed numeric/categorical data."""
    # Keep sparse output compatible with OHE-heavy matrices.
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler(with_mean=False)))

    numeric_transformer = Pipeline(steps=numeric_steps)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


def _make_sparse_poly_features(*, degree: int) -> PolynomialFeatures:
    """
    Create PolynomialFeatures configured to KEEP SPARSE OUTPUT when supported.
    This is critical for feature expansion to avoid OOM errors.

    Version compatibility:
      - sklearn 1.4 and below: `sparse=True` (deprecated parameter)
      - sklearn 1.5–1.7: `sparse_output=True` (renamed parameter)
      - sklearn 1.8+: automatic sparsity preservation (no parameter needed)
    """
    params = inspect.signature(PolynomialFeatures).parameters
    kwargs = dict(degree=degree, interaction_only=True, include_bias=False)
    if "sparse_output" in params:
        kwargs["sparse_output"] = True
    elif "sparse" in params:
        kwargs["sparse"] = True
    # else: sklearn 1.8+ automatically preserves sparsity when input is sparse
    return PolynomialFeatures(**kwargs)


class DenseTransformer(BaseEstimator, TransformerMixin):
    """
    Converts sparse matrices to dense arrays.
    Required for models that don't support sparse input (e.g., HistGradientBoostingClassifier).
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if hasattr(X, 'toarray'):
            return X.toarray()
        return X


def train_and_save_model(model, model_name: str, X_train, y_train, X_test, y_test):
    """
    Train a model and save it to disk, logging evaluation metrics.
    NO PLOTTING - metrics only logged to console/file.
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
# 4. DATA LOADING & PREPROCESSING
# ================

logging.info("\n" + "="*70)
logging.info("PHASE 1: DATA LOADING & PREPROCESSING")
logging.info("="*70)

# --- Data Loading ---
new_data = load_data(DATA_PATH)
if new_data is None:
    logging.error("Data loading failed. Exiting script.")
    raise SystemExit

# Rename columns to descriptive names for modeling
new_data = new_data.rename(columns=FEATURE_RENAME_MAP)

logging.info("Dataset Info:")
new_data.info()

# --- Missing Data Analysis ---
total_missing = new_data.isna().sum().sum()
print("\nTotal missing values:", total_missing)

# Count negative values in numeric columns.
numeric_cols = new_data.select_dtypes(include=[np.number]).columns
negative_counts = new_data[numeric_cols].apply(lambda x: (x < 0).sum())
negative_counts_df = pd.DataFrame({
    'Column': negative_counts.index,
    'Negative_Count': negative_counts.values
})
print("\nNegative value counts by numeric column:")
print(negative_counts_df)

# Replace negative codes (-9, -8) with NaN.
missing_codes = [-9, -8]
new_data[numeric_cols] = new_data[numeric_cols].replace({-9: np.nan, -8: np.nan})

# Compute missing counts and percentages.
missing_counts = new_data.isna().sum()
missing_percent = (new_data.isna().mean() * 100).round(2)
missing_summary = pd.DataFrame({
    'Missing_Count': missing_counts,
    'Missing_Percentage': missing_percent
}).sort_values(by='Missing_Percentage', ascending=False)
print("\nMissing values summary:")
print(missing_summary.to_string())

# --- Identify & Convert Categorical Columns ---
categorical_predictor_cols = new_data.select_dtypes(include=['object', 'category']).columns.tolist()
convert_to_categorical(new_data, categorical_predictor_cols)

logging.info("Verifying data types after conversion:")
logging.info(new_data[categorical_predictor_cols].dtypes)

# --- Train/Test Split ---
X = new_data.drop(TARGET_COL, axis=1)
y = new_data[TARGET_COL]
X_train, X_test, y_train, y_test = create_train_test_split(X, y)

logging.info("Train Set Balance:")
logging.info(y_train.value_counts(normalize=True))
logging.info("Test Set Balance:")
logging.info(y_test.value_counts(normalize=True))

# --- Missing Value Indicators ---
X_train_with_indicators, X_test_with_indicators = create_missing_indicators(X_train, X_test)

# Infer feature types (handles numeric-coded categorical variables)
numeric_features, categorical_features = infer_feature_types(
    X_train_with_indicators,
    categorical_unique_threshold=20,
)
logging.info(f"Inferred numeric features (n={len(numeric_features)}): {numeric_features}")
logging.info(f"Inferred categorical features (n={len(categorical_features)}): {categorical_features}")

# Create a single canonical preprocessor used by all downstream models
preprocessor = build_preprocessor(
    numeric_features=numeric_features,
    categorical_features=categorical_features,
    scale_numeric=True,  # safe default across linear + tree models
)
preprocessor.fit(X_train_with_indicators)
logging.info("Canonical preprocessor fitted successfully.")

# --- Save Preprocessed Data ---
joblib.dump({
    'X_train_with_indicators': X_train_with_indicators,
    'X_test_with_indicators': X_test_with_indicators,
    'y_train': y_train,
    'y_test': y_test,
    'categorical_features': categorical_features,
    'numeric_features': numeric_features,
    'preprocessor': preprocessor,
}, get_preprocessed_data_path())
logging.info(f"Preprocessed data saved to {get_preprocessed_data_path()}")

# %%
# ================
# 5. MODEL TRAINING
# ================

logging.info("\n" + "="*70)
logging.info("PHASE 2: MODEL TRAINING")
logging.info("="*70)

# %%
# ================
# 5.1 ELASTIC NET (Baseline)
# ================

logging.info("\n--- Elastic Net Logistic Regression (Baseline) ---")

elasticnet_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(solver="saga", penalty="elasticnet", l1_ratio=1.0, random_state=RANDOM_STATE)),
    ]
)

cv = RepeatedStratifiedKFold(n_splits=N_SPLITS_CV, n_repeats=1, random_state=RANDOM_STATE)
elasticnet_param_dist = {
    # C: Inverse of regularization strength. Smaller = stronger regularization.
    # Widen range to ensure we capture optimal regularization.
    "classifier__C": loguniform(1e-5, 1e3),
    "classifier__l1_ratio": uniform(0.0, 1.0),
    "classifier__class_weight": [None, "balanced"],
    "classifier__max_iter": [5000],  # Fix at high value to ensure convergence; early stopping will handle it
    "classifier__tol": loguniform(1e-5, 1e-3), # Tighter tolerance for better precision
    "preprocessor__cat__onehot__drop": [None, "first"],
}

# Use 'elasticnet' key if available, else fallback to 'lasso' for now (will update config next)
n_iter_en = _N_ITER.get("elasticnet", _N_ITER.get("lasso", 60))

grid_search = RandomizedSearchCV(
    estimator=elasticnet_pipeline,
    param_distributions=elasticnet_param_dist,
    n_iter=n_iter_en,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    random_state=RANDOM_STATE,
    return_train_score=True,
)

grid_search.fit(X_train_with_indicators, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best CV ROC AUC:", grid_search.best_score_)

best_elasticnet_model = grid_search.best_estimator_
train_and_save_model(
    model=best_elasticnet_model,
    model_name="elasticnet",
    X_train=X_train_with_indicators,
    y_train=y_train,
    X_test=X_test_with_indicators,
    y_test=y_test
)

# %%
# ================
# 5.2 ELASTIC NET (2-way Interactions)
# ================

logging.info("\n--- Elastic Net with 2-way Interactions ---")

# Pre-transform data to avoid redundant computation
print("Pre-transforming data for 2-way interactions (this may take a few minutes)...")
X_train_preprocessed = preprocessor.transform(X_train_with_indicators)
X_test_preprocessed = preprocessor.transform(X_test_with_indicators)

poly_transformer = _make_sparse_poly_features(degree=2)
X_train_poly = poly_transformer.fit_transform(X_train_preprocessed)
X_test_poly = poly_transformer.transform(X_test_preprocessed)

print(f"Original features: {X_train_preprocessed.shape[1]}")
print(f"After polynomial expansion: {X_train_poly.shape[1]}")
print(f"Training samples: {X_train_poly.shape[0]}")

classifier_2way = LogisticRegression(
    solver='saga',
    penalty='elasticnet',
    random_state=RANDOM_STATE,
    max_iter=1000,
    tol=1e-3,
)

param_grid_2way = {
    # Widen search space to capture potential low/high regularization needs
    "C": loguniform(1e-4, 10.0),
    # Allow more Ridge-like behavior if interaction terms cause multicollinearity
    "l1_ratio": uniform(0.5, 0.5), 
    "class_weight": [None, "balanced"],
    "tol": [1e-3],
}

cv_2way = RepeatedStratifiedKFold(n_splits=N_SPLITS_CV, n_repeats=1, random_state=RANDOM_STATE)
n_iter_2way = max(20, _N_ITER.get("elasticnet_interactions_deg2", _N_ITER.get("lasso_interactions_deg2", 30)) // 2)
N_JOBS = min(32, CPU_COUNT)

grid_search_2way = RandomizedSearchCV(
    estimator=classifier_2way,
    param_distributions=param_grid_2way,
    n_iter=n_iter_2way,
    cv=cv_2way,
    scoring="roc_auc",
    n_jobs=N_JOBS,
    verbose=2,
    random_state=RANDOM_STATE,
    return_train_score=True,
)

print(f"Starting RandomizedSearchCV with {n_iter_2way} iterations × {N_SPLITS_CV} folds = {n_iter_2way * N_SPLITS_CV} fits...")
start_time = time.time()

grid_search_2way.fit(X_train_poly, y_train)

elapsed = time.time() - start_time
print(f"\n✓ Training completed in {elapsed/60:.1f} minutes")

print("\n" + "="*70)
print("RESULTS - Degree 2 Interactions")
print("="*70)
print("Best hyperparameters:", grid_search_2way.best_params_)
print("Best cross-validation ROC-AUC: {:.4f}".format(grid_search_2way.best_score_))

best_classifier_2way = grid_search_2way.best_estimator_

# Create a pipeline for compatibility with downstream code
best_model_2way_interactions = Pipeline([
    ('preprocessor', preprocessor),
    ('poly', poly_transformer),
    ('classifier', best_classifier_2way)
])

y_pred_2way = best_classifier_2way.predict(X_test_poly)
y_pred_proba_2way = best_classifier_2way.predict_proba(X_test_poly)[:, 1]

print("Test set ROC-AUC: {:.4f}".format(roc_auc_score(y_test, y_pred_proba_2way)))
print("="*70)

# Save model
model_path_2way = get_model_path("elasticnet_2way")
joblib.dump(best_model_2way_interactions, model_path_2way)
logging.info(f"Elastic Net 2-way interaction model saved to {model_path_2way}")

# %%
# ================
# 5.3 RANDOM FOREST
# ================

logging.info("\n--- Random Forest (Optuna-Optimized) ---")

def rf_optuna_objective(trial):
    """Optuna objective function for Random Forest optimization."""
    print(f"  Starting trial {trial.number}...", flush=True)
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.7]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced', 'balanced_subsample']),
        'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.01),
    }
    
    # Random Forest supports internal parallelism, so we let it use all cores (n_jobs=-1)
    # and keep the CV loop sequential (n_jobs=1).
    rf_clf = RandomForestClassifier(
        **params,
        random_state=RANDOM_STATE,
        n_jobs=-1, 
        oob_score=params['bootstrap'],
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', rf_clf)
    ])
    
    cv_rf = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    
    try:
        scores = cross_val_score(
            pipeline, X_train_with_indicators, y_train,
            cv=cv_rf, scoring=SCORING_METRIC, n_jobs=1
        )
        return scores.mean()
    except Exception as e:
        logging.warning(f"Trial failed with params {params}: {e}")
        return 0.0

try:
    print("Starting Optuna optimization for Random Forest...")
    optuna.logging.set_verbosity(optuna.logging.INFO)
    
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name='rf_optimization'
    )
    
    n_trials = _N_ITER["rf"]
    print(f"Running {n_trials} optimization trials...")
    
    def print_callback(study, trial):
        print(f"Trial {trial.number}: ROC-AUC = {trial.value:.4f} | Best so far: {study.best_value:.4f}")
    
    study.optimize(
        rf_optuna_objective,
        n_trials=n_trials,
        timeout=1800,
        n_jobs=1,
        show_progress_bar=True,
        gc_after_trial=True,
        callbacks=[print_callback],
    )
    
    logging.info(f"Best trial: {study.best_trial.number}")
    logging.info(f"Best cross-validation {SCORING_METRIC}: {study.best_value:.4f}")
    logging.info(f"Best parameters (RF): {study.best_params}")
    
    best_params_rf = study.best_params
    best_rf_clf = RandomForestClassifier(
        n_estimators=best_params_rf['n_estimators'],
        max_depth=best_params_rf['max_depth'],
        min_samples_split=best_params_rf['min_samples_split'],
        min_samples_leaf=best_params_rf['min_samples_leaf'],
        max_features=best_params_rf['max_features'],
        bootstrap=best_params_rf['bootstrap'],
        class_weight=best_params_rf['class_weight'],
        min_impurity_decrease=best_params_rf['min_impurity_decrease'],
        random_state=RANDOM_STATE,
        n_jobs=-1,
        oob_score=best_params_rf['bootstrap'],
    )
    
    best_rf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', best_rf_clf)
    ])
    
    train_and_save_model(
        model=best_rf,
        model_name="rf",
        X_train=X_train_with_indicators,
        y_train=y_train,
        X_test=X_test_with_indicators,
        y_test=y_test
    )
    
    if hasattr(best_rf.named_steps['classifier'], 'oob_score_') and best_rf.named_steps['classifier'].oob_score_:
        logging.info(f"OOB Score: {best_rf.named_steps['classifier'].oob_score_:.4f}")

except Exception as e:
    logging.error(f"An error occurred during Random Forest training: {e}")
    raise

# %%
# ================
# 5.4 GRADIENT BOOSTING (GBT)
# ================

logging.info("\n--- Gradient Boosting Classifier (Optuna-Optimized) ---")

def gbt_optuna_objective(trial):
    """Optuna objective function for GBT optimization."""
    # STRATEGY: Use internal early stopping. Fix n_estimators high, stop when validation score plateaus.
    params = {
        # 'n_estimators': FIXED at 1000 in constructor
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
    }
    
    gbt_clf = GradientBoostingClassifier(
        **params,
        n_estimators=1000, # High cap for early stopping
        random_state=RANDOM_STATE,
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-4,
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', gbt_clf)
    ])
    
    cv_gbt = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    
    try:
        # GBT is sequentially built, so we parallelize the Cross-Validation folds instead.
        # This will train 5 folds in parallel.
        scores = cross_val_score(
            pipeline, X_train_with_indicators, y_train,
            cv=cv_gbt, scoring=SCORING_METRIC, n_jobs=-1
        )
        return scores.mean()
    except Exception as e:
        logging.warning(f"Trial failed: {e}")
        return 0.0

try:
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study_gbt = optuna.create_study(direction='maximize', study_name='gbt_optimization')
    
    n_trials_gbt = _N_ITER["gbt"]
    print(f"Running {n_trials_gbt} optimization trials for GBT...")
    
    study_gbt.optimize(gbt_optuna_objective, n_trials=n_trials_gbt, timeout=1800, show_progress_bar=True)
    
    logging.info(f"Best GBT parameters: {study_gbt.best_params}")
    logging.info(f"Best GBT CV ROC-AUC: {study_gbt.best_value:.4f}")
    
    best_params_gbt = study_gbt.best_params
    best_gbc = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(
            n_estimators=1000, # High cap for early stopping
            learning_rate=best_params_gbt['learning_rate'],
            max_depth=best_params_gbt['max_depth'],
            min_samples_split=best_params_gbt['min_samples_split'],
            min_samples_leaf=best_params_gbt['min_samples_leaf'],
            subsample=best_params_gbt['subsample'],
            max_features=best_params_gbt['max_features'],
            random_state=RANDOM_STATE,
            validation_fraction=0.1,
            n_iter_no_change=10,
            tol=1e-4,
        ))
    ])
    
    train_and_save_model(
        model=best_gbc,
        model_name="gbt",
        X_train=X_train_with_indicators,
        y_train=y_train,
        X_test=X_test_with_indicators,
        y_test=y_test
    )

except Exception as e:
    logging.error(f"Error during GBT training: {e}")
    raise

# %%
# ================
# 5.5 HIST GRADIENT BOOSTING (HGBT)
# ================

logging.info("\n--- Histogram-based Gradient Boosting Classifier ---")

def hgbt_optuna_objective(trial):
    """Optuna objective for HGBT."""
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 10.0),
        'max_bins': trial.suggest_categorical('max_bins', [128, 255]),
    }
    
    hgbt_clf = HistGradientBoostingClassifier(
        **params,
        max_iter=1000, # High cap for early stopping
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('to_dense', DenseTransformer()),  # HGBT requires dense input
        ('classifier', hgbt_clf)
    ])
    
    cv_hgbt = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    
    try:
        scores = cross_val_score(
            pipeline, X_train_with_indicators, y_train,
            cv=cv_hgbt, scoring=SCORING_METRIC, n_jobs=1
        )
        return scores.mean()
    except Exception as e:
        logging.warning(f"Trial failed: {e}")
        return 0.0

try:
    study_hgbt = optuna.create_study(direction='maximize', study_name='hgbt_optimization')
    n_trials_hgbt = _N_ITER["hgbt"]
    
    study_hgbt.optimize(hgbt_optuna_objective, n_trials=n_trials_hgbt, timeout=1800, show_progress_bar=True)
    
    logging.info(f"Best HGBT parameters: {study_hgbt.best_params}")
    logging.info(f"Best HGBT CV ROC-AUC: {study_hgbt.best_value:.4f}")
    
    best_params_hgbt = study_hgbt.best_params
    best_hgbt = Pipeline([
        ('preprocessor', preprocessor),
        ('to_dense', DenseTransformer()),  # HGBT requires dense input
        ('classifier', HistGradientBoostingClassifier(
            learning_rate=best_params_hgbt['learning_rate'],
            max_iter=1000, # High cap for early stopping
            l2_regularization=best_params_hgbt['l2_regularization'],
            max_bins=best_params_hgbt['max_bins'],
            random_state=RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
        ))
    ])
    
    train_and_save_model(
        model=best_hgbt,
        model_name="hgbt",
        X_train=X_train_with_indicators,
        y_train=y_train,
        X_test=X_test_with_indicators,
        y_test=y_test
    )

except Exception as e:
    logging.error(f"Error during HGBT training: {e}")
    raise

# %%
# ================
# 5.6 XGBOOST
# ================

logging.info("\n--- XGBoost Classifier ---")

def xgb_optuna_objective(trial):
    """Optuna objective for XGBoost."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000), # Tuned explicitly
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
    }
    
    xgb_clf = XGBClassifier(
        **params,
        random_state=RANDOM_STATE,
        n_jobs=-1, # Use all cores for tree building
        eval_metric='logloss',
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb_clf)
    ])
    
    cv_xgb = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle= True, random_state=RANDOM_STATE)
    
    try:
        # Keep CV sequential (n_jobs=1) because XGBoost is using all cores internally
        scores = cross_val_score(
            pipeline, X_train_with_indicators, y_train,
            cv=cv_xgb, scoring=SCORING_METRIC, n_jobs=1
        )
        return scores.mean()
    except Exception as e:
        logging.warning(f"Trial failed: {e}")
        return 0.0

try:
    study_xgb = optuna.create_study(direction='maximize', study_name='xgb_optimization')
    n_trials_xgb = _N_ITER["xgb"]
    
    study_xgb.optimize(xgb_optuna_objective, n_trials=n_trials_xgb, timeout=1800, show_progress_bar=True)
    
    logging.info(f"Best XGB parameters: {study_xgb.best_params}")
    logging.info(f"Best XGB CV ROC-AUC: {study_xgb.best_value:.4f}")
    
    best_params_xgb = study_xgb.best_params
    best_xgb = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            n_estimators=best_params_xgb['n_estimators'],
            learning_rate=best_params_xgb['learning_rate'],
            max_depth=best_params_xgb['max_depth'],
            min_child_weight=best_params_xgb['min_child_weight'],
            subsample=best_params_xgb['subsample'],
            colsample_bytree=best_params_xgb['colsample_bytree'],
            gamma=best_params_xgb['gamma'],
            reg_alpha=best_params_xgb['reg_alpha'],
            reg_lambda=best_params_xgb['reg_lambda'],
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric='logloss',
        ))
    ])
    
    train_and_save_model(
        model=best_xgb,
        model_name="xgb",
        X_train=X_train_with_indicators,
        y_train=y_train,
        X_test=X_test_with_indicators,
        y_test=y_test
    )

except Exception as e:
    logging.error(f"Error during XGBoost training: {e}")
    raise

# %%
# ================
# 5.7 CATBOOST
# ================

logging.info("\n--- CatBoost Classifier ---")

def catboost_optuna_objective(trial):
    """Optuna objective for CatBoost."""
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000), # Tuned explicitly
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature',  0.0, 1.0),
    }
    
    cb_clf = CatBoostClassifier(
        **params,
        random_state=RANDOM_STATE,
        verbose=0,
        task_type='CPU',
        thread_count=-1, # Use all available threads
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', cb_clf)
    ])
    
    cv_cb = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    
    try:
        # Keep CV sequential since CatBoost uses all threads
        scores = cross_val_score(
            pipeline, X_train_with_indicators, y_train,
            cv=cv_cb, scoring=SCORING_METRIC, n_jobs=1
        )
        return scores.mean()
    except Exception as e:
        logging.warning(f"Trial failed: {e}")
        return 0.0

try:
    study_cb = optuna.create_study(direction='maximize', study_name='catboost_optimization')
    n_trials_cb = _N_ITER["catboost"]
    
    study_cb.optimize(catboost_optuna_objective, n_trials=n_trials_cb, timeout=1800, show_progress_bar=True)
    
    logging.info(f"Best CatBoost parameters: {study_cb.best_params}")
    logging.info(f"Best CatBoost CV ROC-AUC: {study_cb.best_value:.4f}")
    
    best_params_cb = study_cb.best_params
    best_cb = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', CatBoostClassifier(
            iterations=best_params_cb['iterations'],
            learning_rate=best_params_cb['learning_rate'],
            depth=best_params_cb['depth'],
            l2_leaf_reg=best_params_cb['l2_leaf_reg'],
            border_count=best_params_cb['border_count'],
            bagging_temperature=best_params_cb['bagging_temperature'],
            random_state=RANDOM_STATE,
            verbose=0,
            task_type='CPU',
            thread_count=-1,
        ))
    ])
    
    train_and_save_model(
        model=best_cb,
        model_name="cb",
        X_train=X_train_with_indicators,
        y_train=y_train,
        X_test=X_test_with_indicators,
        y_test=y_test
    )

except Exception as e:
    logging.error(f"Error during CatBoost training: {e}")
    raise

# %%
# ================
# 5.8 STACKING CLASSIFIER
# ================

logging.info("\n--- Stacking Classifier (Ensemble) ---")

# Define base learners using the best pipelines found above
# Note: We use the full pipelines so they can handle the raw feature set independently
estimators_list = [
    ('en', best_elasticnet_model),
    ('en_2way', best_model_2way_interactions),
    ('rf', best_rf),
    ('hgbt', best_hgbt),
    ('xgb', best_xgb),
    ('cb', best_cb),
    # We include GBC as well for diversity, though it is slower to train
    ('gbt', best_gbc),
]

logging.info("Initializing Stacking Classifier...")
logging.info(f"Base models: {[name for name, _ in estimators_list]}")

# We keep n_jobs=1 because the base learners (RF, XGB, CB) are already using all 24 cores.
# Parallelizing the stack would cause massive oversubscription and potential crashes.
stacking_clf = StackingClassifier(
    estimators=estimators_list,
    final_estimator=LogisticRegression(random_state=RANDOM_STATE),
    cv=N_SPLITS_CV, 
    n_jobs=1, 
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

# %%
# ================
# 6. TRAINING SUMMARY
# ================

logging.info("\n" + "="*70)
logging.info("TRAINING COMPLETE")
logging.info("="*70)
logging.info(f"All models saved to: {MODELS_DIR}")
logging.info("Trained models:")
for model_name in ["elasticnet", "elasticnet_2way", "rf", "gbt", "hgbt", "xgb", "cb", "stacking"]:
    path = get_model_path(model_name)
    if os.path.exists(path):
        logging.info(f"  ✓ {model_name}: {path}")
    else:
        logging.warning(f"  ✗ {model_name}: NOT FOUND")

logging.info(f"\nPreprocessed data: {get_preprocessed_data_path()}")
logging.info("\nTo analyze models interactively, run: 03b_model_analysis.py")

print("\n" + "="*70)
print("TRAINING PIPELINE FINISHED SUCCESSFULLY")
print("="*70)
