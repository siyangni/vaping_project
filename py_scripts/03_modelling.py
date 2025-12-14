# %%
# ============================
#  Title:  Multi-Classifier Modeling, Hyperparameter Tuning & Evaluation
#  Author: Siyang Ni
#  Date:   [Date]
#  Notes:  This script builds a comprehensive pipeline for loading data,
#          preprocessing, model training, hyperparameter tuning, and evaluation
#          across multiple algorithms: RandomForest, GradientBoosting,
#          HistGradientBoosting, XGBoost, and CatBoost. Includes interpretability
#          with SHAP, partial dependence plots, and feature importances.
# ============================

# %%
# # Setting Up

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
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score, 
    RandomizedSearchCV, RepeatedStratifiedKFold
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    HistGradientBoostingClassifier 
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import shap
import optuna
from scipy.stats import randint, uniform, loguniform

# %%
# ================
# 2. CONFIGURATION
# ================
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS_CV = 5
SCORING_METRIC = 'roc_auc'
VERBOSE = 1

CPU_COUNT = os.cpu_count()

# ----------------
# Hyperparameter tuning preset
# ----------------
# "standard" is designed to finish in ~1 hour on a typical workstation/server.
# You can override via env var: `TUNING_PRESET=fast|standard|max`.
TUNING_PRESET = os.environ.get("TUNING_PRESET", "standard").strip().lower()


def _tuning_iters(preset: str) -> dict:
    """Centralized tuning budgets per model."""
    if preset == "fast":
        return {
            "lasso": 35,
            "lasso_interactions_deg2": 25,
            "lasso_interactions_deg23": 35,
            "rf": 35,
            "gbt": 35,
            "hgbt": 35,
            "xgb": 35,
            "catboost": 30,
        }
    if preset == "max":
        return {
            "lasso": 120,
            "lasso_interactions_deg2": 80,
            "lasso_interactions_deg23": 120,
            "rf": 120,
            "gbt": 120,
            "hgbt": 120,
            "xgb": 120,
            "catboost": 100,
        }
    # default: standard
    return {
        "lasso": 60,
        "lasso_interactions_deg2": 40,
        "lasso_interactions_deg23": 60,
        "rf": 60,
        "gbt": 60,
        "hgbt": 60,
        "xgb": 60,
        "catboost": 60,
    }


_N_ITER = _tuning_iters(TUNING_PRESET)
logging.info(f"Tuning preset: {TUNING_PRESET} (n_iter={_N_ITER})")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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


def create_preprocessor(categorical_features: list) -> ColumnTransformer:
    """
    Creates a preprocessor for categorical features using OneHotEncoder
    while passing other columns through without transformation.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                'cat', 
                OneHotEncoder(
                    drop='first', 
                    handle_unknown='ignore'
                ),
                categorical_features
            )
        ],
        remainder='passthrough'
    )
    return preprocessor


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


def train_evaluate_model(
    model, 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    model_name: str = "Model", 
    save_path: str = None
):
    """
    Trains, evaluates, and optionally saves a model. 
    Prints confusion matrix, classification report, and ROC AUC.
    Plots the ROC curve.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    logging.info(f"=== {model_name} Evaluation ===")
    logging.info("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))
    logging.info("\nClassification Report:\n" + str(classification_report(y_test, y_pred)))
    logging.info(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_pred_proba):.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve on Test Data')
    plt.legend(loc='lower right')
    plt.show()
    
    if save_path:
        joblib.dump(model, save_path)
        logging.info(f"{model_name} saved to '{save_path}'.")
    
    return model


def perform_grid_search(
    pipeline: Pipeline, 
    param_grid: dict, 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    cv=None, 
    scoring: str = 'roc_auc', 
    n_jobs: int = -1, 
    verbose: int = 1
):
    """
    Performs GridSearchCV for hyperparameter tuning on a pipeline.
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=pipeline, 
        param_grid=param_grid, 
        cv=cv, 
        scoring=scoring, 
        n_jobs=n_jobs, 
        verbose=verbose
    )
    grid_search.fit(X_train, y_train)
    logging.info("Best parameters found: " + str(grid_search.best_params_))
    logging.info(f"Best cross-validation {scoring}: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


def plot_feature_importance(
    model, 
    feature_names: list, 
    top_n: int = 20, 
    title: str = "Feature Importance"
):
    """
    Plots the top N feature importances from a trained model.
    """
    if hasattr(model, 'feature_importances_'): 
        importances = model.feature_importances_
    elif hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
        if hasattr(model.named_steps['classifier'], 'feature_importances_'):
            importances = model.named_steps['classifier'].feature_importances_
        else:
            raise ValueError("Classifier does not have feature_importances_ attribute.")
    else:
        raise ValueError("Provided model does not have feature_importances_ attribute.")
    
    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    fi_df = fi_df.sort_values('Importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=fi_df)
    plt.title(title)
    plt.tight_layout()
    plt.show()


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
    plt.figure(figsize=(10, 6))
    plt.barh(y=top_n_df['Feature'], width=top_n_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# %%
# # Preprocessing

# %%
# --- Data Loading ---
data_filepath = os.path.expanduser('~/autodl-tmp/processed_data_g12nn.csv')
new_data = load_data(data_filepath)
if new_data is None:
    logging.error("Data loading failed. Exiting script.")
    raise SystemExit

# Rename columns to descriptive names for modeling
FEATURE_RENAME_MAP = {
    "V2134": "sedatives_barbiturates_12m",
    "V2166": "political_preference",
    "V2143": "other_narcotics_12m",
    "V2175": "days_missed_school_illness",
    "V2140": "heroin_12m",
    "V2164": "mother_education_level",
    "V2171": "expected_graduation_time",
    "V2156": "mother_in_household",
    "V13": "school_region",
    "V2177": "days_missed_other",
    "V2178": "times_skipped_class",
    "V2152": "growing_up_location",
    "V2179": "average_grades",
    "V2201": "driving_accidents",
    "V2157": "sibling_in_household",
    "V2185": "desire_vocational_school",
    "V2188": "desire_college_grad",
    "V2176": "days_missed_skipping",
    "sex": "respondent_sex",
    "V2163": "father_education_level",
    "RESPONDENT_AGE": "respondent_age",
    "nicotine12d": "nicotine_use_12m",
    "V2189": "desire_grad_prof_school",
    "V2197": "moving_violation_tickets",
    "V2194": "nights_out",
    "V49": "num_siblings",
    "V2137": "tranquilizers_12m",
    "V2182": "likelihood_two_year_grad",
    "V2153": "marital_status",
    "V2101": "ever_smoked",
    "V2116": "marijuana_12m",
    "V2181": "likelihood_military",
    "V2184": "likelihood_grad_prof_school",
    "V2125": "cocaine_12m",
    "wave": "survey_wave",
    "V2105": "alcohol_12m",
    "V2173": "self_rated_school_ability",
    "V2172": "high_school_program",
    "V2460": "crack_12m",
    "V2196": "distance_driven_weekly",
    "V2180": "likelihood_vocational_school",
    "race": "respondent_race",
    "V2195": "dating_frequency",
    "V2187": "desire_two_year_grad",
    "V2191": "hours_worked_per_week",
    "V2193": "weekly_income_other",
    "V2128": "amphetamines_12m",
    "V2186": "desire_military_service",
    "V2155": "father_in_household",
    "V2108": "five_plus_drinks_2wks",
    "V2183": "likelihood_college_grad",
}
new_data = new_data.rename(columns=FEATURE_RENAME_MAP)

TARGET_COL = 'nicotine_use_12m'

logging.info("Dataset Info:")
new_data.info()

# %%
# -----------------------------------------------------------------------------
# Missing Data Analysis
# -----------------------------------------------------------------------------

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

# %%
# Correlation Analysis
# Select numeric variables (excluding the target variable 'nicotine_use_12m' if desired).
cor_vars = new_data.drop(columns=[TARGET_COL], errors='ignore').select_dtypes(include=[np.number])

# Compute the Spearman correlation matrix.
cor_matrix_spearman = cor_vars.corr(method='spearman')

# Check for non-finite values in the correlation matrix.
if not np.all(np.isfinite(cor_matrix_spearman)):
    print("\nWarning: Non-finite values detected in the correlation matrix.")
    # Replace NaN or infinite values with 0 (or another appropriate value).
    cor_matrix_spearman = cor_matrix_spearman.fillna(0)
    cor_matrix_spearman = cor_matrix_spearman.replace([np.inf, -np.inf], 0)

print("\nSpearman Correlation Matrix:")
print(cor_matrix_spearman)

# Create an enhanced heatmap with clustering.
clustergrid = sns.clustermap(cor_matrix_spearman, cmap="coolwarm", figsize=(12, 12))
clustergrid.ax_heatmap.set_title("Enhanced Spearman Correlation Heatmap")
plt.show()

# Identify highly correlated pairs (absolute correlation > 0.5 and less than 1).
high_corr_pairs = []
cols = cor_matrix_spearman.columns
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        corr_value = cor_matrix_spearman.iloc[i, j]
        if 0.5 < abs(corr_value) < 1:
            high_corr_pairs.append({
                "Variable1": cols[i],
                "Variable2": cols[j],
                "Correlation": corr_value
            })

high_corr_df = pd.DataFrame(high_corr_pairs)
print("\nHighly correlated variable pairs (|corr| > 0.5):")
print(high_corr_df)

# %%
# --- Identify & Convert Categorical Columns ---
import logging

# Identify all categorical (object or categorical dtype) columns
categorical_predictor_cols = new_data.select_dtypes(include=['object', 'category']).columns.tolist()

# If you also want to include numerical columns as categorical (optional)
# categorical_predictor_cols = new_data.columns.tolist()  

# Convert identified columns to categorical
convert_to_categorical(new_data, categorical_predictor_cols)

# Logging information
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

# %%
# ================
# SAVE PREPROCESSED DATA
# ================
# Save all training/test data and preprocessor for use without rerunning preprocessing
data_save_dir = os.path.expanduser('~/work/vaping_project_data')
os.makedirs(data_save_dir, exist_ok=True)

# Save train/test splits
joblib.dump({
    'X_train_with_indicators': X_train_with_indicators,
    'X_test_with_indicators': X_test_with_indicators,
    'y_train': y_train,
    'y_test': y_test,
    'categorical_features': categorical_features,
    'numeric_features': numeric_features,
}, os.path.join(data_save_dir, 'preprocessed_data.joblib'))
logging.info(f"Preprocessed data saved to {data_save_dir}/preprocessed_data.joblib")

# %%
# ================
# LOAD PREPROCESSED DATA (for analysis without retraining)
# ================
# Uncomment and run this cell to load preprocessed data instead of rerunning preprocessing
data_load_path = os.path.expanduser('~/work/vaping_project_data/preprocessed_data.joblib')
if os.path.exists(data_load_path):
    loaded_data = joblib.load(data_load_path)
    X_train_with_indicators = loaded_data['X_train_with_indicators']
    X_test_with_indicators = loaded_data['X_test_with_indicators']
    y_train = loaded_data['y_train']
    y_test = loaded_data['y_test']
    categorical_features = loaded_data['categorical_features']
    numeric_features = loaded_data.get('numeric_features', [])
    logging.info("Preprocessed data loaded successfully")
else:
    logging.warning("Preprocessed data file not found - run preprocessing first")

# %%
# # Model Training

# %%
# ## Lasso

# %%
# Canonical pipeline (uses canonical preprocessor built above)
lasso_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        # scikit-learn >= 1.8: `penalty` is deprecated; control regularization with `l1_ratio` and `C`.
        # - l1_ratio=0.0 -> L2
        # - l1_ratio=1.0 -> L1
        # - 0<l1_ratio<1 -> ElasticNet
        ("classifier", LogisticRegression(solver="saga", l1_ratio=1.0, random_state=RANDOM_STATE)),
    ]
)

# Stronger randomized tuning (log-scale for C, continuous l1_ratio)
cv = RepeatedStratifiedKFold(n_splits=N_SPLITS_CV, n_repeats=1, random_state=RANDOM_STATE)
lasso_param_dist = {
    "classifier__C": loguniform(1e-4, 1e2),
    "classifier__l1_ratio": uniform(0.0, 1.0),
    "classifier__class_weight": [None, "balanced"],
    "classifier__max_iter": [3000, 5000, 8000],
    "classifier__tol": loguniform(1e-4, 1e-2),
    # encoder choice can materially affect linear model stability
    "preprocessor__cat__onehot__drop": [None, "first"],
}

grid_search = RandomizedSearchCV(
    estimator=lasso_pipeline,
    param_distributions=lasso_param_dist,
    n_iter=_N_ITER["lasso"],
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

best_lasso_model = grid_search.best_estimator_
train_evaluate_model(
    model=best_lasso_model,
    X_train=X_train_with_indicators,
    y_train=y_train,
    X_test=X_test_with_indicators,
    y_test=y_test,
    model_name="Tuned LASSO Logistic Regression"
)

# %%
# Save the Lasso model
lasso_model_filename = os.path.expanduser('~/work/vaping_project_data/best_lasso_model.joblib')
joblib.dump(best_lasso_model, lasso_model_filename)
logging.info(f"Lasso model saved to {lasso_model_filename}")

# %%
# Load the Lasso model (for analysis without retraining)
lasso_model_filename = os.path.expanduser('~/work/vaping_project_data/best_lasso_model.joblib')
best_lasso_model = joblib.load(lasso_model_filename)
logging.info("Lasso model loaded successfully")

# %%
import numpy as np
import pandas as pd

# Extract the logistic regression model from the pipeline.
lr = best_lasso_model.named_steps['classifier']

# For binary classification, lr.coef_ has shape (1, n_features)
coefficients = lr.coef_[0]

# Get the preprocessor (the ColumnTransformer) from the pipeline.
preprocessor = best_lasso_model.named_steps['preprocessor']

# -------------------------------
# 1. Numeric Features and Importances
# -------------------------------
# The numeric transformer was applied first.
numeric_features = preprocessor.transformers_[0][2]  # list (or Index) of numeric feature names
n_numeric = len(numeric_features)
numeric_coefs = coefficients[:n_numeric]
numeric_importances = pd.Series(np.abs(numeric_coefs), index=numeric_features)

# -------------------------------
# 2. Categorical Features (Aggregation)
# -------------------------------
# Get the original categorical columns from the transformer.
cat_features = preprocessor.transformers_[1][2]

# Check if there are any categorical features
if len(cat_features) > 0:
    # Retrieve the OneHotEncoder from the categorical pipeline.
    onehot_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    
    # The remaining coefficients correspond to the one-hot encoded features.
    categorical_coefs = coefficients[n_numeric:]
    
    aggregated_cat_importance = {}
    start_idx = 0
    # Loop over each original categorical feature and its categories.
    for feature, categories in zip(cat_features, onehot_encoder.categories_):
        n_categories = len(categories)
        # Get the coefficients for the dummy columns of this feature.
        feature_coefs = categorical_coefs[start_idx:start_idx + n_categories]
        # Aggregate by summing the absolute values.
        aggregated_cat_importance[feature] = np.sum(np.abs(feature_coefs))
        start_idx += n_categories

    aggregated_cat_importance = pd.Series(aggregated_cat_importance)
else:
    # If there are no categorical features, create an empty Series.
    aggregated_cat_importance = pd.Series(dtype=float)

# -------------------------------
# 3. Combine and Select Top 20
# -------------------------------
combined_importances = pd.concat([numeric_importances, aggregated_cat_importance])
top20_features = combined_importances.sort_values(ascending=False).head(20)

print("Top 20 Aggregated Feature Importances (by absolute coefficient value):")
print(top20_features)

# Plot the top 20 features
plt.figure(figsize=(10, 6))
sns.barplot(x=top20_features.values, y=top20_features.index, palette="viridis")
plt.title('Top 20 Aggregated Feature Importances (by absolute coefficient value)')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# %%
#########################################
# 2. Permutation Importance (Aggregated by Original Feature)
#########################################

from sklearn.inspection import permutation_importance

# Compute permutation importance using the original features (X_test_with_indicators).
perm_results = permutation_importance(
    best_lasso_model,
    X_test_with_indicators,
    y_test,
    scoring='roc_auc',
    n_repeats=10,
    random_state=RANDOM_STATE,
    n_jobs=-1  # Use all available cores for parallel computation
)

perm_imp_df = pd.DataFrame({
    'Feature': X_test_with_indicators.columns,
    'Importance': perm_results.importances_mean
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=perm_imp_df.head(20), palette='magma')
plt.title("Permutation Importance (Aggregated Original Features)")
plt.xlabel("Mean Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Display the top 20 features by permutation importance
print(perm_imp_df.head(20))

# %%
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# SHAP must be computed in the *same feature space* the classifier sees:
# i.e., the output of the preprocessor (numeric + one-hot categorical columns).
_pre = best_lasso_model.named_steps["preprocessor"]
X_train_pre = _pre.transform(X_train_with_indicators)

# Use a small background sample for stability + speed.
_bg_n = min(2000, X_train_pre.shape[0])
X_bg = X_train_pre[:_bg_n]

explainer = shap.LinearExplainer(best_lasso_model.named_steps["classifier"], X_bg)
shap_values = explainer.shap_values(X_train_pre)

# Binary classification can sometimes return a list; take the "positive class".
if isinstance(shap_values, list):
    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

transformed_feature_names = _pre.get_feature_names_out()
mean_abs_shap = np.abs(shap_values).mean(axis=0)

# Aggregate transformed feature importances back to original feature names so we can
# use them with X_train_with_indicators / PDP later.
try:
    _cat_cols = list(_pre.transformers_[1][2])
except Exception:
    _cat_cols = []
_cat_cols_sorted = sorted(_cat_cols, key=lambda s: len(str(s)), reverse=True)

feature_importance = {}
for fname, imp in zip(transformed_feature_names, mean_abs_shap):
    # ColumnTransformer prefixes with "<transformer>__"
    if fname.startswith("num__"):
        original = fname.split("__", 1)[1]
    elif fname.startswith("cat__"):
        # OneHotEncoder names look like "<feature>_<category...>".
        # IMPORTANT: feature names themselves can contain underscores, so we match against
        # known categorical columns rather than splitting on the first "_".
        remainder = fname.split("__", 1)[1]
        original = None
        for col in _cat_cols_sorted:
            col = str(col)
            if remainder == col or remainder.startswith(col + "_"):
                original = col
                break
        if original is None:
            # Fallback: best-effort split
            original = remainder.split("_", 1)[0]
    else:
        original = fname
    feature_importance[original] = feature_importance.get(original, 0.0) + float(imp)

# Convert to DataFrame and sort
importance_df = pd.DataFrame({
    'Feature': list(feature_importance.keys()),
    'Importance': list(feature_importance.values())
}).sort_values('Importance', ascending=False)

# Display top 20 features
print("\nTop 20 Important Features:")
print(importance_df.head(20))

# Create visualization
plt.figure(figsize=(12, 8))
sns.barplot(data=importance_df.head(20), x='Importance', y='Feature')
plt.title('Top 20 Feature Importance Based on SHAP Values')
plt.xlabel('Mean Absolute SHAP Value')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# %%
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import numpy as np

def _nunique_nonnull(s):
    return s.dropna().nunique()

def plot_discrete_pdp(estimator, X, feature, ax, max_levels=12):
    """Manual PDP for discrete (incl. binary/categorical) features in original feature space."""
    values = X[feature].dropna().unique()
    if len(values) < 2:
        raise ValueError(f"{feature} needs at least two unique values for PDP")
    if len(values) > max_levels:
        raise ValueError(f"{feature} has too many unique values ({len(values)}) for discrete PDP")

    # Keep a stable ordering (numeric sorted; otherwise as-seen)
    try:
        values = np.sort(values)
    except Exception:
        values = list(values)

    X_work = X.copy()
    pdp_vals = []
    for val in values:
        X_work[feature] = val
        proba = estimator.predict_proba(X_work)[:, 1]
        pdp_vals.append(float(np.mean(proba)))

    ax.plot(range(len(values)), pdp_vals, marker="o")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([str(v) for v in values], rotation=45, ha="right")
    ax.set_xlabel(feature)
    ax.set_ylabel("Partial dependence (avg predicted P(y=1))")

# Pick the top-10 *real* features that exist in X and actually vary.
top_10_features = []
for f in importance_df["Feature"].tolist():
    if f not in X_train_with_indicators.columns:
        continue
    if _nunique_nonnull(X_train_with_indicators[f]) < 2:
        continue
    top_10_features.append(f)
    if len(top_10_features) == 10:
        break

print("Top 10 Features:", top_10_features)
for feature in top_10_features:
    print(f"{feature}: {X_train_with_indicators[feature].dtype}")

print("Unique Values in Top 10 Features:")
feature_uniques = {}
for feature in top_10_features:
    unique_values = X_train_with_indicators[feature].dropna().unique()
    feature_uniques[feature] = unique_values
    print(f"{feature}: {unique_values}")


def plot_continuous_pdp(estimator, X, feature, ax):
    """Try sklearn PDP first; if it fails, fallback to a simple quantile grid."""
    try:
        PartialDependenceDisplay.from_estimator(
            estimator,
            X,
            features=[feature],
            ax=ax,
            grid_resolution=20
        )
        ax.set_xlabel(feature)
        ax.set_ylabel("Partial dependence")
        return
    except Exception:
        pass

    # Fallback: manual grid over quantiles
    s = X[feature]
    qs = np.linspace(0.05, 0.95, 10)
    grid = np.unique(np.quantile(s.dropna().to_numpy(), qs))
    if len(grid) < 2:
        raise ValueError(f"{feature} does not vary enough for PDP")
    X_work = X.copy()
    pdp_vals = []
    for val in grid:
        X_work[feature] = val
        proba = estimator.predict_proba(X_work)[:, 1]
        pdp_vals.append(float(np.mean(proba)))
    ax.plot(grid, pdp_vals, marker="o")
    ax.set_xlabel(feature)
    ax.set_ylabel("Partial dependence (avg predicted P(y=1))")


fig, axes = plt.subplots(2, 5, figsize=(20, 10))
axes = axes.flatten()
if len(top_10_features) == 0:
    print("No varying features found in importance_df that exist in X_train_with_indicators; skipping PDP.")
    for ax in axes:
        ax.set_visible(False)
else:
    for i, feature in enumerate(top_10_features):
        ax = axes[i]
        uniques = feature_uniques[feature]
        try:
            # If discrete/low-cardinality (incl. binary/categorical), do manual PDP in original space.
            if len(uniques) <= 12:
                plot_discrete_pdp(best_lasso_model, X_train_with_indicators, feature, ax, max_levels=12)
            # Otherwise treat as continuous (numeric) and attempt sklearn PDP (with fallback).
            else:
                plot_continuous_pdp(best_lasso_model, X_train_with_indicators, feature, ax)
            ax.set_title(f"PDP for {feature}")
        except Exception as e:
            print(f"Error plotting PDP for {feature}: {e}")
            ax.set_visible(False)

    # Hide any unused axes if <10 features
    for j in range(len(top_10_features), len(axes)):
        axes[j].set_visible(False)

plt.tight_layout()
plt.show()

# %%
#############################
# Degree 2 Interaction 
###############################

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from joblib import Memory
import os
import inspect

# Set a random state for reproducibility.
RANDOM_STATE = 42

def _make_sparse_poly_features(*, degree: int) -> PolynomialFeatures:
    """
    Create PolynomialFeatures configured to KEEP SPARSE OUTPUT when supported.

    scikit-learn compatibility:
      - sklearn 0.21–1.4: `sparse=True` (explicit parameter)
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

# ==============================================================================
# PERFORMANCE OPTIMIZATION: Pre-transform data to avoid redundant computation
# ==============================================================================
# Instead of having the pipeline repeat preprocessing + polynomial expansion 200 times
# (40 candidates × 5 folds), we do it ONCE here and pass the transformed data.
print("Pre-transforming data (this may take a few minutes)...")
X_train_preprocessed = preprocessor.transform(X_train_with_indicators)
X_test_preprocessed = preprocessor.transform(X_test_with_indicators)

# Apply polynomial features
poly_transformer = _make_sparse_poly_features(degree=2)
X_train_poly = poly_transformer.fit_transform(X_train_preprocessed)
X_test_poly = poly_transformer.transform(X_test_preprocessed)

print(f"Original features: {X_train_preprocessed.shape[1]}")
print(f"After polynomial expansion: {X_train_poly.shape[1]}")
print(f"Training samples: {X_train_poly.shape[0]}")
print(f"Data is sparse: {hasattr(X_train_poly, 'nnz')}")

# Memory usage estimation
if hasattr(X_train_poly, 'nnz'):
    sparsity = X_train_poly.nnz / (X_train_poly.shape[0] * X_train_poly.shape[1])
    memory_mb = X_train_poly.data.nbytes / (1024**2)
    print(f"Sparsity: {sparsity:.4f} (lower is sparser)")
    print(f"Sparse matrix memory: ~{memory_mb:.1f} MB")
    
    # Warn if feature space is too large
    if X_train_poly.shape[1] > 10000:
        print(f"\n⚠️  WARNING: Very large feature space ({X_train_poly.shape[1]} features)")
        print("This may cause memory issues even with reduced parallelism.")
        print("Consider using feature selection or reducing the polynomial degree if training is too slow.")

# ----------------------------
# 1. Build the Classifier (without preprocessing pipeline)
# ----------------------------
# MODERN SKLEARN 1.8+ API: Use l1_ratio instead of deprecated penalty parameter
# - l1_ratio=1.0: Pure L1 regularization (LASSO)
# - l1_ratio=0.0: Pure L2 regularization (Ridge)
# - 0 < l1_ratio < 1: Elastic net (combination of L1 and L2)
# 
# Note: We use 'saga' solver (supports l1_ratio for sklearn 1.8+)
# While slightly slower than liblinear per iteration, with pre-transformed data
# and the optimizations above, it's still much faster than the original approach.
classifier = LogisticRegression(
    solver='saga',  # Supports l1_ratio (required for sklearn 1.8+ elastic net API)
    random_state=RANDOM_STATE,
    max_iter=1000,  # Reduced from 5000
    tol=1e-3,  # More lenient for speed
    # n_jobs removed: deprecated in sklearn 1.8+, parallelism handled at CV level
)

# ----------------------------
# 2. Set Up Hyperparameter Tuning
# ----------------------------
# OPTIMIZED parameter grid - removes slow combinations for practical training time
# Based on empirical testing with 62K features:
# - High C values (>0.1) cause 30+ minute fits
# - tol=0.0001 causes 3x longer training than tol=0.001
# - l1_ratio<0.9 (elastic net) is slower than pure L1
param_grid = {
    "C": loguniform(1e-3, 1e-1),  # Narrower range - avoids slow high-C values
    "l1_ratio": [0.9, 1.0],  # Near-L1 and pure L1 only (fastest, most sparse)
    "class_weight": [None, "balanced"],
    "tol": [1e-3],  # Single value - 0.001 is sufficient for model selection
}


# ----------------------------
# 3. Create and Fit RandomizedSearchCV
# ----------------------------
# Use ROC-AUC (primary objective used across the project)
cv = RepeatedStratifiedKFold(n_splits=N_SPLITS_CV, n_repeats=1, random_state=RANDOM_STATE)

# Reduced n_iter for faster results - adjust if needed
n_iter_fast = max(20, _N_ITER["lasso_interactions_deg2"] // 2)

# ==============================================================================
# MEMORY vs SPEED TRADEOFF
# ==============================================================================
# n_jobs controls parallelism but also memory usage:
# - n_jobs=1:  Sequential (slowest, ~30-60 min, but uses minimal memory)
# - n_jobs=4:  4 parallel workers (balanced, ~15-30 min, moderate memory)
# - n_jobs=8:  8 parallel workers (faster, but may OOM with >10K features)
# - n_jobs=24: 24 parallel workers (can OOM even with 120GB RAM)
#
# If you get MemoryError or TerminatedWorkerError (OOM killer), reduce n_jobs.
N_JOBS = 32  # Adjust based on available memory

grid_search = RandomizedSearchCV(
    estimator=classifier,
    param_distributions=param_grid,
    n_iter=n_iter_fast,  # Reduced search space
    cv=cv,
    scoring="roc_auc",
    n_jobs=N_JOBS,
    verbose=2,  # More verbose to see progress
    random_state=RANDOM_STATE,
    return_train_score=True,
)

print(f"Starting RandomizedSearchCV with {n_iter_fast} iterations × {N_SPLITS_CV} folds = {n_iter_fast * N_SPLITS_CV} fits...")
import time
start_time = time.time()

# Fit using pre-transformed data (MUCH faster!)
grid_search.fit(X_train_poly, y_train)

elapsed = time.time() - start_time
print(f"\n✓ Training completed in {elapsed/60:.1f} minutes")

# ----------------------------
# 4. Evaluate the Best Model
# ----------------------------
print("\n" + "="*70)
print("RESULTS - Degree 2 Interactions")
print("="*70)
print("Best hyperparameters:", grid_search.best_params_)
print("Best cross-validation ROC-AUC: {:.4f}".format(grid_search.best_score_))

# Use the best model to predict on the test set.
best_classifier = grid_search.best_estimator_

# Create a pipeline for compatibility with downstream code
# (wraps the classifier with the pre-fitted transformers)
best_model_2way_interactions = Pipeline([
    ('preprocessor', preprocessor),
    ('poly', poly_transformer),
    ('classifier', best_classifier)
])

# Predict on test set using transformed data
y_pred = best_classifier.predict(X_test_poly)
y_pred_proba = best_classifier.predict_proba(X_test_poly)[:, 1]

test_accuracy = accuracy_score(y_test, y_pred)
test_roc_auc = roc_auc_score(y_test, y_pred_proba)

print("Test set accuracy: {:.4f}".format(test_accuracy))
print("Test set ROC-AUC: {:.4f}".format(test_roc_auc))
print("="*70)

# %%
# Save the 2-way interaction Lasso model
lasso_2way_model_filename = os.path.expanduser('~/work/vaping_project_data/best_lasso_2way_model.joblib')
joblib.dump(best_model_2way_interactions, lasso_2way_model_filename)
logging.info(f"Lasso 2-way interaction model saved to {lasso_2way_model_filename}")

# %%
# Load the 2-way interaction Lasso model (for analysis without retraining)
lasso_2way_model_filename = os.path.expanduser('~/work/vaping_project_data/best_lasso_2way_model.joblib')
best_model_2way_interactions = joblib.load(lasso_2way_model_filename)
best_model = best_model_2way_interactions  # For compatibility with analysis code below
logging.info("Lasso 2-way interaction model loaded successfully")

# %%
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    classification_report,
    roc_curve  # For ROC curve
)
import matplotlib.pyplot as plt  # For plotting

# ----------------------------
# 4. Evaluate the Best Model
# ----------------------------
print("Best hyperparameters:", grid_search.best_params_)
print("Best cross-validation accuracy: {:.4f}".format(grid_search.best_score_))

# Use the best model to predict on the test set.
# IMPORTANT:
# In this "Degree 2 Interaction" section, `grid_search` is run on *pre-transformed* data
# (`X_train_poly`). Therefore `grid_search.best_estimator_` is a bare LogisticRegression
# that expects `X_test_poly`, NOT the raw `X_test_with_indicators` (which still contains NaNs).
#
# For downstream code compatibility (feature names, coefficients, SHAP helpers, etc.) we keep
# `best_model` as the wrapped pipeline that contains the (already fitted) preprocessor + poly
# transformer + tuned classifier.
best_model = best_model_2way_interactions

# Predict on the test set using the same transformed matrix used during tuning.
_clf = best_model.named_steps["classifier"]
y_pred = _clf.predict(X_test_poly)
y_pred_proba = _clf.predict_proba(X_test_poly)[:, 1]  # probabilities for the positive class

# Calculate additional metrics
test_accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)

print("Test set accuracy: {:.4f}".format(test_accuracy))
print("Test set ROC AUC: {:.4f}".format(roc_auc))
print("Test set F1 score: {:.4f}".format(f1))

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ----------------------------
# 5. Plot the ROC Curve
# ----------------------------
# Compute FPR, TPR, and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})", color='blue')
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")  # Diagonal line for reference
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# %%
# -----------------------------
# Rank interactions at the ORIGINAL VARIABLE level (collapse one-hot levels)
# -----------------------------
from sklearn.utils.validation import check_is_fitted

# Placeholder so static analysis (and out-of-order execution) doesn't choke;
# the real value is set later when the degree=3 model is trained/loaded.
best_model_3way_interactions = globals().get("best_model_3way_interactions", None)


def _map_preprocessed_feature_to_original(
    feat_name: str,
    *,
    numeric_features: list,
    categorical_features: list,
) -> str:
    """
    Map a preprocessor output feature name (e.g. 'num__age', 'cat__gender_Male')
    back to its ORIGINAL column name (e.g. 'age', 'gender'), collapsing one-hot levels.
    """
    if feat_name.startswith("num__"):
        return feat_name[len("num__"):]

    if feat_name.startswith("cat__"):
        rem = feat_name[len("cat__"):]
        # Prefer the LONGEST matching original categorical feature prefix to handle names with underscores.
        candidates = [
            col for col in categorical_features
            if rem == col or rem.startswith(col + "_")
        ]
        if candidates:
            return max(candidates, key=len)
        # Fallback heuristic if we can't match cleanly.
        return rem.split("_", 1)[0]

    # Fallback: try to recover original name by matching known columns.
    if feat_name in numeric_features or feat_name in categorical_features:
        return feat_name
    return feat_name


def rank_original_interactions_from_lasso(
    pipe: Pipeline,
    *,
    X_fit,
    y_fit,
    numeric_features: list,
    categorical_features: list,
) -> pd.DataFrame:
    """
    Returns a DataFrame of interaction importance aggregated to ORIGINAL variable combinations.
    Importance = sum(|coef|) across all expanded (incl. one-hot) terms that map to the same combo.
    """
    pre = pipe.named_steps["preprocessor"]
    poly = pipe.named_steps["poly"]
    clf = pipe.named_steps["classifier"]

    # Ensure transformers are fitted (robust to out-of-order cell execution).
    try:
        check_is_fitted(pre)
    except Exception:
        pre.fit(X_fit, y_fit)

    try:
        check_is_fitted(poly)
    except Exception:
        poly.fit(pre.transform(X_fit))

    pre_names = pre.get_feature_names_out()
    poly_names = poly.get_feature_names_out(pre_names)

    coefs = np.asarray(clf.coef_).ravel()
    if coefs.shape[0] != len(poly_names):
        raise ValueError(
            f"Coefficient length mismatch: len(coefs)={coefs.shape[0]} vs len(features)={len(poly_names)}"
        )

    df = pd.DataFrame(
        {
            "poly_feature": poly_names,
            "coef": coefs,
            "abs_coef": np.abs(coefs),
        }
    )

    # Interaction terms contain spaces; main effects do not.
    df_int = df[df["poly_feature"].str.contains(" ", regex=False)].copy()

    def _term_to_original_combo(term: str) -> tuple[str, ...]:
        parts = term.split(" ")
        orig = [
            _map_preprocessed_feature_to_original(
                p,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
            )
            for p in parts
        ]
        # Collapse any duplicates (shouldn't happen with interaction_only=True but keep safe).
        return tuple(sorted(set(orig)))

    df_int["original_combo"] = df_int["poly_feature"].apply(_term_to_original_combo)
    df_int["order"] = df_int["original_combo"].apply(len)

    agg = (
        df_int.groupby(["order", "original_combo"], as_index=False)["abs_coef"]
        .sum()
        .rename(columns={"abs_coef": "aggregated_importance"})
        .sort_values(["order", "aggregated_importance"], ascending=[True, False])
    )
    return agg


# --- 2-way interaction model (degree=2) ---
agg_deg2 = rank_original_interactions_from_lasso(
    best_model,
    X_fit=X_train_with_indicators,
    y_fit=y_train,
    numeric_features=numeric_features,
    categorical_features=categorical_features,
)

top20_deg2 = agg_deg2[agg_deg2["order"] == 2].head(20)
print("\nTop 20 ORIGINAL-VARIABLE 2-way interactions (degree=2 model; sum(|coef|) over one-hot levels):")
print(top20_deg2)


# --- 3-way interaction model (degree=3), if available ---
if "best_model_3way_interactions" in globals() and best_model_3way_interactions is not None:
    agg_deg3 = rank_original_interactions_from_lasso(
        best_model_3way_interactions,
        X_fit=X_train_with_indicators,
        y_fit=y_train,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    top20_deg3_2way = agg_deg3[agg_deg3["order"] == 2].head(20)
    top20_deg3_3way = agg_deg3[agg_deg3["order"] == 3].head(20)

    print("\nTop 20 ORIGINAL-VARIABLE 2-way interactions (degree=3 model; sum(|coef|) over one-hot levels):")
    print(top20_deg3_2way)

    print("\nTop 20 ORIGINAL-VARIABLE 3-way interactions (degree=3 model; sum(|coef|) over one-hot levels):")
    print(top20_deg3_3way)

    # Combined ranking across 2-way + 3-way within the degree=3 model
    top_combined = agg_deg3.sort_values("aggregated_importance", ascending=False).head(30)
    print("\nTop 30 ORIGINAL-VARIABLE interactions (combined 2-way + 3-way; degree=3 model):")
    print(top_combined)
else:
    print(
        "\n[Info] 3-way model not available (`best_model_3way_interactions` is None or not created yet). "
        "Run the 3-way training cell above or load `~/work/vaping_project_data/best_lasso_3way_model.joblib`."
    )

# %%
#############################################
# SHAP (2-way interaction Lasso / degree=2)
#############################################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.base import clone


def _get_poly_feature_names_from_pipeline(pipe: Pipeline):
    """Returns feature names after preprocessor + PolynomialFeatures."""
    pre = pipe.named_steps["preprocessor"]
    poly = pipe.named_steps["poly"]
    base_names = pre.get_feature_names_out()
    return poly.get_feature_names_out(base_names)


def _transform_to_poly_matrix(pipe: Pipeline, X):
    """Transforms raw X -> preprocessor -> poly feature matrix."""
    pre = pipe.named_steps["preprocessor"]
    poly = pipe.named_steps["poly"]
    X_pre = pre.transform(X)
    X_poly = poly.transform(X_pre)
    return X_poly


def _fit_shap_and_save_plots_for_interaction_lasso(
    pipe: Pipeline,
    X_train,
    out_dir: str,
    tag: str,
    sample_size: int = 2000,
    background_size: int = 200,
    max_features: int = 400,
    random_state: int = 42,
    max_display: int = 20,
):
    """
    Computes SHAP for the interaction Lasso pipeline (preprocessor + poly + LogisticRegression),
    and saves:
      - SHAP beeswarm summary plot
      - SHAP bar summary plot
      - top-N dependence plots
      - CSV of mean(|SHAP|) feature importance
    """
    os.makedirs(out_dir, exist_ok=True)

    clf = pipe.named_steps["classifier"]
    poly_feature_names = np.asarray(_get_poly_feature_names_from_pipeline(pipe))
    X_poly = _transform_to_poly_matrix(pipe, X_train)

    # ------------------------------------------------------------------
    # Helper: map poly feature names (preprocessor + PolynomialFeatures)
    # back to ORIGINAL variable names (collapse one-hot levels)
    # ------------------------------------------------------------------
    # We rely on the globally-defined `numeric_features` / `categorical_features`
    # lists created earlier in the notebook/script.
    def _poly_name_to_original_combo_label(poly_name: str) -> str:
        parts = str(poly_name).split(" ")

        def _map_part(part: str) -> str:
            if part.startswith("num__"):
                return part[len("num__"):]
            if part.startswith("cat__"):
                rem = part[len("cat__"):]
                # Prefer longest matching categorical column prefix (handles underscores in column names)
                candidates = [
                    col for col in categorical_features
                    if rem == col or rem.startswith(col + "_")
                ]
                if candidates:
                    return max(candidates, key=len)
                return rem.split("_", 1)[0]
            return part

        orig = sorted(set(_map_part(p) for p in parts if p))
        # Use a single "x" to denote interaction, per reporting convention requested.
        return " x ".join(orig)

    # --- sample rows for compute / plots ---
    rng = np.random.default_rng(random_state)
    n_rows = X_poly.shape[0]
    if n_rows == 0:
        raise ValueError("Empty training data passed to SHAP computation.")

    sample_idx = rng.choice(n_rows, size=min(sample_size, n_rows), replace=False)
    bg_idx = rng.choice(n_rows, size=min(background_size, n_rows), replace=False)
    X_sample = X_poly[sample_idx]
    X_bg = X_poly[bg_idx]

    # --- restrict columns to keep memory reasonable (favor non-zero / largest coefficients) ---
    coef = clf.coef_[0]
    intercept = float(clf.intercept_[0]) if hasattr(clf, "intercept_") and len(clf.intercept_) else 0.0

    nonzero = np.flatnonzero(coef)
    selected = nonzero if nonzero.size > 0 else np.arange(coef.shape[0])
    if selected.size > max_features:
        top = np.argsort(np.abs(coef[selected]))[::-1][:max_features]
        selected = selected[top]
    selected = np.sort(selected)

    X_sample_sel = X_sample[:, selected]
    X_bg_sel = X_bg[:, selected]
    feature_names_sel = poly_feature_names[selected]

    # SHAP plotting utilities (notably dependence_plot) may call `len(X)`.
    # SciPy sparse *arrays* raise: "sparse array length is ambiguous".
    # We keep the matrices sparse up to this point, then densify the *small*
    # sampled/feature-selected matrices (sample_size × max_features) for plotting.
    def _to_dense(mat):
        return mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)

    X_sample_sel = _to_dense(X_sample_sel)
    X_bg_sel = _to_dense(X_bg_sel)

    # --- build a tiny "reduced" linear model so SHAP matches the selected columns ---
    class _ReducedLinearModel:
        def __init__(self, coef_1d: np.ndarray, intercept_0: float):
            self.coef_ = np.asarray(coef_1d, dtype=float).reshape(1, -1)
            self.intercept_ = np.asarray([intercept_0], dtype=float)

    reduced_model = _ReducedLinearModel(coef[selected], intercept)

    explainer = shap.LinearExplainer(reduced_model, X_bg_sel)
    shap_values = explainer.shap_values(X_sample_sel)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    # ------------------------------------------------------------------
    # Aggregate SHAP + feature values to ORIGINAL-variable level.
    # This collapses one-hot levels and shows interactions in terms of original columns.
    # ------------------------------------------------------------------
    group_labels = np.asarray([_poly_name_to_original_combo_label(n) for n in feature_names_sel], dtype=object)
    uniq_labels, inv = np.unique(group_labels, return_inverse=True)

    # Aggregate SHAP by summing contributions from all expanded features that map to the same original combo
    shap_values_agg = np.zeros((shap_values.shape[0], uniq_labels.shape[0]), dtype=float)
    for j in range(shap_values.shape[1]):
        shap_values_agg[:, inv[j]] += shap_values[:, j]

    # Aggregate feature values similarly (so beeswarm/dependence have something to plot/color)
    X_sample_agg = np.zeros((X_sample_sel.shape[0], uniq_labels.shape[0]), dtype=float)
    X_bg_agg = np.zeros((X_bg_sel.shape[0], uniq_labels.shape[0]), dtype=float)
    for j in range(X_sample_sel.shape[1]):
        X_sample_agg[:, inv[j]] += X_sample_sel[:, j]
        X_bg_agg[:, inv[j]] += X_bg_sel[:, j]

    # Swap to aggregated representations for all plots/exports below
    shap_values = shap_values_agg
    X_sample_sel = X_sample_agg
    X_bg_sel = X_bg_agg
    feature_names_sel = uniq_labels

    # --- export mean(|SHAP|) importance ---
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    shap_imp_df = (
        pd.DataFrame({"feature": feature_names_sel, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    shap_imp_path = os.path.join(out_dir, f"shap_{tag}_importance.csv")
    shap_imp_df.to_csv(shap_imp_path, index=False)
    print(f"[SHAP] Saved mean(|SHAP|) importances to: {shap_imp_path}")

    # Also save + show a compact table version (same information as the bar/summary plots).
    top_table = shap_imp_df.head(max_display).copy()
    top_table.insert(0, "rank", np.arange(1, len(top_table) + 1))
    top_table_path = os.path.join(out_dir, f"shap_{tag}_top{len(top_table)}_table.csv")
    top_table.to_csv(top_table_path, index=False)
    print(f"[SHAP] Saved top-{len(top_table)} table to: {top_table_path}")
    try:
        from IPython.display import display  # type: ignore
        display(top_table)
    except Exception:
        print(top_table.to_string(index=False))

    # --- custom horizontal bar chart for top features (same numbers as the table above) ---
    top_20_df = shap_imp_df.head(max_display).copy()
    top_20_df = top_20_df.sort_values("mean_abs_shap", ascending=True)  # Reverse for horizontal bar
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create gradient colors from purple (high) to teal (low)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_20_df)))
    
    bars = ax.barh(
        range(len(top_20_df)),
        top_20_df["mean_abs_shap"].values,
        color=colors,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Feature labels are already aggregated to original-variable combos (with " x " separator).
    feature_labels = [str(f) for f in top_20_df["feature"].values]
    
    ax.set_yticks(range(len(top_20_df)))
    ax.set_yticklabels(feature_labels, fontsize=9)
    ax.set_xlabel("Aggregated Importance (Mean |SHAP|)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Feature Combination", fontsize=11, fontweight='bold')
    ax.set_title(f"Top {len(top_20_df)} Aggregated Interaction Features", fontsize=13, fontweight='bold', pad=15)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(top_20_df.iterrows()):
        ax.text(
            row["mean_abs_shap"] + max(top_20_df["mean_abs_shap"]) * 0.01,
            i,
            f'{row["mean_abs_shap"]:.4f}',
            va='center',
            fontsize=8
        )
    
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(left=0)
    plt.tight_layout()
    
    top20_bar_path = os.path.join(out_dir, f"shap_{tag}_top20_horizontal_bar.png")
    plt.savefig(top20_bar_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[SHAP] Saved top {len(top_20_df)} horizontal bar chart to: {top20_bar_path}")

    # --- summary plots ---
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample_sel,
        feature_names=feature_names_sel,
        max_display=max_display,
        show=False,
    )
    beeswarm_path = os.path.join(out_dir, f"shap_{tag}_summary_beeswarm.png")
    plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Saved beeswarm summary plot to: {beeswarm_path}")

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample_sel,
        feature_names=feature_names_sel,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    bar_path = os.path.join(out_dir, f"shap_{tag}_summary_bar.png")
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Saved bar summary plot to: {bar_path}")

    # --- dependence plots for the top-k features ---
    top_k = min(5, shap_imp_df.shape[0])
    for rank in range(top_k):
        feat = shap_imp_df.loc[rank, "feature"]
        feat_idx = int(np.where(feature_names_sel == feat)[0][0])
        plt.figure()
        shap.dependence_plot(
            feat_idx,
            shap_values,
            X_sample_sel,
            feature_names=feature_names_sel,
            show=False,
        )
        dep_path = os.path.join(out_dir, f"shap_{tag}_dependence_top{rank+1}.png")
        plt.savefig(dep_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] Saved dependence plot to: {dep_path}")


_SHAP_OUT_DIR = os.path.expanduser("~/work/vaping_project_data/shap_lasso_interactions")
_fit_shap_and_save_plots_for_interaction_lasso(
    pipe=best_model_2way_interactions,
    X_train=X_train_with_indicators,
    out_dir=_SHAP_OUT_DIR,
    tag="lasso_interactions_degree2",
)



# %%
# ## Random Forest Classifier

# %%
# Define Random State for reproducibility
RANDOM_STATE = 42

# Use StratifiedKFold for validation (faster than RepeatedStratifiedKFold)
N_SPLITS_CV = 5
SCORING_METRIC = 'roc_auc'
VERBOSE = 1

logging.info("\n--- Random Forest (Optuna-Optimized) ---")

# ==============================================================================
# OPTUNA-BASED HYPERPARAMETER OPTIMIZATION
# ==============================================================================
# Benefits over RandomizedSearchCV:
# 1. Bayesian optimization learns from previous trials → smarter search
# 2. Pruning stops unpromising trials early → saves compute time
# 3. Tighter, empirically-grounded hyperparameter ranges → faster convergence

def rf_optuna_objective(trial):
    """Optuna objective function for Random Forest optimization."""
    print(f"  Starting trial {trial.number}...", flush=True)
    
    # Optimized hyperparameter ranges based on empirical best practices
    params = {
        # n_estimators: 100-400 covers most gains; beyond 400 has diminishing returns
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        # max_depth: 5-20 balances complexity and generalization; None can overfit
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        # min_samples_split: 2-20 is sufficient for most datasets
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        # min_samples_leaf: 1-10 prevents overly specific leaves
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        # max_features: sqrt and log2 are most common; include a few float options
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.7]),
        # bootstrap: True enables OOB scoring and is generally preferred
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        # class_weight: handle imbalanced classes
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced', 'balanced_subsample']),
        # min_impurity_decrease: small regularization to prevent overfitting
        'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.01),
    }
    
    # Build pipeline with trial parameters
    rf_clf = RandomForestClassifier(
        **params,
        random_state=RANDOM_STATE,
        n_jobs=-1,  # Use ALL cores for tree building (most expensive part)
        oob_score=params['bootstrap'],  # Use OOB score when bootstrap=True
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', rf_clf)
    ])
    
    # Cross-validation with StratifiedKFold (faster than RepeatedStratifiedKFold)
    cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    
    try:
        # n_jobs=1 for CV folds (sequential) since RF already uses all cores
        scores = cross_val_score(
            pipeline, X_train_with_indicators, y_train,
            cv=cv, scoring=SCORING_METRIC, n_jobs=1
        )
        return scores.mean()
    except Exception as e:
        logging.warning(f"Trial failed with params {params}: {e}")
        return 0.0  # Return low score for failed trials


try:
    print("Starting Optuna optimization for Random Forest...")
    logging.info("Starting Optuna optimization for Random Forest...")
    
    # Enable Optuna's verbose logging so you can see progress
    optuna.logging.set_verbosity(optuna.logging.INFO)
    
    # Configure Optuna study with TPE sampler (Tree-structured Parzen Estimator)
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    
    # MedianPruner stops unpromising trials early based on intermediate values
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    
    study = optuna.create_study(
        direction='maximize',  # Maximize ROC-AUC
        sampler=sampler,
        pruner=pruner,
        study_name='rf_optimization'
    )
    
    # Number of trials: use same budget as _N_ITER["rf"] for fair comparison
    n_trials = _N_ITER["rf"]
    print(f"Running {n_trials} optimization trials...")
    
    # Callback to print progress after each trial
    def print_callback(study, trial):
        print(f"Trial {trial.number}: ROC-AUC = {trial.value:.4f} | Best so far: {study.best_value:.4f}")
    
    # Run optimization with timeout safeguard (optional: 30 min max)
    study.optimize(
        rf_optuna_objective,
        n_trials=n_trials,
        timeout=1800,  # 30 minutes max
        n_jobs=1,  # Sequential trials (CV already parallelized)
        show_progress_bar=True,
        gc_after_trial=True,  # Free memory after each trial
        callbacks=[print_callback],  # Print after each trial
    )
    
    logging.info(f"Best trial: {study.best_trial.number}")
    logging.info(f"Best cross-validation {SCORING_METRIC}: {study.best_value:.4f}")
    logging.info(f"Best parameters (RF): {study.best_params}")
    
    # Build the best model with optimal parameters
    best_params = study.best_params
    best_rf_clf = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        max_features=best_params['max_features'],
        bootstrap=best_params['bootstrap'],
        class_weight=best_params['class_weight'],
        min_impurity_decrease=best_params['min_impurity_decrease'],
        random_state=RANDOM_STATE,
        n_jobs=-1,  # Use all cores for final model
        oob_score=best_params['bootstrap'],
    )
    
    best_rf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', best_rf_clf)
    ])

except Exception as e:
    logging.error(f"An error occurred during Random Forest Optuna optimization: {e}")
    raise

# Evaluate the best Random Forest
try:
    logging.info("Fitting best Random Forest model on full training data...")
    best_rf.fit(X_train_with_indicators, y_train)
    
    y_pred_rf = best_rf.predict(X_test_with_indicators)
    y_pred_proba_rf = best_rf.predict_proba(X_test_with_indicators)[:, 1]

    logging.info("=== Best Random Forest Evaluation ===")
    logging.info("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred_rf)))
    logging.info("\nClassification Report:\n" + str(classification_report(y_test, y_pred_rf)))
    logging.info(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba_rf):.4f}")
    
    # Log OOB score if available
    if hasattr(best_rf.named_steps['classifier'], 'oob_score_') and best_rf.named_steps['classifier'].oob_score_:
        logging.info(f"OOB Score: {best_rf.named_steps['classifier'].oob_score_:.4f}")

    # Plot ROC
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_rf, tpr_rf, label=f'AUC = {roc_auc_score(y_test, y_pred_proba_rf):.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Random Forest ROC Curve on Test Data')
    plt.legend(loc='lower right')
    plt.show()
    
    # Plot Optuna optimization history
    try:
        fig_history = optuna.visualization.plot_optimization_history(study)
        fig_history.show()
        
        fig_importance = optuna.visualization.plot_param_importances(study)
        fig_importance.show()
    except Exception as viz_e:
        logging.warning(f"Could not generate Optuna visualizations: {viz_e}")

except Exception as e:
    logging.error(f"An error occurred during Random Forest training/evaluation: {e}")
    raise

logging.info("Random Forest optimization completed successfully.")

# %%
# Define the model file path
model_filename = os.path.expanduser('~/work/vaping_project_data/best_rf_model.joblib')

# Save the trained model
joblib.dump(best_rf, model_filename)
logging.info(f"Model saved to {model_filename}")

# %%
# Load the model from the specified path
logging.info("Loading the model...")
best_rf = joblib.load(os.path.expanduser('~/work/vaping_project_data/best_rf_model.joblib'))
logging.info("Model loaded successfull")

# %%
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import logging
import math

# Filter features: remove "respondent_sex" (case-insensitive) and any feature starting with "missing_"
aggregated_feature_names = [
    feat for feat in X_train_with_indicators.columns.tolist()
    if feat.lower() != 'respondent_sex' and not feat.lower().startswith("missing_")
]

# Print and log the features being plotted along with their count
print("Features being plotted:", aggregated_feature_names)
print("Total number of features plotted:", len(aggregated_feature_names))
logging.info(f"Features being plotted: {aggregated_feature_names}")
logging.info(f"Total number of features plotted: {len(aggregated_feature_names)}")

# Plot partial dependence curves in a grid instead of one figure per feature
n_features = len(aggregated_feature_names)
n_cols = 3
n_rows = math.ceil(n_features / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), squeeze=False)

for ax, feature in zip(axes.flat, aggregated_feature_names):
    PartialDependenceDisplay.from_estimator(best_rf, X_train_with_indicators, [feature], ax=ax)
    ax.grid(False)  # Disable grid lines on the plot
    ax.set_title(f'Partial Dependence of {feature}')

# Hide any unused subplots
for ax in axes.flat[n_features:]:
    ax.set_visible(False)

plt.tight_layout()
plt.show()

# %%
try:
    logging.info("Starting feature importance analysis...")

    # Access the RandomForestClassifier from the pipeline
    rf_model = best_rf.named_steps['classifier']

    # Get feature importances
    feature_importance = rf_model.feature_importances_

    # Access the preprocessor step
    preprocessor = best_rf.named_steps['preprocessor']

    # Get transformed feature names
    if hasattr(preprocessor, 'get_feature_names_out'):
        feature_names = preprocessor.get_feature_names_out()
    else:
        # Fallback: Generate feature names if get_feature_names_out is not available
        X_train_transformed = preprocessor.transform(X_train_with_indicators)
        feature_names = [f"Feature_{idx}" for idx in range(X_train_transformed.shape[1])]

    # Debugging: Print shapes and lengths
    logging.info(f"Shape of X_train_with_indicators: {X_train_with_indicators.shape}")
    logging.info(f"Length of feature_importance: {len(feature_importance)}")
    logging.info(f"Number of feature names: {len(feature_names)}")
    logging.info(f"Feature names: {feature_names}")

    # Check if lengths match
    if len(feature_names) != len(feature_importance):
        raise ValueError(
            f"Mismatch in lengths: feature_names ({len(feature_names)}) != feature_importance ({len(feature_importance)})"
        )

    # Create a DataFrame for visualization
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })

    # Aggregate importances for original features
    original_feature_importance = {}

    for feature, importance in zip(feature_names, feature_importance):
        # Extract the original feature name (e.g., 'cat__school_region_2' -> 'school_region')
        original_feature = feature.split('__')[1].rsplit('_', 1)[0]

        # Sum importances for each original feature
        if original_feature in original_feature_importance:
            original_feature_importance[original_feature] += importance
        else:
            original_feature_importance[original_feature] = importance

    # Create a DataFrame for aggregated importances
    aggregated_importance_df = pd.DataFrame({
        'Feature': list(original_feature_importance.keys()),
        'Importance': list(original_feature_importance.values())
    })

    # Sort features by importance
    aggregated_importance_df = aggregated_importance_df.sort_values(by='Importance', ascending=False)

    # Plot aggregated feature importance
    plt.figure(figsize=(20, 12))
    sns.barplot(x='Importance', y='Feature', data=aggregated_importance_df, palette='viridis')
    plt.title('Aggregated Feature Importance (Original Features)')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

    # Display top 20 feature importances
    top_20_features = aggregated_importance_df.head(20)
    print("Top 20 Feature Importances:")
    print(top_20_features)

except Exception as e:
    logging.error(f"An error occurred during feature importance analysis: {e}")
    raise

# %%
# best_rf is the best estimator from your RandomizedSearchCV
tree_model = best_rf.named_steps['classifier']

# Transform the entire training set
X_train_processed_full = best_rf.named_steps['preprocessor'].transform(X_train_with_indicators)

# Convert sparse matrix to dense array if needed
from scipy import sparse
if sparse.issparse(X_train_processed_full):
    X_train_processed_full = X_train_processed_full.toarray()

# Convert to DataFrame for easier sampling & feature naming
feature_names = best_rf.named_steps['preprocessor'].get_feature_names_out()
X_train_processed_df = pd.DataFrame(X_train_processed_full, columns=feature_names)

# Randomly sample 5000 rows from the processed data
X_background = X_train_processed_df.sample(n=1000, random_state=42)

# Create the explainer on just the 5000 background points
explainer = shap.TreeExplainer(tree_model, data=X_background)

# If you also want to compute shap values for the same subset (typical):
shap_values = explainer.shap_values(X_background)

# Handle different SHAP output formats
# For binary classification, shap_values can be:
#   - A list of 2 arrays [class_0, class_1], each (n_samples, n_features)
#   - A single array (n_samples, n_features) for the positive class
#   - An array (n_samples, n_features, n_classes)
if isinstance(shap_values, list):
    # Old format: list of arrays per class
    shap_values_class1 = shap_values[1]
elif len(shap_values.shape) == 3:
    # Shape (n_samples, n_features, n_classes)
    shap_values_class1 = shap_values[:, :, 1]
else:
    # Single array for positive class (n_samples, n_features)
    shap_values_class1 = shap_values

print(f"SHAP values shape: {shap_values_class1.shape}")
print(f"Number of features: {len(feature_names)}")


# %%
# 1. Map processed feature names back to original feature names
def map_to_original_feature(processed_name, original_names):
    """
    Map a processed feature name to its original feature name.
    - 'num__feature_name' -> 'feature_name'
    - 'cat__feature_name_value' -> 'feature_name'
    - 'remainder__feature_name' -> 'feature_name'
    """
    # Remove prefix
    clean_name = processed_name
    for prefix in ['num__', 'cat__', 'remainder__']:
        if clean_name.startswith(prefix):
            clean_name = clean_name[len(prefix):]
            break
    
    # For numeric features, the clean name should match directly
    if clean_name in original_names:
        return clean_name
    
    # For one-hot encoded categorical features, find the matching original feature
    # The format is 'original_feature_category_value'
    for orig_name in original_names:
        if clean_name.startswith(orig_name + '_'):
            return orig_name
    
    # Fallback: return the clean name
    return clean_name

# Get the original column names from the training data
original_feature_names = list(X_train_with_indicators.columns)

# Create mapping from processed feature index to original feature name
feature_to_original = {}
for i, col in enumerate(feature_names):
    original_name = map_to_original_feature(col, original_feature_names)
    feature_to_original[i] = original_name

# 2. Aggregate SHAP values by original feature (sum values for each sample)
unique_original_features = list(dict.fromkeys(feature_to_original.values()))
original_shap_values = np.zeros((shap_values_class1.shape[0], len(unique_original_features)))

for i, orig_feat in enumerate(unique_original_features):
    # Find all processed feature indices that map to this original feature
    indices = [idx for idx, name in feature_to_original.items() if name == orig_feat]
    # Sum the SHAP values (preserves direction for interpretability)
    original_shap_values[:, i] = shap_values_class1[:, indices].sum(axis=1)

# 3. Calculate mean absolute SHAP value for each original feature
mean_abs_shap = np.abs(original_shap_values).mean(axis=0)
aggregated_importances = dict(zip(unique_original_features, mean_abs_shap))

# 4. Sort features by importance
sorted_importances = sorted(
    aggregated_importances.items(), key=lambda item: item[1], reverse=True
)

# 5. Create a DataFrame for plotting
importance_df = pd.DataFrame(sorted_importances, columns=['Feature', 'Importance'])

# Filter to show only the top 20 features
top_20_importance_df = importance_df.head(20)

# %%
importance_df.head(20)

# %%
# 5. Create the bar plot
plt.figure(figsize=(12, 8))  # Adjust size as needed
plt.barh(top_20_importance_df['Feature'], top_20_importance_df['Importance'], color='dodgerblue')
plt.xlabel('Mean Absolute SHAP Value')
plt.ylabel('Feature')
plt.title('Top 20 Features Ranked by Mean Absolute SHAP Value')
plt.gca().invert_yaxis()  # Most important feature on top
plt.tight_layout()
plt.show()

# %%
# Partial Dependence Plots for Top 20 SHAP Features
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import numpy as np

# Get top 20 SHAP features from importance_df (already sorted by SHAP importance)
top_20_shap_features = importance_df.head(20)['Feature'].tolist()

print(f"Top 20 SHAP features for PDP:")
for i, feat in enumerate(top_20_shap_features, 1):
    shap_val = importance_df[importance_df['Feature'] == feat]['Importance'].values[0]
    print(f"  {i:2d}. {feat} (SHAP: {shap_val:.4f})")

# Filter to features that exist in X_train_with_indicators
valid_features = [f for f in top_20_shap_features if f in X_train_with_indicators.columns]
print(f"\nValid features for PDP: {len(valid_features)} / {len(top_20_shap_features)}")

if len(valid_features) == 0:
    print("ERROR: No valid features found for PDP plots!")
else:
    # Calculate grid dimensions
    n_features = len(valid_features)
    n_cols = 4  # Number of columns in the grid
    n_rows = int(np.ceil(n_features / n_cols))
    
    print(f"\nGenerating PDP grid: {n_rows} rows x {n_cols} cols for {n_features} features...")
    
    # Create all PDPs at once using sklearn's built-in grid
    display = PartialDependenceDisplay.from_estimator(
        best_rf,
        X_train_with_indicators,
        features=valid_features,
        n_cols=n_cols,
        random_state=42,
        n_jobs=-1  # Use all available cores for faster computation
    )
    
    # Resize the figure
    display.figure_.set_size_inches(20, 4 * n_rows)
    
    # Update titles to include SHAP rank
    for idx, feat in enumerate(valid_features):
        if idx < len(display.axes_.flatten()):
            ax = display.axes_.flatten()[idx]
            shap_val = importance_df[importance_df['Feature'] == feat]['Importance'].values[0]
            ax.set_title(f"#{idx+1}: {feat}\n(SHAP: {shap_val:.4f})", fontsize=10, pad=8)
    
    display.figure_.suptitle("Partial Dependence Plots - Top 20 SHAP Features", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()

# %%
# ## Gradient Boosting Trees

# %%
import logging
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import optuna

# Random State for reproducibility
RANDOM_STATE = 42

# CV configuration - StratifiedKFold is faster than RepeatedStratifiedKFold
N_SPLITS_CV = 5
SCORING_METRIC = 'roc_auc'

logging.info("\n--- Gradient Boosting (Optuna-Optimized) ---")

# ==============================================================================
# OPTUNA-BASED HYPERPARAMETER OPTIMIZATION
# ==============================================================================
# Advantages over RandomizedSearchCV:
# 1. Bayesian optimization (TPE) learns from previous trials → smarter search
# 2. Pruning stops unpromising trials early → saves compute time
# 3. Early stopping in GBT prevents overfitting and reduces training time
# 4. Tighter, empirically-grounded hyperparameter ranges → faster convergence

def gbc_optuna_objective(trial):
    """Optuna objective function for Gradient Boosting optimization."""
    print(f"  Starting trial {trial.number}...", flush=True)
    
    # Optimized hyperparameter ranges based on empirical best practices for GBT
    # Key insight: use high n_estimators with early stopping instead of random sampling
    params = {
        # n_estimators: set high, rely on early stopping to find optimal count
        # This is faster than searching the full range since early stopping kicks in
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        # learning_rate: log scale for better coverage; lower rates need more trees
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        # max_depth: 3-6 is the sweet spot for GBT (deeper often overfits)
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        # subsample: stochastic gradient boosting helps regularization
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        # min_samples_split: moderate values prevent overfitting
        'min_samples_split': trial.suggest_int('min_samples_split', 5, 30),
        # min_samples_leaf: 1-15 is sufficient for most cases
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),
        # max_features: sqrt is often optimal, include a few alternatives
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.8, None]),
    }
    
    # Build classifier with early stopping enabled
    # n_iter_no_change + validation_fraction enables early stopping
    gbc_clf = GradientBoostingClassifier(
        **params,
        random_state=RANDOM_STATE,
        # Early stopping: stop if validation score doesn't improve for 15 rounds
        n_iter_no_change=15,
        validation_fraction=0.15,  # Use 15% of training data for early stopping validation
        tol=1e-4,  # Minimum improvement threshold
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', gbc_clf)
    ])
    
    # Cross-validation with StratifiedKFold (faster than RepeatedStratifiedKFold)
    cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    
    try:
        # Use n_jobs=1 to avoid nested parallelism when Optuna is already parallelizing
        # Optuna handles parallelization at the trial level, so each trial should use single-threaded CV
        scores = cross_val_score(
            pipeline, X_train_with_indicators, y_train,
            cv=cv, scoring=SCORING_METRIC, n_jobs=1
        )
        return scores.mean()
    except Exception as e:
        logging.warning(f"Trial failed with params {params}: {e}")
        return 0.0  # Return low score for failed trials


try:
    print("Starting Optuna optimization for Gradient Boosting...")
    logging.info("Starting Optuna optimization for Gradient Boosting...")
    
    # Enable Optuna's verbose logging
    optuna.logging.set_verbosity(optuna.logging.INFO)
    
    # TPE sampler: Tree-structured Parzen Estimator for Bayesian optimization
    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE,
        n_startup_trials=10,  # Random exploration before TPE kicks in
    )
    
    # MedianPruner: stops unpromising trials early based on intermediate values
    # n_startup_trials: wait for some trials before pruning
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    
    study_gbc = optuna.create_study(
        direction='maximize',  # Maximize ROC-AUC
        sampler=sampler,
        pruner=pruner,
        study_name='gbc_optimization'
    )
    
    # Number of trials: use same budget as _N_ITER["gbt"] for fair comparison
    n_trials = _N_ITER["gbt"]
    print(f"Running {n_trials} optimization trials...")
    
    # Callback to print progress after each trial
    def print_callback(study, trial):
        print(f"Trial {trial.number}: ROC-AUC = {trial.value:.4f} | Best so far: {study.best_value:.4f}")
    
    # Run optimization with timeout safeguard (45 min max for GBT)
    study_gbc.optimize(
        gbc_optuna_objective,
        n_trials=n_trials,
        timeout=2700,  # 45 minutes max
        n_jobs=-1, 
        show_progress_bar=True,
        gc_after_trial=True,  # Free memory after each trial
        callbacks=[print_callback],
    )
    
    logging.info(f"Best trial: {study_gbc.best_trial.number}")
    logging.info(f"Best cross-validation {SCORING_METRIC}: {study_gbc.best_value:.4f}")
    logging.info(f"Best parameters (GBC): {study_gbc.best_params}")
    
    # Build the best model with optimal parameters (without early stopping for final fit)
    best_params = study_gbc.best_params
    best_gbc_clf = GradientBoostingClassifier(
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        subsample=best_params['subsample'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        max_features=best_params['max_features'],
        random_state=RANDOM_STATE,
        # Keep early stopping for final model to prevent overfitting
        n_iter_no_change=15,
        validation_fraction=0.1,
        tol=1e-4,
    )
    
    best_gbc = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', best_gbc_clf)
    ])

except Exception as e:
    logging.error(f"An error occurred during Gradient Boosting Optuna optimization: {e}")
    raise

# Evaluate the best Gradient Boosting model
try:
    logging.info("Fitting best Gradient Boosting model on full training data...")
    best_gbc.fit(X_train_with_indicators, y_train)
    
    # Log actual number of estimators used (may be less due to early stopping)
    actual_n_estimators = best_gbc.named_steps['classifier'].n_estimators_
    logging.info(f"Actual estimators used (after early stopping): {actual_n_estimators}")
    
    y_pred_gbc = best_gbc.predict(X_test_with_indicators)
    y_pred_proba_gbc = best_gbc.predict_proba(X_test_with_indicators)[:, 1]

    logging.info("=== Best Gradient Boosting Evaluation ===")
    logging.info("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred_gbc)))
    logging.info("\nClassification Report:\n" + str(classification_report(y_test, y_pred_gbc)))
    logging.info(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba_gbc):.4f}")

    # Plot ROC Curve
    fpr_gbc, tpr_gbc, _ = roc_curve(y_test, y_pred_proba_gbc)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_gbc, tpr_gbc, label=f'AUC = {roc_auc_score(y_test, y_pred_proba_gbc):.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Gradient Boosting ROC Curve on Test Data')
    plt.legend(loc='lower right')
    plt.show()
    
    # Plot Optuna optimization history and parameter importance
    try:
        fig_history = optuna.visualization.plot_optimization_history(study_gbc)
        fig_history.show()
        
        fig_importance = optuna.visualization.plot_param_importances(study_gbc)
        fig_importance.show()
    except Exception as viz_e:
        logging.warning(f"Could not generate Optuna visualizations: {viz_e}")

except Exception as e:
    logging.error(f"An error occurred during Gradient Boosting training/evaluation: {e}")
    raise

logging.info("Gradient Boosting optimization completed successfully.")

# %%
# Define the model file path
model_filename = os.path.expanduser('~/work/vaping_project_data/best_gbt_model.joblib')

# Save the trained model
joblib.dump(best_gbc, model_filename)
logging.info(f"Model saved to {model_filename}")

# %%
# Load the model (when needed)
file_path = os.path.expanduser('~/work/vaping_project_data/best_gbt_model.joblib')
loaded_gbt = joblib.load(file_path)
print("Model loaded successfully.")

# %%
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import logging

# Filter features: remove "respondent_sex" (case-insensitive) and any feature starting with "missing_"
aggregated_feature_names = [
    feat for feat in X_train_with_indicators.columns.tolist() 
    if feat.lower() != 'respondent_sex' and not feat.lower().startswith("missing_")
]

# Print and log the features being plotted along with their count
print("Features being plotted:", aggregated_feature_names)
print("Total number of features plotted:", len(aggregated_feature_names))
logging.info(f"Features being plotted: {aggregated_feature_names}")
logging.info(f"Total number of features plotted: {len(aggregated_feature_names)}")

# Loop through each feature and plot its partial dependence on a separate figure
for feature in aggregated_feature_names:
    fig, ax = plt.subplots(figsize=(8, 6))
    PartialDependenceDisplay.from_estimator(loaded_gbt, X_train_with_indicators, [feature], ax=ax)
    ax.grid(False)  # Disable grid lines on the plot
    ax.set_title(f'Partial Dependence of {feature}')
    plt.tight_layout()
    plt.show()

# %%
from sklearn.inspection import permutation_importance

# Calculate permutation importance on RAW DATA (let pipeline handle preprocessing)
result = permutation_importance(
    loaded_gbt,  # This is your full pipeline
    X_test_with_indicators,  # Raw data with missing indicators
    y_test,
    n_repeats=5,
    random_state=RANDOM_STATE,
    n_jobs=CPU_COUNT
)

# Get feature names from the raw data (including missing indicators)
feature_names = X_test_with_indicators.columns.tolist()

# Create importance DataFrame
perm_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': result.importances_mean
}).sort_values('Importance', ascending=False)

# Plot top 20
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=perm_importance.head(20))
plt.title("Top 20 Features by Permutation Importance (Raw Features)")
plt.show()

# %%
# Create a table for top 20 feature importances
top_20_features = perm_importance.head(20)

# Display the table
print("Top 20 Feature Importances:")
display(top_20_features.style.background_gradient(cmap='Blues', subset=['Importance']))

# %%
# Access the pipeline for numeric features
num_pipeline = loaded_gbt.named_steps['preprocessor'].named_transformers_['num']
# Get all feature names from the preprocessor (which is fitted)
all_feature_names = loaded_gbt.named_steps['preprocessor'].get_feature_names_out()
# Get the number of numeric features from the scaler (if fitted) or use numeric_features
scaler = num_pipeline.named_steps['scaler']
if hasattr(scaler, 'n_features_in_'):
    n_numeric = scaler.n_features_in_
    encoded_feature_names = all_feature_names[:n_numeric]
elif 'numeric_features' in locals() or 'numeric_features' in globals():
    # Use numeric_features directly since StandardScaler doesn't change feature names
    encoded_feature_names = numeric_features
else:
    # Last resort: try to get from numeric pipeline
    try:
        encoded_feature_names = num_pipeline.get_feature_names_out()
    except (NotFittedError, AttributeError):
        # If all else fails, use all feature names (not ideal but won't crash)
        encoded_feature_names = all_feature_names
encoded_feature_names

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Get the numerical pipeline
num_pipeline = loaded_gbt.named_steps['preprocessor'].named_transformers_['num']

# 2. Get the feature names from the preprocessor (which is fitted)
all_feature_names = loaded_gbt.named_steps['preprocessor'].get_feature_names_out()

# 3. Get numeric feature names - they come first in the preprocessor output
# Get the number of numeric features from the scaler (if fitted) or use numeric_features
scaler = num_pipeline.named_steps['scaler']
if hasattr(scaler, 'n_features_in_'):
    n_numeric = scaler.n_features_in_
    encoded_feature_names = all_feature_names[:n_numeric]
elif 'numeric_features' in locals() or 'numeric_features' in globals():
    # Use numeric_features directly since StandardScaler doesn't change feature names
    encoded_feature_names = numeric_features
else:
    # Last resort: try to get from numeric pipeline
    try:
        encoded_feature_names = num_pipeline.get_feature_names_out()
    except (NotFittedError, AttributeError):
        # If all else fails, use all feature names (not ideal but won't crash)
        encoded_feature_names = all_feature_names

# 4. Get the trained classifier and its feature importances
gbt_classifier = loaded_gbt.named_steps['classifier']
importances = gbt_classifier.feature_importances_

# 5. Build a DataFrame of features vs. importances
feature_importance_df = pd.DataFrame({
    'Feature': encoded_feature_names,
    'Importance': importances
})

# 6. Sort by ascending order of importance
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=True)

# 7. Take top 20 features and plot
top_20 = feature_importance_df.tail(20)

plt.figure(figsize=(12, 8))
plt.barh(y=top_20['Feature'], width=top_20['Importance'])
plt.title('Top 20 Most Important Original Features')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# 8. Print the DataFrame in descending order for readability
print("\nFeature Importance Rankings:")
print(feature_importance_df.sort_values('Importance', ascending=False))

# %%
# 1) Sort by descending importance and take top 20.
feature_importance_df = perm_importance.sort_values('Importance', ascending=False)
top_20_features = feature_importance_df['Feature'].head(20).tolist()

# 2) Exclude features that start with "missing_", that don't exist in X_train_with_indicators, or that are "respondent_sex"
filtered_features = [
    f for f in top_20_features 
    if not f.startswith('missing_') and f.lower() != 'respondent_sex' and f in X_train_with_indicators.columns
]

print("Top 20 original features:", top_20_features)
print("Filtered features (excluding 'missing_' and 'respondent_sex'):", filtered_features)

# 3) Plot PDPs for the filtered features in separate figures.
for feat in filtered_features:
    fig, ax = plt.subplots(figsize=(8, 6))
    PartialDependenceDisplay.from_estimator(
        estimator=loaded_gbt,
        X=X_train_with_indicators,
        features=[feat],
        feature_names=X_train_with_indicators.columns,
        kind='average',  # or 'both' if you want ICE lines as well
        ax=ax
    )
    ax.set_title(f"PDP for {feat}")
    plt.tight_layout()
    plt.show()

# %%
X_train_transformed = preprocessor.transform(X_train_with_indicators)
print("Shape of X_train_transformed:", X_train_transformed.shape)
print("Dtypes (if it is a NumPy array):", X_train_transformed.dtype)

# If X_train_transformed is a Pandas DataFrame:
if hasattr(X_train_transformed, 'dtypes'):
    print(X_train_transformed.dtypes)

X_train_with_indicators.info()  # or X_train_with_indicators.isna().sum() if DataFrame

# %%
##### SHAP Feature Importance ####

# Extract the GradientBoostingClassifier
gbt_models = loaded_gbt.named_steps['classifier']
# Get preprocessed features
X_preprocessed = loaded_gbt.named_steps['preprocessor'].transform(X_test_with_indicators)
# Create SHAP explainer
explainer = shap.TreeExplainer(gbt_models)
# Calculate SHAP values
shap_values = explainer.shap_values(X_preprocessed)
# Get feature names after preprocessing
feature_names = loaded_gbt.named_steps['preprocessor'].get_feature_names_out()

# %%
# Create visualizations
# Summary plot
shap.summary_plot(shap_values, X_preprocessed, feature_names=feature_names)
# Bar plot of feature importance
shap.summary_plot(shap_values, X_preprocessed, feature_names=feature_names, plot_type='bar')

# %%
# 1. Aggregate SHAP values by base feature
feature_importances = {}
for i, col in enumerate(feature_names):
    base_feature = col.replace('num__', '')  # Extract base feature name
    if base_feature not in feature_importances:
        feature_importances[base_feature] = []
    feature_importances[base_feature].extend(np.abs(shap_values[:, i]))

# 2. Calculate mean absolute SHAP value for each base feature
aggregated_importances = {
    feature: np.mean(values) for feature, values in feature_importances.items()
}

# 3. Sort features by importance
sorted_importances = sorted(
    aggregated_importances.items(), key=lambda item: item[1], reverse=True
)

# 4. Create a DataFrame for plotting
importance_df = pd.DataFrame(sorted_importances, columns=['Feature', 'Importance'])

# Filter to show only the top 20 features
top_20_importance_df = importance_df.head(20)

# %%
importance_df.head(20)

# %%
# 5. Create the bar plot
plt.figure(figsize=(12, 8))  # Adjust size as needed
plt.barh(top_20_importance_df['Feature'], top_20_importance_df['Importance'], color='dodgerblue')
plt.xlabel('Mean Absolute SHAP Value')
plt.ylabel('Feature')
plt.title('Top 20 Features Ranked by Mean Absolute SHAP Value')
plt.gca().invert_yaxis()  # Most important feature on top
plt.tight_layout()
plt.show()

# %%
# For individual predictions (e.g., first sample)
shap.initjs()  # Initialize JavaScript visualization
single_sample_idx = 0
shap.force_plot(explainer.expected_value[1] if isinstance(shap_values, list) else explainer.expected_value,
                shap_values[single_sample_idx] if isinstance(shap_values, list) else shap_values[single_sample_idx,:],
                X_preprocessed[single_sample_idx],
                feature_names=feature_names)

# %%
###### Demonstration of SHAP feature importance for one individual case ###########

# For individual predictions (e.g., first sample)
single_sample_idx = 0
single_sample_shap_values = shap_values[single_sample_idx] if isinstance(shap_values, list) else shap_values[single_sample_idx, :]

# Aggregate SHAP values by base feature for the single sample
feature_importances = {}
for i, feature_name in enumerate(feature_names):
    base_feature = col.replace('num__', '')
  # Extract base feature name
    shap_value = single_sample_shap_values[i]
    feature_importances[base_feature] = feature_importances.get(base_feature, 0) + abs(shap_value)

# Sort features by importance
sorted_importances = sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)

# Print the ranked feature importance for the single prediction
print(f"Overall Feature Importance for Sample {single_sample_idx}:")
for feature, importance in sorted_importances:
    print(f"{feature}: {importance:.4f}")

# %%
# SHAP feature interaction

# Calculate SHAP interaction values
shap_interaction_values = explainer.shap_interaction_values(X_preprocessed)

# %%
#  We'll sample 100 rows (or choose your own size)
SAMPLE_SIZE = 5000
if X_preprocessed.shape[0] > SAMPLE_SIZE:
    idx = np.random.choice(X_preprocessed.shape[0], size=SAMPLE_SIZE, replace=False)
    X_preprocessed_subset = X_preprocessed[idx, :]
else:
    X_preprocessed_subset = X_preprocessed

# Now compute SHAP interaction on the subset
shap_interaction_values = explainer.shap_interaction_values(X_preprocessed_subset)

# %%
import numpy as np
import pandas as pd

def aggregate_shap_interactions(shap_interaction_values, feature_names, get_base_feature):
    """
    Aggregates pairwise SHAP interaction values back to their original (pre–one-hot) features.
    
    Parameters
    ----------
    shap_interaction_values : np.ndarray
        SHAP interaction values of shape [n_samples, n_features, n_features].
    feature_names : list of str
        The one-hot-encoded feature names corresponding to shap_interaction_values.
    get_base_feature : callable
        A function that takes a one-hot-encoded feature name and returns the base/original feature name.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with ["Feature1", "Feature2", "InteractionValue", "AbsInteraction"] 
        sorted in descending order of AbsInteraction.
    """
    # 1. Aggregate across samples (e.g., mean absolute interactions)
    interaction_matrix = np.mean(np.abs(shap_interaction_values), axis=0)
    
    # 2. Map each OHE feature to a base feature
    base_feature_names = [get_base_feature(n) for n in feature_names]
    unique_base_features = list(set(base_feature_names))
    
    # 3. Build a structure to accumulate aggregated pairwise interactions
    aggregated_interactions = {
        bf_i: {bf_j: 0.0 for bf_j in unique_base_features}
        for bf_i in unique_base_features
    }

    n_features = len(feature_names)
    for i in range(n_features):
        for j in range(i+1, n_features): # i+1 => no diagonal, no duplicates
            bf_i = base_feature_names[i]
            bf_j = base_feature_names[j]
            aggregated_interactions[bf_i][bf_j] += interaction_matrix[i, j]
    
    # 4. Convert to DataFrame
    data_records = []
    for bf_i in unique_base_features:
        for bf_j in unique_base_features:
            # If you want to keep only i <= j, add a condition to avoid duplicates
            interaction_val = aggregated_interactions[bf_i][bf_j]
            data_records.append((bf_i, bf_j, interaction_val))
    
    df_interactions = pd.DataFrame(data_records, columns=["Feature1", "Feature2", "InteractionValue"])
    df_interactions["AbsInteraction"] = df_interactions["InteractionValue"].abs()
    
    # Sort descending by absolute interaction
    df_interactions.sort_values("AbsInteraction", ascending=False, inplace=True)
    df_interactions.reset_index(drop=True, inplace=True)
    df_interactions_no_diagonal = df_interactions[df_interactions['Feature1'] != df_interactions['Feature2']]
    return df_interactions_no_diagonal


# Example usage:
def simple_get_base_feature(name):
    # If it has the "num__" prefix, strip it off
    if name.startswith("num__"):
        name = name[len("num__"):]  # "times_skipped_class"
    # If it has the "cat__" prefix, strip that as well
    if name.startswith("cat__"):
        name = name[len("cat__"):] 
    # Now 'name' might look like "times_skipped_class"
    # Just return it as the base feature
    return name

df_agg_interactions = aggregate_shap_interactions(
    shap_interaction_values=shap_interaction_values,
    feature_names=feature_names,
    get_base_feature=simple_get_base_feature
)

# Print the top 20 interactions
print(df_agg_interactions.head(20))

# --- Pick Top 20 Interactions ---
df_top_20 = df_agg_interactions.head(20).copy()

# Create a convenient label for each pair
df_top_20["Pair"] = df_top_20["Feature1"] + " & " + df_top_20["Feature2"]

# --- Plot ---
plt.figure(figsize=(10, 6))
sns.barplot(
    data=df_top_20, 
    y="Pair", 
    x="AbsInteraction", 
    color="royalblue"
)
plt.title("Top 20 Pairwise Feature Interactions by Absolute SHAP Value")
plt.xlabel("Absolute SHAP Interaction Value")
plt.ylabel("Feature Pair")
plt.tight_layout()
plt.show()

# %%
# Interaction of the top 2 features
top_features_indices = np.argsort(np.abs(shap_values).mean(0))[-2:]  # Get indices of top 2 features
feature1_idx = top_features_indices[0]
feature2_idx = top_features_indices[1]
feature1_name = feature_names[feature1_idx]
feature2_name = feature_names[feature2_idx]

# %%
# Visualize the interaction between the top two features
shap.dependence_plot(
    feature1_idx,
    shap_values,
    X_preprocessed,
    feature_names=feature_names,
    interaction_index=feature2_idx,
)

shap.dependence_plot(
    feature2_idx,
    shap_values,
    X_preprocessed,
    feature_names=feature_names,
    interaction_index=feature1_idx,
)

# %%
# 1. Group feature indices by base feature (removing "num__" prefix).
base_feature_indices = {}
for i, feature_name in enumerate(feature_names):
    # Remove "num__" so each feature remains distinct
    base_feature = feature_name.replace("num__", "")
    if base_feature not in base_feature_indices:
        base_feature_indices[base_feature] = []
    base_feature_indices[base_feature].append(i)

base_feature_list = list(base_feature_indices.keys())

# 2. Calculate overall interaction importance for each pair of base features
#    without duplicating reversed pairs (bf1,bf2) vs (bf2,bf1).
base_feature_interaction_importance = {}

for bf1_idx in range(len(base_feature_list)):
    for bf2_idx in range(bf1_idx + 1, len(base_feature_list)):
        bf1 = base_feature_list[bf1_idx]
        bf2 = base_feature_list[bf2_idx]

        # Sum up the pairwise interactions between *all* sub-indices of bf1 and bf2.
        interaction_sum = 0.0
        for i in base_feature_indices[bf1]:
            for j in base_feature_indices[bf2]:
                if isinstance(shap_interaction_values, list):
                    # e.g., for multiclass or ensemble, focusing on shap_interaction_values[0]
                    interaction_sum += shap_interaction_values[0][0, i, j]
                    # If you want both i->j and j->i, add shap_interaction_values[0][0, j, i]
                    # but usually shap_interaction_values[i,j] == shap_interaction_values[j,i].
                else:
                    interaction_sum += shap_interaction_values[0, i, j]
                    # Same note here if you want both directions.

        # Use absolute value as "importance"
        pair_key = (bf1, bf2)  # We already enforce bf1_idx < bf2_idx
        base_feature_interaction_importance[pair_key] = abs(interaction_sum)

# 3. Sort base feature interactions by importance
sorted_base_feature_interactions = sorted(
    base_feature_interaction_importance.items(),
    key=lambda item: item[1],
    reverse=True
)

# 4. Print the top 50 ranked base feature interactions (no (bf, bf), no reversed duplicates)
print("Top 50 Overall Base Feature Interaction Importance (Skipping self-interactions & duplicates):")
for (bf1, bf2), importance in sorted_base_feature_interactions[:50]:
    print(f"Interaction between {bf1} and {bf2}: {importance:.4f}")

# %%
# 3. Create a matrix for the heatmap
num_base_features = len(base_feature_list)
interaction_matrix = np.zeros((num_base_features, num_base_features))

for i, bf1 in enumerate(base_feature_list):
    for j, bf2 in enumerate(base_feature_list):
        # Use the sorted tuple for lookup
        key = tuple(sorted((bf1, bf2)))
        if key in base_feature_interaction_importance:
            interaction_matrix[i, j] = base_feature_interaction_importance[key]
        #else:
        #    print(f"Warning: Interaction not found for {bf1}, {bf2}") # Optional debugging

# 4. Visualize the aggregated interaction using a heatmap
plt.figure(figsize=(30, 28))
plt.imshow(interaction_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar(label='Aggregated Interaction Strength')
plt.xticks(range(num_base_features), base_feature_list, rotation=45, ha="right")
plt.yticks(range(num_base_features), base_feature_list)
plt.title('Aggregated SHAP Interaction Between Base Features')
plt.tight_layout()
plt.show()

# %%
# Interaction Summary Plot (for overall interaction strengths)
shap.summary_plot(shap_interaction_values, X_preprocessed, feature_names=feature_names)

# %%
import numpy as np
import pandas as pd
import shap
import seaborn as sns
import matplotlib.pyplot as plt

##############################
# 1. Pairwise Aggregation
##############################
def aggregate_shap_interactions_pairwise(shap_interaction_values, feature_names, get_base_feature):
    """
    Aggregates pairwise SHAP interaction values back to their original (pre–one-hot) features,
    and returns a DataFrame of the mean absolute interaction for each (Feature1, Feature2).
    """
    # shap_interaction_values: [n_samples, n_features, n_features]
    # 1) Mean of absolute interaction across samples
    interaction_matrix = np.mean(np.abs(shap_interaction_values), axis=0)  # shape: (n_features, n_features)

    # 2) Map each OHE feature to a base feature
    base_feature_names = [get_base_feature(f) for f in feature_names]
    unique_base_features = sorted(set(base_feature_names))

    # 3) Accumulate pairwise interactions
    #    We'll sum interactions for each pair (bf_i, bf_j)
    #    Because i<j is symmetric in the shap_interaction_values, we avoid double counting.
    aggregated = {}
    for bf_i in unique_base_features:
        aggregated[bf_i] = {}
        for bf_j in unique_base_features:
            aggregated[bf_i][bf_j] = 0.0

    n_features = len(feature_names)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            bf_i = base_feature_names[i]
            bf_j = base_feature_names[j]
            aggregated[bf_i][bf_j] += interaction_matrix[i, j]
            # You could also decide to add symmetrical entries if desired:
            aggregated[bf_j][bf_i] += interaction_matrix[i, j]  # for simpler referencing

    # 4) Convert to a long DataFrame
    records = []
    for bf_i in unique_base_features:
        for bf_j in unique_base_features:
            # Avoid i == j and only keep i < j in the final output (unique pairs)
            if bf_i < bf_j:
                val = aggregated[bf_i][bf_j]
                records.append({
                    "FeatureCombo": f"{bf_i} & {bf_j}",
                    "Feature1": bf_i,
                    "Feature2": bf_j,
                    "InteractionValue": val,
                    "AbsInteraction": abs(val),
                    "Order": 2  # Mark as pairwise
                })

    df = pd.DataFrame(records)
    df.sort_values("AbsInteraction", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

##############################
# 2. "Three-way" Aggregation
##############################
def aggregate_shap_interactions_three_way(shap_interaction_values, feature_names, get_base_feature):
    """
    Naively computes a 'three-way' measure by summing the absolute pairwise interactions
    (i,j), (i,k), and (j,k) for each triple i < j < k.

    NOTE: This does not represent a true SHAP 3-way synergy. It's a proxy 
    by combining the pairwise interactions among the triple.
    """
    # 1) Mean of absolute interaction across samples
    interaction_matrix = np.mean(np.abs(shap_interaction_values), axis=0)  # shape: (n_features, n_features)

    # 2) Map each OHE feature to a base feature
    base_feature_names = [get_base_feature(f) for f in feature_names]
    unique_base_features = sorted(set(base_feature_names))

    # 3) We will group by base features for i,j,k. Because different OHE columns might map
    #    to the same base feature, let's gather all indices that map to each base feature.
    feature_indices_by_base = {}
    for idx, bf in enumerate(base_feature_names):
        if bf not in feature_indices_by_base:
            feature_indices_by_base[bf] = []
        feature_indices_by_base[bf].append(idx)

    # 4) For each triple (bf_i, bf_j, bf_k) with i < j < k, compute sum of pairwise interactions
    records = []
    bf_list = list(unique_base_features)
    n_bf = len(bf_list)

    for i in range(n_bf):
        for j in range(i + 1, n_bf):
            for k in range(j + 1, n_bf):
                bf_i, bf_j, bf_k = bf_list[i], bf_list[j], bf_list[k]

                # For all actual columns that map to bf_i, bf_j, bf_k, sum the relevant pairwise interactions:
                indices_i = feature_indices_by_base[bf_i]
                indices_j = feature_indices_by_base[bf_j]
                indices_k = feature_indices_by_base[bf_k]

                # We'll sum the absolute interaction_matrix across all pairs (i', j'), (i', k'), (j', k')
                total_interaction = 0.0
                for ii in indices_i:
                    for jj in indices_j:
                        total_interaction += interaction_matrix[ii, jj]
                        total_interaction += interaction_matrix[jj, ii]  # symmetrical

                    for kk in indices_k:
                        total_interaction += interaction_matrix[ii, kk]
                        total_interaction += interaction_matrix[kk, ii]

                for jj in indices_j:
                    for kk in indices_k:
                        total_interaction += interaction_matrix[jj, kk]
                        total_interaction += interaction_matrix[kk, jj]

                # We'll keep it as the approximate measure
                records.append({
                    "FeatureCombo": f"{bf_i} & {bf_j} & {bf_k}",
                    "Feature1": bf_i,
                    "Feature2": bf_j,
                    "Feature3": bf_k,
                    "InteractionValue": total_interaction,
                    "AbsInteraction": abs(total_interaction),
                    "Order": 3  # Mark as three-way
                })

    df = pd.DataFrame(records)
    df.sort_values("AbsInteraction", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


##############################
# 3. Putting it all together
##############################

# Suppose you already have:
#   shap_interaction_values = explainer.shap_interaction_values(X_preprocessed)
#   feature_names = loaded_gbt.named_steps['preprocessor'].get_feature_names_out()
#   We'll reuse your simple_get_base_feature function:
def simple_get_base_feature(name):
    if name.startswith("num__"):
        name = name[len("num__"):]  # e.g. "times_skipped_class"
    if name.startswith("cat__"):
        name = name[len("cat__"):]
    return name

# -- 3.1 Compute pairwise and "three-way" aggregates --
df_pairwise = aggregate_shap_interactions_pairwise(
    shap_interaction_values, 
    feature_names, 
    get_base_feature=simple_get_base_feature
)

df_three_way = aggregate_shap_interactions_three_way(
    shap_interaction_values, 
    feature_names, 
    get_base_feature=simple_get_base_feature
)

# -- 3.2 Combine them into one table --
#    For consistency, we can unify the column names
df_pairwise["Feature3"] = None  # no third feature for pairs
combined_df = pd.concat([df_pairwise, df_three_way], ignore_index=True)

# Sort by absolute interaction
combined_df.sort_values("AbsInteraction", ascending=False, inplace=True)
combined_df.reset_index(drop=True, inplace=True)

# -- 3.3 Pick top 20 overall --
df_top_20 = combined_df.head(20).copy()

# For a nicer label that works for both pairwise and triple combos:
df_top_20["Label"] = df_top_20["FeatureCombo"]

# -- 3.4 Plot a bar chart of the top 20 combos --
plt.figure(figsize=(10, 6))
sns.barplot(data=df_top_20, y="Label", x="AbsInteraction", color="royalblue")
plt.title("Top 20 Interactions (Pairwise + Three-way) by Absolute SHAP Value")
plt.xlabel("Absolute SHAP Interaction Value")
plt.ylabel("Feature Combination")
plt.tight_layout()
plt.show()

# You now have a bar plot of the top 20 interactions (including both 2- and 3-feature combos).

# %%
df_top_20

# %%
# ## Histogram-based Gradient Boost Classifier

# %%
import logging
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import FunctionTransformer

# Random State for reproducibility
RANDOM_STATE = 42

# Use RepeatedStratifiedKFold for more robust validation
N_SPLITS_CV = 5
N_REPEATS = 2  # Repeat the CV multiple times if desired
SCORING_METRIC = 'roc_auc'
VERBOSE = 1

logging.info("\n--- Gradient Boosting (Revised) ---")

# Define the transformer:
from sklearn.preprocessing import FunctionTransformer

def to_dense_func(X):
    """Convert sparse matrices to dense arrays (if needed)."""
    return X.toarray() if hasattr(X, 'toarray') else X

to_dense = FunctionTransformer(to_dense_func)

# Build pipeline
gbc_pipeline = Pipeline([
    ('preprocessor', preprocessor), 
    ('to_dense', to_dense),
    ('classifier', HistGradientBoostingClassifier(
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
    ))
])

# Expanded parameter distributions for RandomizedSearch
# Optimized parameter grid for 24-core/100GB RAM
param_grid = {
    "classifier__learning_rate": loguniform(0.01, 0.2),
    "classifier__max_depth": [None, 6, 10, 14, 18],
    "classifier__min_samples_leaf": randint(20, 200),
    "classifier__l2_regularization": loguniform(1e-4, 10.0),
    "classifier__max_bins": [255],
    "classifier__max_leaf_nodes": randint(31, 257),
    # Let early stopping pick the effective iteration count; still cap search ranges.
    "classifier__max_iter": randint(300, 2001),
}

try:
    logging.info("Starting randomized search for Gradient Boosting...")

    # Use RepeatedStratifiedKFold without shuffle
    cv_gbc = RepeatedStratifiedKFold(
        n_splits=N_SPLITS_CV, 
        n_repeats=N_REPEATS, 
        random_state=RANDOM_STATE
    )

    # RandomizedSearchCV to cover more combinations within reasonable compute time
    gbc_random_search = RandomizedSearchCV(
        estimator=gbc_pipeline,
        param_distributions=param_grid,
        n_iter=_N_ITER["hgbt"],
        cv=cv_gbc,
        scoring=SCORING_METRIC,
        n_jobs=-1,  # Use all available cores
        random_state=RANDOM_STATE,
        verbose=VERBOSE,
        return_train_score=True,
    )

    # Fit the RandomizedSearchCV
    gbc_random_search.fit(X_train_with_indicators, y_train)

    logging.info(f"Best parameters (GBC): {gbc_random_search.best_params_}")
    logging.info(f"Best cross-validation {SCORING_METRIC}: {gbc_random_search.best_score_:.4f}")

    # Extract the best estimator
    best_gbc = gbc_random_search.best_estimator_

except Exception as e:
    logging.error(f"An error occurred during Gradient Boosting randomized search: {e}")
    raise

# Evaluate the best Gradient Boosting model
try:
    best_gbc.fit(X_train_with_indicators, y_train)
    y_pred_gbc = best_gbc.predict(X_test_with_indicators)
    y_pred_proba_gbc = best_gbc.predict_proba(X_test_with_indicators)[:, 1]

    logging.info("=== Best Gradient Boosting Evaluation ===")
    logging.info("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred_gbc)))
    logging.info("\nClassification Report:\n" + str(classification_report(y_test, y_pred_gbc)))
    logging.info(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba_gbc):.4f}")

    # Plot ROC Curve
    fpr_gbc, tpr_gbc, _ = roc_curve(y_test, y_pred_proba_gbc)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_gbc, tpr_gbc, label=f'AUC = {roc_auc_score(y_test, y_pred_proba_gbc):.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Gradient Boosting ROC Curve on Test Data')
    plt.legend(loc='lower right')
    plt.show()

except Exception as e:
    logging.error(f"An error occurred during Gradient Boosting training/evaluation: {e}")
    raise

logging.info("Script completed successfully.")

# %%
# Define the model file path
model_filename = os.path.expanduser('~/work/vaping_project_data/best_hgbt_model.joblib')

# Save the trained model
joblib.dump(best_gbc, model_filename)
logging.info(f"Model saved to {model_filename}")

# %%
# Load the model (when needed)
# Define the transformer:
from sklearn.preprocessing import FunctionTransformer

def to_dense_func(X):
    """Convert sparse matrices to dense arrays (if needed)."""
    return X.toarray() if hasattr(X, 'toarray') else X

to_dense = FunctionTransformer(to_dense_func)


file_path = os.path.expanduser('~/work/vaping_project_data/best_hgbt_model.joblib')
loaded_hgbt = joblib.load(file_path)
print("Model loaded successfully.")

# %%
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import logging

# Filter features: remove "respondent_sex" (case-insensitive) and any feature starting with "missing_"
aggregated_feature_names = [
    feat for feat in X_train_with_indicators.columns.tolist() 
    if feat.lower() != 'respondent_sex' and not feat.lower().startswith("missing_")
]

# Print and log the features being plotted along with their count
print("Features being plotted:", aggregated_feature_names)
print("Total number of features plotted:", len(aggregated_feature_names))
logging.info(f"Features being plotted: {aggregated_feature_names}")
logging.info(f"Total number of features plotted: {len(aggregated_feature_names)}")

# Loop through each feature and plot its partial dependence on a separate figure
for feature in aggregated_feature_names:
    fig, ax = plt.subplots(figsize=(8, 6))
    PartialDependenceDisplay.from_estimator(loaded_hgbt, X_train_with_indicators, [feature], ax=ax)
    ax.grid(False)  # Disable grid lines on the plot
    ax.set_title(f'Partial Dependence of {feature}')
    plt.tight_layout()
    plt.show()

# %%
from sklearn.inspection import permutation_importance

# Calculate permutation importance on RAW DATA (let pipeline handle preprocessing)
result = permutation_importance(
    loaded_hgbt,  # This is your full pipeline
    X_test_with_indicators,  # Raw data with missing indicators
    y_test,
    n_repeats=5,
    random_state=RANDOM_STATE,
    n_jobs=CPU_COUNT
)

# Get feature names from the raw data (including missing indicators)
feature_names = X_test_with_indicators.columns.tolist()

# Create importance DataFrame
perm_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': result.importances_mean
}).sort_values('Importance', ascending=False)

# Plot top 20
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=perm_importance.head(20))
plt.title("Top 20 Features by Permutation Importance (Raw Features)")
plt.show()

# %%
# Create a table for top 20 feature importances
top_20_features = perm_importance.head(20)

# Display the table
print("Top 20 Feature Importances:")
display(top_20_features.style.background_gradient(cmap='Blues', subset=['Importance']))

# %%
##### SHAP Feature Importance ####

# Extract the GradientBoostingClassifier
hgbt_models = loaded_hgbt.named_steps['classifier']
# Get preprocessed features
X_preprocessed = loaded_hgbt.named_steps['preprocessor'].transform(X_test_with_indicators)
# Convert sparse matrix to DataFrame
X_preprocessed = pd.DataFrame(X_preprocessed.toarray())
# Create SHAP explainer
explainer = shap.TreeExplainer(hgbt_models)
# Calculate SHAP values
shap_values = explainer.shap_values(X_preprocessed)
# Get feature names after preprocessing
feature_names = loaded_hgbt.named_steps['preprocessor'].get_feature_names_out()

# %%
# 1. Aggregate SHAP values by base feature
feature_importances = {}
for i, col in enumerate(feature_names):
    base_feature = '_'.join(col.split('_')[:-1])  # Extract base feature name
    if base_feature not in feature_importances:
        feature_importances[base_feature] = []
    feature_importances[base_feature].extend(np.abs(shap_values[:, i]))

# 2. Calculate mean absolute SHAP value for each base feature
aggregated_importances = {
    feature: np.mean(values) for feature, values in feature_importances.items()
}

# 3. Sort features by importance
sorted_importances = sorted(
    aggregated_importances.items(), key=lambda item: item[1], reverse=True
)

# 4. Create a DataFrame for plotting
importance_df = pd.DataFrame(sorted_importances, columns=['Feature', 'Importance'])

# Filter to show only the top 20 features
top_20_importance_df = importance_df.head(20)

# 5. Create the bar plot
plt.figure(figsize=(12, 8))  # Adjust size as needed
plt.barh(top_20_importance_df['Feature'], top_20_importance_df['Importance'], color='dodgerblue')
plt.xlabel('Mean Absolute SHAP Value')
plt.ylabel('Feature')
plt.title('Top 20 Features Ranked by Mean Absolute SHAP Value')
plt.gca().invert_yaxis()  # Most important feature on top
plt.tight_layout()
plt.show()

# %%
top_20_importance_df

# %%
# 1) Sort by descending importance and take top 10 features.
feature_importance_df = top_20_importance_df.sort_values('Importance', ascending=False)
top_features = feature_importance_df['Feature'].head(20).tolist()

# 2) Remove the "cat__" prefix from each feature, if present.
mapped_features = [feat.replace("cat__", "") for feat in top_features]

print("Top original features:", top_features)
print("Mapped features (prefix removed):", mapped_features)

# 3) Filter features to include only those that exist in X_train_with_indicators 
#    and skip the "respondent_sex" variable.
filtered_features = [
    f for f in mapped_features 
    if f in X_train_with_indicators.columns and f != "respondent_sex"
]

print("Filtered features (matching X_train columns and excluding 'respondent_sex'):", filtered_features)

if len(filtered_features) == 0:
    raise ValueError("No valid features found after filtering. Please check your feature names and filtering criteria.")

# 4) Plot PDPs for each filtered feature individually,
#    skipping any that have less than two unique values.
for feat in filtered_features:
    if X_train_with_indicators[feat].nunique() < 2:
        print(f"Skipping PDP for '{feat}' due to insufficient unique values.")
        continue

    fig, ax = plt.subplots(figsize=(8, 6))
    PartialDependenceDisplay.from_estimator(
        estimator=loaded_hgbt,
        X=X_train_with_indicators,
        features=[feat],
        feature_names=X_train_with_indicators.columns,
        kind='average',  # or 'both' for ICE lines if desired.
        target=1,      # specify the target class for binary classification
        ax=ax
    )
    ax.set_title(f"PDP for {feat}")
    plt.tight_layout()
    plt.show()

# %%
import numpy as np

# Suppose your X_preprocessed is shape (n_samples, n_features).
# Randomly sample e.g. 300 rows:
sample_size = 300
if X_preprocessed.shape[0] > sample_size:
    rnd_idx = np.random.choice(X_preprocessed.shape[0], sample_size, replace=False)
    X_sampled = X_preprocessed.iloc[rnd_idx]

else:
    X_sampled = X_preprocessed

# Now compute interaction values on this smaller subset
shap_interaction_values = explainer.shap_interaction_values(X_sampled)

# %%
import numpy as np
import pandas as pd

def aggregate_shap_interactions(shap_interaction_values, feature_names, get_base_feature):
    """
    Aggregates pairwise SHAP interaction values back to their original (pre–one-hot) features.
    
    Parameters
    ----------
    shap_interaction_values : np.ndarray
        SHAP interaction values of shape [n_samples, n_features, n_features].
    feature_names : list of str
        The one-hot-encoded feature names corresponding to shap_interaction_values.
    get_base_feature : callable
        A function that takes a one-hot-encoded feature name and returns the base/original feature name.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with ["Feature1", "Feature2", "InteractionValue", "AbsInteraction"] 
        sorted in descending order of AbsInteraction.
    """
    # 1. Aggregate across samples (e.g., mean absolute interactions)
    interaction_matrix = np.mean(np.abs(shap_interaction_values), axis=0)
    
    # 2. Map each OHE feature to a base feature
    base_feature_names = [get_base_feature(n) for n in feature_names]
    unique_base_features = list(set(base_feature_names))
    
    # 3. Build a structure to accumulate aggregated pairwise interactions
    aggregated_interactions = {
        bf_i: {bf_j: 0.0 for bf_j in unique_base_features}
        for bf_i in unique_base_features
    }

    n_features = len(feature_names)
    for i in range(n_features):
        for j in range(i+1, n_features): # i+1 => no diagonal, no duplicates
            bf_i = base_feature_names[i]
            bf_j = base_feature_names[j]
            aggregated_interactions[bf_i][bf_j] += interaction_matrix[i, j]
    
    # 4. Convert to DataFrame
    data_records = []
    for bf_i in unique_base_features:
        for bf_j in unique_base_features:
            # If you want to keep only i <= j, add a condition to avoid duplicates
            interaction_val = aggregated_interactions[bf_i][bf_j]
            data_records.append((bf_i, bf_j, interaction_val))
    
    df_interactions = pd.DataFrame(data_records, columns=["Feature1", "Feature2", "InteractionValue"])
    df_interactions["AbsInteraction"] = df_interactions["InteractionValue"].abs()
    
    # Sort descending by absolute interaction
    df_interactions.sort_values("AbsInteraction", ascending=False, inplace=True)
    df_interactions.reset_index(drop=True, inplace=True)
    df_interactions_no_diagonal = df_interactions[df_interactions['Feature1'] != df_interactions['Feature2']]
    return df_interactions_no_diagonal


# Example usage:
def simple_get_base_feature(name):
    # Remove the cat__ prefix if present
    if name.startswith("cat__"):
        name = name[len("cat__"):]
    # Then split on the first underscore only
    return name.split("_", 1)[0]

df_agg_interactions = aggregate_shap_interactions(
    shap_interaction_values=shap_interaction_values,
    feature_names=feature_names,
    get_base_feature=simple_get_base_feature
)

# Print the top 20 interactions
print(df_agg_interactions.head(20))

# --- Pick Top 20 Interactions ---
df_top_20 = df_agg_interactions.head(20).copy()

# Create a convenient label for each pair
df_top_20["Pair"] = df_top_20["Feature1"] + " & " + df_top_20["Feature2"]

# --- Plot ---
plt.figure(figsize=(10, 6))
sns.barplot(
    data=df_top_20, 
    y="Pair", 
    x="AbsInteraction", 
    color="royalblue"
)
plt.title("Top 20 Pairwise Feature Interactions by Absolute SHAP Value")
plt.xlabel("Absolute SHAP Interaction Value")
plt.ylabel("Feature Pair")
plt.tight_layout()
plt.show()


# %%
import numpy as np
import pandas as pd
import shap
import seaborn as sns
import matplotlib.pyplot as plt

##############################
# 1. Pairwise Aggregation
##############################
def aggregate_shap_interactions_pairwise(shap_interaction_values, feature_names, get_base_feature):
    """
    Aggregates pairwise SHAP interaction values back to their original (pre–one-hot) features,
    and returns a DataFrame of the mean absolute interaction for each (Feature1, Feature2).
    """
    # shap_interaction_values: [n_samples, n_features, n_features]
    # 1) Mean of absolute interaction across samples
    interaction_matrix = np.mean(np.abs(shap_interaction_values), axis=0)  # shape: (n_features, n_features)

    # 2) Map each OHE feature to a base feature
    base_feature_names = [get_base_feature(f) for f in feature_names]
    unique_base_features = sorted(set(base_feature_names))

    # 3) Accumulate pairwise interactions
    #    We'll sum interactions for each pair (bf_i, bf_j)
    #    Because i<j is symmetric in the shap_interaction_values, we avoid double counting.
    aggregated = {}
    for bf_i in unique_base_features:
        aggregated[bf_i] = {}
        for bf_j in unique_base_features:
            aggregated[bf_i][bf_j] = 0.0

    n_features = len(feature_names)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            bf_i = base_feature_names[i]
            bf_j = base_feature_names[j]
            aggregated[bf_i][bf_j] += interaction_matrix[i, j]
            # You could also decide to add symmetrical entries if desired:
            aggregated[bf_j][bf_i] += interaction_matrix[i, j]  # for simpler referencing

    # 4) Convert to a long DataFrame
    records = []
    for bf_i in unique_base_features:
        for bf_j in unique_base_features:
            # Avoid i == j and only keep i < j in the final output (unique pairs)
            if bf_i < bf_j:
                val = aggregated[bf_i][bf_j]
                records.append({
                    "FeatureCombo": f"{bf_i} & {bf_j}",
                    "Feature1": bf_i,
                    "Feature2": bf_j,
                    "InteractionValue": val,
                    "AbsInteraction": abs(val),
                    "Order": 2  # Mark as pairwise
                })

    df = pd.DataFrame(records)
    df.sort_values("AbsInteraction", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

##############################
# 2. "Three-way" Aggregation
##############################
def aggregate_shap_interactions_three_way(shap_interaction_values, feature_names, get_base_feature):
    """
    Naively computes a 'three-way' measure by summing the absolute pairwise interactions
    (i,j), (i,k), and (j,k) for each triple i < j < k.

    NOTE: This does not represent a true SHAP 3-way synergy. It's a proxy 
    by combining the pairwise interactions among the triple.
    """
    # 1) Mean of absolute interaction across samples
    interaction_matrix = np.mean(np.abs(shap_interaction_values), axis=0)  # shape: (n_features, n_features)

    # 2) Map each OHE feature to a base feature
    base_feature_names = [get_base_feature(f) for f in feature_names]
    unique_base_features = sorted(set(base_feature_names))

    # 3) We will group by base features for i,j,k. Because different OHE columns might map
    #    to the same base feature, let's gather all indices that map to each base feature.
    feature_indices_by_base = {}
    for idx, bf in enumerate(base_feature_names):
        if bf not in feature_indices_by_base:
            feature_indices_by_base[bf] = []
        feature_indices_by_base[bf].append(idx)

    # 4) For each triple (bf_i, bf_j, bf_k) with i < j < k, compute sum of pairwise interactions
    records = []
    bf_list = list(unique_base_features)
    n_bf = len(bf_list)

    for i in range(n_bf):
        for j in range(i + 1, n_bf):
            for k in range(j + 1, n_bf):
                bf_i, bf_j, bf_k = bf_list[i], bf_list[j], bf_list[k]

                # For all actual columns that map to bf_i, bf_j, bf_k, sum the relevant pairwise interactions:
                indices_i = feature_indices_by_base[bf_i]
                indices_j = feature_indices_by_base[bf_j]
                indices_k = feature_indices_by_base[bf_k]

                # We'll sum the absolute interaction_matrix across all pairs (i', j'), (i', k'), (j', k')
                total_interaction = 0.0
                for ii in indices_i:
                    for jj in indices_j:
                        total_interaction += interaction_matrix[ii, jj]
                        total_interaction += interaction_matrix[jj, ii]  # symmetrical

                    for kk in indices_k:
                        total_interaction += interaction_matrix[ii, kk]
                        total_interaction += interaction_matrix[kk, ii]

                for jj in indices_j:
                    for kk in indices_k:
                        total_interaction += interaction_matrix[jj, kk]
                        total_interaction += interaction_matrix[kk, jj]

                # We'll keep it as the approximate measure
                records.append({
                    "FeatureCombo": f"{bf_i} & {bf_j} & {bf_k}",
                    "Feature1": bf_i,
                    "Feature2": bf_j,
                    "Feature3": bf_k,
                    "InteractionValue": total_interaction,
                    "AbsInteraction": abs(total_interaction),
                    "Order": 3  # Mark as three-way
                })

    df = pd.DataFrame(records)
    df.sort_values("AbsInteraction", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


##############################
# 3. Putting it all together
##############################

# Suppose you already have:
#   shap_interaction_values = explainer.shap_interaction_values(X_preprocessed)
#   feature_names = loaded_gbt.named_steps['preprocessor'].get_feature_names_out()
#   We'll reuse your simple_get_base_feature function:
import re

def simple_get_base_feature(name):
    """
    Strips off numerical one-hot encoding suffixes (e.g., 'survey_wave_2021' → 'survey_wave').
    """
    # Remove 'num__' or 'cat__' prefix if present
    if name.startswith("num__"):
        name = name[len("num__"):]
    if name.startswith("cat__"):
        name = name[len("cat__"):]

    # Remove one-hot encoding suffix (_YYYY or _X.X for categorical variables)
    name = re.sub(r"_\d+(\.\d+)?$", "", name)  # Removes _2021, _2.0, etc.

    return name

# -- 3.1 Compute pairwise and "three-way" aggregates --
df_pairwise = aggregate_shap_interactions_pairwise(
    shap_interaction_values, 
    feature_names, 
    get_base_feature=simple_get_base_feature
)

df_three_way = aggregate_shap_interactions_three_way(
    shap_interaction_values, 
    feature_names, 
    get_base_feature=simple_get_base_feature
)

# -- 3.2 Combine them into one table --
#    For consistency, we can unify the column names
df_pairwise["Feature3"] = None  # no third feature for pairs
combined_df = pd.concat([df_pairwise, df_three_way], ignore_index=True)

# Sort by absolute interaction
combined_df.sort_values("AbsInteraction", ascending=False, inplace=True)
combined_df.reset_index(drop=True, inplace=True)

# -- 3.3 Pick top 20 overall --
df_top_20 = combined_df.head(20).copy()

# For a nicer label that works for both pairwise and triple combos:
df_top_20["Label"] = df_top_20["FeatureCombo"]

# -- 3.4 Plot a bar chart of the top 20 combos --
plt.figure(figsize=(10, 6))
sns.barplot(data=df_top_20, y="Label", x="AbsInteraction", color="royalblue")
plt.title("Top 20 Interactions (Pairwise + Three-way) by Absolute SHAP Value")
plt.xlabel("Absolute SHAP Interaction Value")
plt.ylabel("Feature Combination")
plt.tight_layout()
plt.show()

# You now have a bar plot of the top 20 interactions (including both 2- and 3-feature combos).

# %%
df_top_20

# %%
# ## XGBOOST

# %%
import os
import logging
import joblib
import xgboost as xgb

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import FunctionTransformer
import numpy as np

# ======================
# 1. Expanded Hyperparameter Grid
# ======================
param_dist = {
    "classifier__n_estimators": randint(400, 2500),
    "classifier__learning_rate": loguniform(0.01, 0.2),
    "classifier__max_depth": randint(3, 11),
    "classifier__min_child_weight": randint(1, 12),
    "classifier__subsample": uniform(0.6, 0.4),         # [0.6, 1.0]
    "classifier__colsample_bytree": uniform(0.6, 0.4),  # [0.6, 1.0] (must be <= 1.0)
    "classifier__gamma": loguniform(1e-8, 1.0),
    "classifier__reg_alpha": loguniform(1e-8, 10.0),
    "classifier__reg_lambda": loguniform(0.1, 10.0),
}

# ============================
# 2. OPTIONAL FEATURE ENGINEERING
# ============================
# Example: add a custom transformer to create domain-specific or interaction features
# (Here, we just pass data through, but you'd modify 'feature_engineering' to transform the input DataFrame.)

def feature_engineering(X):
    """
    Placeholder function where you can create domain-specific,
    polynomial, or ratio features. This must return a DataFrame or array.
    """
    # Example: create a simple ratio of two columns (if they exist)
    # if 'colA' in X and 'colB' in X:
    #     X['ratio_A_B'] = X['colA'] / (X['colB'] + 1e-9)
    
    return X

feature_eng_transformer = FunctionTransformer(feature_engineering, validate=False)

# ============================
# 3. Build Pipeline
# ============================
# We assume:
#   - You already have 'preprocessor' for OneHotEncoder, etc.
#   - 'X_train_with_indicators', 'y_train', 'X_test_with_indicators', 'y_test'
#   - 'train_evaluate_model(...)' function
#   - Constants: RANDOM_STATE, SCORING_METRIC, N_SPLITS_CV

xgb_clf = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    n_jobs=1,  # avoid nested parallelism (CV/search handles parallelism)
    random_state=RANDOM_STATE,
)

# Here we insert a feature engineering step *before* the preprocessor:
xgb_pipeline = Pipeline([
    ('feature_engineering', feature_eng_transformer),
    ('preprocessor', preprocessor),
    ('classifier', xgb_clf)
])

# =========================
# 4. (Optional) Early Stopping
# =========================
# In scikit-learn's RandomizedSearchCV, providing early_stopping_rounds is non-trivial because
# you need a separate validation set or a custom approach inside each CV fold.
# If you wish to do a simple holdout for early stopping, you'd do something like:
#
# fit_params = {
#    'classifier__early_stopping_rounds': 30,
#    'classifier__eval_metric': 'logloss',
#    'classifier__eval_set': [(X_val, y_val)],  # separate validation set
# }
#
# Then pass fit_params=fit_params to random_search.fit(...).
# This is more advanced; we’ll skip it here for brevity.

# ============================
# 5. Randomized Search Setup
# ============================
cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)

random_search = RandomizedSearchCV(
    estimator=xgb_pipeline,
    param_distributions=param_dist,
    n_iter=_N_ITER["xgb"],
    cv=cv,
    scoring=SCORING_METRIC,
    n_jobs=-1,
    verbose=1,
    random_state=RANDOM_STATE,
    return_train_score=True,
)

logging.info("Starting RandomizedSearchCV for expanded XGBoost grid...")
random_search.fit(X_train_with_indicators, y_train)
logging.info("RandomizedSearchCV complete.")

best_xgb_model = random_search.best_estimator_
logging.info(f"Best parameters: {random_search.best_params_}")
logging.info(f"Best CV {SCORING_METRIC}: {random_search.best_score_:.4f}")

# ============================
# 6. Evaluate & Save Best Model
# ============================
trained_best_xgb = train_evaluate_model(
    model=best_xgb_model,
    X_train=X_train_with_indicators,
    y_train=y_train,
    X_test=X_test_with_indicators,
    y_test=y_test,
    model_name="Tuned XGBoost Model"
)

model_filename = os.path.expanduser('~/work/vaping_project_data/best_xgb_model.joblib')
joblib.dump(trained_best_xgb, model_filename)
logging.info(f"Final XGBoost model saved to: {model_filename}")

# %%
# Load the model (when needed)
def feature_engineering(X):
    """
    Placeholder function where you can create domain-specific,
    polynomial, or ratio features. This must return a DataFrame or array.
    """
    # Example: create a simple ratio of two columns (if they exist)
    # if 'colA' in X and 'colB' in X:
    #     X['ratio_A_B'] = X['colA'] / (X['colB'] + 1e-9)
    
    return X

file_path = os.path.expanduser('~/work/vaping_project_data/best_xgb_model.joblib')
loaded_xgb = joblib.load(file_path)
print("Model loaded successfully.")

# %%
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import logging

# Filter features: remove "respondent_sex" (case-insensitive) and any feature starting with "missing_"
aggregated_feature_names = [
    feat for feat in X_train_with_indicators.columns.tolist() 
    if feat.lower() != 'respondent_sex' and not feat.lower().startswith("missing_")
]

# Print and log the features being plotted along with their count
print("Features being plotted:", aggregated_feature_names)
print("Total number of features plotted:", len(aggregated_feature_names))
logging.info(f"Features being plotted: {aggregated_feature_names}")
logging.info(f"Total number of features plotted: {len(aggregated_feature_names)}")

# Loop through each feature and plot its partial dependence on a separate figure
for feature in aggregated_feature_names:
    fig, ax = plt.subplots(figsize=(8, 6))
    PartialDependenceDisplay.from_estimator(loaded_xgb, X_train_with_indicators, [feature], ax=ax)
    ax.grid(False)  # Disable grid lines on the plot
    ax.set_title(f'Partial Dependence of {feature}')
    plt.tight_layout()
    plt.show()

# %%
# Function to plot feature importances
def plot_feature_importance(loaded_xgb, feature_names, top_n=20, title="Feature Importance"):
    """
    Plots the top N feature importances from a trained model.
    """
    if hasattr(loaded_xgb, 'feature_importances_'):
        importances = loaded_xgb.feature_importances_
    elif hasattr(loaded_xgb, 'named_steps') and 'classifier' in loaded_xgb.named_steps:
        if hasattr(loaded_xgb.named_steps['classifier'], 'feature_importances_'):
            importances = loaded_xgb.named_steps['classifier'].feature_importances_
        else:
            raise ValueError("Classifier does not have feature_importances_ attribute.")
    else:
        raise ValueError("Provided model does not have feature_importances_ attribute.")

    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    fi_df = fi_df.sort_values('Importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=fi_df)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Get the feature names from the preprocessor
feature_names = loaded_xgb.named_steps['preprocessor'].get_feature_names_out()

# Plot the top 20 most important features
plot_feature_importance(loaded_xgb, feature_names, top_n=20, title="Top 20 Most Important Features")

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_aggregated_feature_importance(model, top_n=20, title="Top 20 Aggregated Feature Importance"):
    """
    Plots the top N aggregated feature importances for a CatBoost model
    that is wrapped inside a Pipeline with a ColumnTransformer.
    
    Parameters
    ----------
    model : Pipeline
        A scikit-learn Pipeline that includes:
          - 'preprocessor': a ColumnTransformer or other transformer
          - 'classifier': a CatBoostClassifier
    top_n : int, optional (default=20)
        How many top aggregated features to display.
    title : str, optional
        Title of the plot.
    """
    # 1. Get the feature names after the preprocessor step
    feature_names = loaded_xgb.named_steps['preprocessor'].get_feature_names_out()
    
    # 2. Get the feature importances from CatBoost
    xgboost_estimator = loaded_xgb.named_steps['classifier']
    if not hasattr(xgboost_estimator, 'feature_importances_'):
        raise AttributeError("The CatBoost classifier does not expose 'feature_importances_'.")

    importances = xgboost_estimator.feature_importances_

    # 3. Aggregate importances by original feature
    aggregated_importance = {}
    
    for name, imp in zip(feature_names, importances):
        # Example naming conventions after ColumnTransformer + OneHotEncoder:
        #   "onehotencoder__Gender_Male"
        #   "remainder__Age"
        # Adjust this parsing logic as necessary for your pipeline.
        
        if "__" in name:
            # Split on the double underscore to separate the transformer name vs. the actual column
            parts = name.split("__", maxsplit=1)
            # parts[0] might be 'onehotencoder' or 'remainder'
            # parts[1] might be 'Gender_Male' or 'Age'
            # We'll then split again on '_' if needed to get just the original column name
            col_part = parts[1]
            
            # If the column was numeric (remainder), it may be simply 'Age'.
            # If the column was OHE, it might be 'Gender_Male' or 'Gender_Female'.
            # A simple approach is to take everything before the first underscore as the feature name:
            if "_" in col_part:
                original_feature = col_part.split("_", maxsplit=1)[0]
            else:
                original_feature = col_part
        else:
            # If there's no double underscore, assume the whole name is the feature
            original_feature = name
        
        # Sum up the importances
        aggregated_importance[original_feature] = aggregated_importance.get(original_feature, 0.0) + imp

    # 4. Make a DataFrame of aggregated importances and sort
    agg_df = pd.DataFrame(list(aggregated_importance.items()), columns=["Feature", "Importance"])
    agg_df.sort_values("Importance", ascending=False)

    # 5. Plot top N aggregated feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=agg_df.head(top_n))
    plt.title(title)
    plt.tight_layout()
    plt.show()

    # 6. Print the importance table (Top 20 Features)
    print(agg_df.head(20))


# --- Usage Example ---
# Assuming you have your best_model pipeline (with 'preprocessor' and CatBoost 'classifier'):

plot_aggregated_feature_importance(loaded_xgb, top_n=20, title="Top 20 Aggregated Feature Importance")

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import pandas as pd

def get_top_aggregated_features(loaded_xgb, top_n=20):
    """
    Returns a list of top_n original (aggregated) feature names
    based on the CatBoost feature importances in a Pipeline.
    
    Parameters
    ----------
    loaded_xgb : Pipeline
        A scikit-learn Pipeline with steps:
          - "preprocessor" : ColumnTransformer (or similar)
          - "classifier" : CatBoostClassifier
    top_n : int
        Number of top features to return
        
    Returns
    -------
    list of str
        Top N aggregated feature names
    """
    # Extract feature names that come out of the preprocessor
    feature_names = loaded_xgb.named_steps['preprocessor'].get_feature_names_out()
    
    # Extract importances from CatBoost
    catboost_estimator = loaded_xgb.named_steps['classifier']
    importances = catboost_estimator.feature_importances_

    # Aggregate importances by the original (pre-encoding) feature name
    aggregated_importance = {}
    for name, imp in zip(feature_names, importances):
        if "__" in name:
            # Example: "onehotencoder__Gender_Male" -> original_feature = "Gender"
            parts = name.split("__", maxsplit=1)
            col_part = parts[1]
            if "_" in col_part: 
                # For OHE columns like "Gender_Male"
                original_feature = col_part.split("_", maxsplit=1)[0]
            else:
                # For remainder numeric columns
                original_feature = col_part
        else:
            # If no __, assume name is the feature
            original_feature = name
        
        aggregated_importance[original_feature] = (
            aggregated_importance.get(original_feature, 0.0) + imp
        )

    # Turn into a DataFrame, sort by descending importance, and get the top_n
    agg_df = pd.DataFrame(
        list(aggregated_importance.items()), 
        columns=["Feature", "Importance"]
    ).sort_values("Importance", ascending=False)

    return agg_df.head(top_n)["Feature"].tolist()

# 1. Get the top 20 features by aggregated importance
top_features = get_top_aggregated_features(loaded_xgb, top_n=20)

# Ensure that "likelihood_grad_prof_school" is included even if it is not among the top 20
if "likelihood_grad_prof_school" not in top_features:
    top_features.append("likelihood_grad_prof_school")

print("Top aggregated features:\n", top_features)

# 2. Plot the partial dependence plot for each feature
for feat in top_features:
    if feat not in X_train_with_indicators.columns:
        print(f"Skipping feature '{feat}' as it is not found in the DataFrame.")
        continue

    fig, ax = plt.subplots(figsize=(6, 4))
    PartialDependenceDisplay.from_estimator(
        estimator=loaded_xgb,
        X=X_train_with_indicators,
        features=[feat],
        kind='average',
        grid_resolution=50,
        target=1,  # positive class for binary classification
        ax=ax
    )
    plt.title(f"Partial Dependence of {feat}")
    plt.show()

# %%
# Add this import at the top of your script
from scipy import sparse

# --- SHAP Feature Importance ---
# Suppose your ColumnTransformer has the name "cat" for the OneHotEncoder step
# and you pass in categorical_features as the input_features:
encoded_feature_names = (
    preprocessor
    .named_transformers_['cat']  # "cat" is the name of the OHE step in ColumnTransformer
    .get_feature_names_out(input_features=categorical_features)
)


# Extract components from the pipeline
preprocessor = loaded_xgb.named_steps['preprocessor']
classifier = loaded_xgb.named_steps['classifier']

# Process the data through the pipeline
X_processed = preprocessor.transform(X_train_with_indicators)

# Convert sparse matrix to dense if needed
if isinstance(X_processed, (sparse.csr_matrix, sparse.csc_matrix)):
    X_processed = X_processed.toarray()

# Create a SHAP explainer
explainer = shap.TreeExplainer(classifier)

# Calculate SHAP values (using a sample for faster computation)
sample_idx = np.random.choice(X_processed.shape[0], 100, replace=False)
shap_values = explainer.shap_values(X_processed[sample_idx])

# Get feature names from the preprocessor
feature_names = encoded_feature_names  # From your existing code

# %%
# Summary plot (feature importance)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, 
                 X_processed[sample_idx], 
                 feature_names=feature_names,
                 plot_type="bar",
                 show=False)
plt.title("SHAP Feature Importance (Mean Absolute Impact)")
plt.tight_layout()
plt.show()

# Detailed summary plot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, 
                 X_processed[sample_idx], 
                 feature_names=feature_names,
                 show=False)
plt.title("SHAP Value Distribution")
plt.tight_layout()
plt.show()

# %%
# Force plot for average prediction
plt.figure()
shap.force_plot(explainer.expected_value, 
                shap_values[0], 
                X_processed[0], 
                feature_names=feature_names,
                matplotlib=True,
                show=False)
plt.title("SHAP Force Plot for First Sample")
plt.tight_layout()
plt.show()

# %%
# Extract original categorical features from one-hot encoded column names
import re

# Get the list of one-hot encoded column names
one_hot_encoded_columns = feature_names.tolist()

# Extract original categorical features by splitting at the first underscore or dot
original_categorical_features = list(set([re.split(r'[_.]', col)[0] for col in one_hot_encoded_columns]))

print("Original Categorical Features:")
print(original_categorical_features)

# %%
# List of original categorical features
original_categorical_features = [
    'tranquilizers_12m', 'high_school_program', 'likelihood_military', 'times_skipped_class', 'sedatives_barbiturates_12m', 'father_education_level', 'moving_violation_tickets', 'desire_college_grad', 'hours_worked_per_week', 'father_in_household', 
    'amphetamines_12m', 'alcohol_12m', 'days_missed_school_illness', 'desire_vocational_school', 'marital_status', 'nights_out', 'likelihood_college_grad', 'other_narcotics_12m', 'likelihood_grad_prof_school', 'crack_12m', 
    'respondent_race', 'V2907', 'V2494', 'RESPONDENT', 'mother_education_level', 'V2146', 'num_siblings', 'likelihood_two_year_grad', 'school_region', 'growing_up_location', 
    'days_missed_skipping', 'distance_driven_weekly', 'desire_two_year_grad', 'self_rated_school_ability', 'five_plus_drinks_2wks', 'V2033', 'days_missed_other', 'V2030', 'expected_graduation_time', 'V2119', 
    'V2908', 'dating_frequency', 'marijuana_12m', 'likelihood_vocational_school', 'desire_military_service', 'political_preference', 'heroin_12m', 'mother_in_household', 'desire_grad_prof_school', 'driving_accidents', 
    'V2169', 'V2122', 'missing', 'respondent_sex', 'cocaine_12m', 'average_grades', 'weekly_income_other', 'ever_smoked', 'survey_wave', 'sibling_in_household'
]

# Compute mean absolute SHAP values
mean_abs_shap_values = np.abs(shap_values).mean(axis=0)

# Create a dictionary to map one-hot encoded features to their original features
feature_mapping = {}
for feature in original_categorical_features:
    feature_mapping[feature] = [col for col in feature_names if col.startswith(feature)]

# Aggregate SHAP values for each original feature
aggregated_shap_values = {}
for feature, cols in feature_mapping.items():
    # Find the indices of the one-hot encoded columns for this feature
    indices = [feature_names.tolist().index(col) for col in cols]
    # Sum the mean absolute SHAP values for these columns
    aggregated_shap_values[feature] = np.sum(mean_abs_shap_values[indices])

# Convert the aggregated SHAP values to a DataFrame
aggregated_shap_df = pd.DataFrame({
    'Feature': list(aggregated_shap_values.keys()),
    'Aggregated_SHAP': list(aggregated_shap_values.values())
})

# Sort by aggregated SHAP values in descending order
aggregated_shap_df = aggregated_shap_df.sort_values(by='Aggregated_SHAP', ascending=False)

# Display the top 20 aggregated features
top_n = 20  # Set to 20 for top 20 features
print("Top 20 Aggregated SHAP Features:")
print(aggregated_shap_df.head(top_n))

# Plot the top 20 aggregated features
plt.figure(figsize=(12, 8))
sns.barplot(x='Aggregated_SHAP', y='Feature', data=aggregated_shap_df.head(top_n), palette='viridis')
plt.title(f'Top {top_n} Aggregated SHAP Features')
plt.xlabel('Aggregated SHAP Value')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# %%
# Ensure SHAP is installed
import shap
shap.initjs()  # For visualization in notebooks

# Compute SHAP interaction values (using a sample for faster computation)
sample_idx = np.random.choice(X_processed.shape[0], 100, replace=False)  # Use a sample of 100 instances
shap_interaction_values = explainer.shap_interaction_values(X_processed[sample_idx])

# Get feature names from the preprocessor
feature_names = encoded_feature_names  # From your existing code

# %%
# Step 1: Store interactions and their values, avoiding duplicates and self-interactions
interaction_results = []

for i, feature_i in enumerate(original_categorical_features):
    for j, feature_j in enumerate(original_categorical_features):
        # Skip self-interactions
        if feature_i == feature_j:
            continue

        # Ensure unique pairs by sorting feature names
        feature_pair = tuple(sorted([feature_i, feature_j]))

        # Skip if the pair is already processed
        if feature_pair in [result[0] for result in interaction_results]:
            continue

        # Get indices for feature_i and feature_j
        indices_i = [feature_names.tolist().index(col) for col in feature_mapping[feature_i] if col in feature_names.tolist()]
        indices_j = [feature_names.tolist().index(col) for col in feature_mapping[feature_j] if col in feature_names.tolist()]

        if not indices_i or not indices_j:
            continue

        # Compute interaction value
        value = np.sum(np.abs(shap_interaction_values[:, indices_i, :][:, :, indices_j]))
        interaction_results.append((feature_pair, value))

# Step 2: Sort the interactions by their values
sorted_interactions = sorted(interaction_results, key=lambda x: x[1], reverse=True)

# Step 3: Convert results to a DataFrame for easy aggregation
interaction_df = pd.DataFrame(sorted_interactions, columns=["Feature Pair", "Interaction Value"])
interaction_df[["Feature A", "Feature B"]] = pd.DataFrame(interaction_df["Feature Pair"].tolist(), index=interaction_df.index)
interaction_df = interaction_df.drop(columns=["Feature Pair"])

# Step 4: Select the top 30 interactions
top_30_interactions = interaction_df.sort_values(by="Interaction Value", ascending=False).head(30)

# Step 5: Display the results
print("Top 30 Feature Interactions (Unique Pairs):")
print(top_30_interactions)

# Step 6: Visualize the top 30 interactions
import matplotlib.pyplot as plt

# Create a bar plot
plt.figure(figsize=(12, 8))
plt.barh(
    top_30_interactions.apply(lambda row: f"{row['Feature A']} & {row['Feature B']}", axis=1),
    top_30_interactions["Interaction Value"],
    color="skyblue",
)
plt.xlabel("Interaction Value")
plt.ylabel("Feature Pairs")
plt.title("Top 30 Feature Interactions (Unique Pairs)")
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.tight_layout()
plt.show()

# %%
import numpy as np
import pandas as pd
import shap
import seaborn as sns
import matplotlib.pyplot as plt

##############################
# 1. Pairwise Aggregation
##############################
def aggregate_shap_interactions_pairwise(shap_interaction_values, feature_names, get_base_feature):
    """
    Aggregates pairwise SHAP interaction values back to their original (pre–one-hot) features,
    and returns a DataFrame of the mean absolute interaction for each (Feature1, Feature2).
    """
    # shap_interaction_values: [n_samples, n_features, n_features]
    # 1) Mean of absolute interaction across samples
    interaction_matrix = np.mean(np.abs(shap_interaction_values), axis=0)  # shape: (n_features, n_features)

    # 2) Map each OHE feature to a base feature
    base_feature_names = [get_base_feature(f) for f in feature_names]
    unique_base_features = sorted(set(base_feature_names))

    # 3) Accumulate pairwise interactions
    #    We'll sum interactions for each pair (bf_i, bf_j)
    #    Because i<j is symmetric in the shap_interaction_values, we avoid double counting.
    aggregated = {}
    for bf_i in unique_base_features:
        aggregated[bf_i] = {}
        for bf_j in unique_base_features:
            aggregated[bf_i][bf_j] = 0.0

    n_features = len(feature_names)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            bf_i = base_feature_names[i]
            bf_j = base_feature_names[j]
            aggregated[bf_i][bf_j] += interaction_matrix[i, j]
            # You could also decide to add symmetrical entries if desired:
            aggregated[bf_j][bf_i] += interaction_matrix[i, j]  # for simpler referencing

    # 4) Convert to a long DataFrame
    records = []
    for bf_i in unique_base_features:
        for bf_j in unique_base_features:
            # Avoid i == j and only keep i < j in the final output (unique pairs)
            if bf_i < bf_j:
                val = aggregated[bf_i][bf_j]
                records.append({
                    "FeatureCombo": f"{bf_i} & {bf_j}",
                    "Feature1": bf_i,
                    "Feature2": bf_j,
                    "InteractionValue": val,
                    "AbsInteraction": abs(val),
                    "Order": 2  # Mark as pairwise
                })

    df = pd.DataFrame(records)
    df.sort_values("AbsInteraction", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

##############################
# 2. "Three-way" Aggregation
##############################
def aggregate_shap_interactions_three_way(shap_interaction_values, feature_names, get_base_feature):
    """
    Naively computes a 'three-way' measure by summing the absolute pairwise interactions
    (i,j), (i,k), and (j,k) for each triple i < j < k.

    NOTE: This does not represent a true SHAP 3-way synergy. It's a proxy 
    by combining the pairwise interactions among the triple.
    """
    # 1) Mean of absolute interaction across samples
    interaction_matrix = np.mean(np.abs(shap_interaction_values), axis=0)  # shape: (n_features, n_features)

    # 2) Map each OHE feature to a base feature
    base_feature_names = [get_base_feature(f) for f in feature_names]
    unique_base_features = sorted(set(base_feature_names))

    # 3) We will group by base features for i,j,k. Because different OHE columns might map
    #    to the same base feature, let's gather all indices that map to each base feature.
    feature_indices_by_base = {}
    for idx, bf in enumerate(base_feature_names):
        if bf not in feature_indices_by_base:
            feature_indices_by_base[bf] = []
        feature_indices_by_base[bf].append(idx)

    # 4) For each triple (bf_i, bf_j, bf_k) with i < j < k, compute sum of pairwise interactions
    records = []
    bf_list = list(unique_base_features)
    n_bf = len(bf_list)

    for i in range(n_bf):
        for j in range(i + 1, n_bf):
            for k in range(j + 1, n_bf):
                bf_i, bf_j, bf_k = bf_list[i], bf_list[j], bf_list[k]

                # For all actual columns that map to bf_i, bf_j, bf_k, sum the relevant pairwise interactions:
                indices_i = feature_indices_by_base[bf_i]
                indices_j = feature_indices_by_base[bf_j]
                indices_k = feature_indices_by_base[bf_k]

                # We'll sum the absolute interaction_matrix across all pairs (i', j'), (i', k'), (j', k')
                total_interaction = 0.0
                for ii in indices_i:
                    for jj in indices_j:
                        total_interaction += interaction_matrix[ii, jj]
                        total_interaction += interaction_matrix[jj, ii]  # symmetrical

                    for kk in indices_k:
                        total_interaction += interaction_matrix[ii, kk]
                        total_interaction += interaction_matrix[kk, ii]

                for jj in indices_j:
                    for kk in indices_k:
                        total_interaction += interaction_matrix[jj, kk]
                        total_interaction += interaction_matrix[kk, jj]

                # We'll keep it as the approximate measure
                records.append({
                    "FeatureCombo": f"{bf_i} & {bf_j} & {bf_k}",
                    "Feature1": bf_i,
                    "Feature2": bf_j,
                    "Feature3": bf_k,
                    "InteractionValue": total_interaction,
                    "AbsInteraction": abs(total_interaction),
                    "Order": 3  # Mark as three-way
                })

    df = pd.DataFrame(records)
    df.sort_values("AbsInteraction", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


##############################
# 3. Putting it all together
##############################

# Suppose you already have:
#   shap_interaction_values = explainer.shap_interaction_values(X_preprocessed)
#   feature_names = loaded_gbt.named_steps['preprocessor'].get_feature_names_out()
#   We'll reuse your simple_get_base_feature function:
import re

def simple_get_base_feature(name):
    """
    Strips off numerical one-hot encoding suffixes (e.g., 'survey_wave_2021' → 'survey_wave').
    """
    # Remove 'num__' or 'cat__' prefix if present
    if name.startswith("num__"):
        name = name[len("num__"):]
    if name.startswith("cat__"):
        name = name[len("cat__"):]

    # Remove one-hot encoding suffix (_YYYY or _X.X for categorical variables)
    name = re.sub(r"_\d+(\.\d+)?$", "", name)  # Removes _2021, _2.0, etc.

    return name

# -- 3.1 Compute pairwise and "three-way" aggregates --
df_pairwise = aggregate_shap_interactions_pairwise(
    shap_interaction_values, 
    feature_names, 
    get_base_feature=simple_get_base_feature
)

df_three_way = aggregate_shap_interactions_three_way(
    shap_interaction_values, 
    feature_names, 
    get_base_feature=simple_get_base_feature
)

# -- 3.2 Combine them into one table --
#    For consistency, we can unify the column names
df_pairwise["Feature3"] = None  # no third feature for pairs
combined_df = pd.concat([df_pairwise, df_three_way], ignore_index=True)

# Sort by absolute interaction
combined_df.sort_values("AbsInteraction", ascending=False, inplace=True)
combined_df.reset_index(drop=True, inplace=True)

# -- 3.3 Pick top 20 overall --
df_top_20 = combined_df.head(20).copy()

# For a nicer label that works for both pairwise and triple combos:
df_top_20["Label"] = df_top_20["FeatureCombo"]

# -- 3.4 Plot a bar chart of the top 20 combos --
plt.figure(figsize=(10, 6))
sns.barplot(data=df_top_20, y="Label", x="AbsInteraction", color="royalblue")
plt.title("Top 20 Interactions (Pairwise + Three-way) by Absolute SHAP Value")
plt.xlabel("Absolute SHAP Interaction Value")
plt.ylabel("Feature Combination")
plt.tight_layout()
plt.show()

# You now have a bar plot of the top 20 interactions (including both 2- and 3-feature combos).

# %%
df_top_20

# %%
# ## CatBoost

# %%
import os
import logging
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import MissingIndicator
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from catboost import CatBoostClassifier
from scipy.stats import uniform, randint

# Create a pipeline with the preprocessor and CatBoost classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=0,
        random_seed=RANDOM_STATE,
        thread_count=1,  # avoid nested parallelism (CV/search handles parallelism)
    ))
])

# Define the parameter distribution for RandomizedSearchCV
param_dist = {
    "classifier__iterations": randint(500, 4000),
    "classifier__learning_rate": loguniform(1e-3, 0.3),
    "classifier__depth": randint(4, 11),
    "classifier__l2_leaf_reg": loguniform(1.0, 50.0),
    "classifier__border_count": randint(32, 255),
    "classifier__bagging_temperature": uniform(0.0, 1.0),
    "classifier__random_strength": loguniform(1e-8, 10.0),
    "classifier__auto_class_weights": [None, "Balanced"],
    "classifier__od_type": ["IncToDec", "Iter"],
    "classifier__od_wait": randint(10, 80),
}

# Perform RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=_N_ITER["catboost"],
    cv=StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE),
    scoring=SCORING_METRIC,
    n_jobs=-1,
    verbose=VERBOSE,
    random_state=RANDOM_STATE,
    return_train_score=True,
)

random_search.fit(X_train_with_indicators, y_train)

# Log the best parameters and score
logging.info("Best parameters found: " + str(random_search.best_params_))
logging.info(f"Best cross-validation {SCORING_METRIC}: {random_search.best_score_:.4f}")

# Evaluate the best model
best_model = random_search.best_estimator_

# Transform the data using the fitted preprocessor in the best model
X_train_transformed = best_model.named_steps['preprocessor'].transform(X_train_with_indicators)
X_test_transformed = best_model.named_steps['preprocessor'].transform(X_test_with_indicators)

# Predict and evaluate
y_pred = best_model.predict(X_test_with_indicators)
y_pred_proba = best_model.predict_proba(X_test_with_indicators)[:, 1]

logging.info("=== Best CatBoost Model Evaluation ===")
logging.info("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))
logging.info("\nClassification Report:\n" + str(classification_report(y_test, y_pred)))
logging.info(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_pred_proba):.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Best CatBoost Model ROC Curve on Test Data')
plt.legend(loc='lower right')
plt.show()

# Save the best model
model_save_path = 'best_catboost_model.pkl'
joblib.dump(best_model, model_save_path)
logging.info(f"Best CatBoost model saved to '{model_save_path}'.")

# %%
model_filename = os.path.expanduser('~/work/vaping_project_data/best_cb_model.joblib')
joblib.dump(best_model, model_filename)
logging.info(f"Final CatBoost model saved to: {model_filename}")

# %%
# Load the model (when needed)
file_path = os.path.expanduser('~/work/vaping_project_data/best_cb_model.joblib')
loaded_cb = joblib.load(file_path)
print("Model loaded successfully.")

# %%
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import logging

# Filter features: remove "respondent_sex" (case-insensitive) and any feature starting with "missing_"
aggregated_feature_names = [
    feat for feat in X_train_with_indicators.columns.tolist() 
    if feat.lower() != 'respondent_sex' and not feat.lower().startswith("missing_")
]

# Print and log the features being plotted along with their count
print("Features being plotted:", aggregated_feature_names)
print("Total number of features plotted:", len(aggregated_feature_names))
logging.info(f"Features being plotted: {aggregated_feature_names}")
logging.info(f"Total number of features plotted: {len(aggregated_feature_names)}")

# Loop through each feature and plot its partial dependence on a separate figure
for feature in aggregated_feature_names:
    fig, ax = plt.subplots(figsize=(8, 6))
    PartialDependenceDisplay.from_estimator(loaded_cb, X_train_with_indicators, [feature], ax=ax)
    ax.grid(False)  # Disable grid lines on the plot
    ax.set_title(f'Partial Dependence of {feature}')
    plt.tight_layout()
    plt.show()

# %%
# Function to plot feature importances
def plot_feature_importance(loaded_cb, feature_names, top_n=20, title="Feature Importance"):
    """
    Plots the top N feature importances from a trained model.
    """
    if hasattr(loaded_cb, 'feature_importances_'):
        importances = loaded_cb.feature_importances_
    elif hasattr(loaded_cb, 'named_steps') and 'classifier' in loaded_cb.named_steps:
        if hasattr(loaded_cb.named_steps['classifier'], 'feature_importances_'):
            importances = loaded_cb.named_steps['classifier'].feature_importances_
        else:
            raise ValueError("Classifier does not have feature_importances_ attribute.")
    else:
        raise ValueError("Provided model does not have feature_importances_ attribute.")

    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    fi_df = fi_df.sort_values('Importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=fi_df)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Get the feature names from the preprocessor
feature_names = loaded_cb.named_steps['preprocessor'].get_feature_names_out()

# Plot the top 20 most important features
plot_feature_importance(loaded_cb, feature_names, top_n=20, title="Top 20 Most Important Features")

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_aggregated_feature_importance(model, top_n=20, title="Top 20 Aggregated Feature Importance"):
    """
    Plots the top N aggregated feature importances for a CatBoost model
    that is wrapped inside a Pipeline with a ColumnTransformer.
    
    Parameters
    ----------
    model : Pipeline
        A scikit-learn Pipeline that includes:
          - 'preprocessor': a ColumnTransformer or other transformer
          - 'classifier': a CatBoostClassifier
    top_n : int, optional (default=20)
        How many top aggregated features to display.
    title : str, optional
        Title of the plot.
    """
    # 1. Get the feature names after the preprocessor step
    feature_names = loaded_cb.named_steps['preprocessor'].get_feature_names_out()
    
    # 2. Get the feature importances from CatBoost
    catboost_estimator = loaded_cb.named_steps['classifier']
    if not hasattr(catboost_estimator, 'feature_importances_'):
        raise AttributeError("The CatBoost classifier does not expose 'feature_importances_'.")

    importances = catboost_estimator.feature_importances_

    # 3. Aggregate importances by original feature
    aggregated_importance = {}
    
    for name, imp in zip(feature_names, importances):
        # Example naming conventions after ColumnTransformer + OneHotEncoder:
        #   "onehotencoder__Gender_Male"
        #   "remainder__Age"
        # Adjust this parsing logic as necessary for your pipeline.
        
        if "__" in name:
            # Split on the double underscore to separate the transformer name vs. the actual column
            parts = name.split("__", maxsplit=1)
            # parts[0] might be 'onehotencoder' or 'remainder'
            # parts[1] might be 'Gender_Male' or 'Age'
            # We'll then split again on '_' if needed to get just the original column name
            col_part = parts[1]
            
            # If the column was numeric (remainder), it may be simply 'Age'.
            # If the column was OHE, it might be 'Gender_Male' or 'Gender_Female'.
            # A simple approach is to take everything before the first underscore as the feature name:
            if "_" in col_part:
                original_feature = col_part.split("_", maxsplit=1)[0]
            else:
                original_feature = col_part
        else:
            # If there's no double underscore, assume the whole name is the feature
            original_feature = name
        
        # Sum up the importances
        aggregated_importance[original_feature] = aggregated_importance.get(original_feature, 0.0) + imp

    # 4. Make a DataFrame of aggregated importances and sort
    agg_df = pd.DataFrame(list(aggregated_importance.items()), columns=["Feature", "Importance"])
    agg_df = agg_df.sort_values("Importance", ascending=False)

    print(agg_df.head(20))

    # 5. Plot top N aggregated feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=agg_df.head(top_n))
    plt.title(title)
    plt.tight_layout()
    plt.show()


# --- Usage Example ---
# Assuming you have your best_model pipeline (with 'preprocessor' and CatBoost 'classifier'):

plot_aggregated_feature_importance(loaded_cb, top_n=20, title="Top 20 Aggregated Feature Importance")

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay


def get_top_aggregated_features(loaded_cb, top_n=20):
    """
    Returns a list of top_n original (aggregated) feature names
    based on the CatBoost feature importances in a Pipeline.
    
    Parameters
    ----------
    model : Pipeline
        A scikit-learn Pipeline with steps:
          - "preprocessor" : ColumnTransformer (or similar)
          - "classifier" : CatBoostClassifier
    top_n : int
        Number of top features to return
        
    Returns
    -------
    list of str
        Top N aggregated feature names
    """
    # Extract feature names that come out of the preprocessor
    feature_names = loaded_cb.named_steps['preprocessor'].get_feature_names_out()
    
    # Extract importances from CatBoost
    catboost_estimator = loaded_cb.named_steps['classifier']
    importances = catboost_estimator.feature_importances_

    # Aggregate importances by the original (pre-encoding) feature name
    aggregated_importance = {}
    for name, imp in zip(feature_names, importances):
        if "__" in name:
            # Example: "onehotencoder__Gender_Male" -> original_feature = "Gender"
            parts = name.split("__", maxsplit=1)
            col_part = parts[1]
            if "_" in col_part: 
                # For OHE columns like "Gender_Male"
                original_feature = col_part.split("_", maxsplit=1)[0]
            else:
                # For remainder numeric columns
                original_feature = col_part
        else:
            # If no __, assume name is the feature
            original_feature = name
        
        aggregated_importance[original_feature] = (
            aggregated_importance.get(original_feature, 0.0) + imp
        )

    # Turn into a DataFrame, sort, and get top_n
    agg_df = pd.DataFrame(
        list(aggregated_importance.items()), 
        columns=["Feature", "Importance"]
    ).sort_values("Importance", ascending=False)

    return agg_df.head(top_n)["Feature"].tolist()

# 1. Get the top 10 features by aggregated importance
top_features = get_top_aggregated_features(loaded_cb, top_n=20)
print("Top 10 aggregated features:\n", top_features)

# Plot the partial dependence plot
for feat in top_features:
    if feat not in X_train_with_indicators.columns:
        print(f"Skipping feature '{feat}' as it is not found in the DataFrame.")
        continue

    fig, ax = plt.subplots(figsize=(6, 4))
    PartialDependenceDisplay.from_estimator(
        estimator=loaded_cb,
        X=X_train_with_indicators,
        features=[feat],
        kind='average',
        grid_resolution=50,
        target=1,  # positive class for binary classification
        ax=ax
    )
    plt.title(f"Partial Dependence of {feat}")
    plt.show()



# %%
import shap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

X_train_transformed = loaded_cb.named_steps['preprocessor'].transform(X_train_with_indicators)


def plot_top_shap_aggregated_features(model, X_train_transformed, feature_names, top_n=20, title="Top 20 SHAP Aggregated Features"):
    """
    Plots the top N aggregated SHAP feature importances for a model.
    
    Parameters
    ----------
    model : Pipeline
        A scikit-learn Pipeline that includes:
          - 'preprocessor': a ColumnTransformer or other transformer
          - 'classifier': a CatBoostClassifier
    X_train_transformed : array-like
        Transformed training data (output of the preprocessor).
    feature_names : array-like
        Feature names after transformation.
    top_n : int, optional (default=20)
        How many top aggregated features to display.
    title : str, optional
        Title of the plot.
    """
    # 1. Extract the classifier from the pipeline
    catboost_estimator = loaded_cb.named_steps['classifier']
    
    # 2. Initialize SHAP explainer for the CatBoost model
    explainer = shap.TreeExplainer(catboost_estimator)
    
    # 3. Compute SHAP values for the transformed training data
    shap_values = explainer.shap_values(X_train_transformed)

    # For classification problems, shap_values is a list (one element per class).
    # We use the positive class for binary classification (index 1).
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # 4. Aggregate SHAP values back to the original feature names
    aggregated_shap = {}
    for i, name in enumerate(feature_names):
        # Parse original feature name from encoded feature name
        if "__" in name:
            col_part = name.split("__", maxsplit=1)[1]  # Split after the double underscore
            if "_" in col_part:
                original_feature = col_part.split("_", maxsplit=1)[0]
            else:
                original_feature = col_part
        else:
            original_feature = name

        # Sum SHAP values for the same original feature
        aggregated_shap[original_feature] = aggregated_shap.get(original_feature, 0.0) + abs(shap_values[:, i]).mean()
    
    # 5. Create a DataFrame of aggregated SHAP values and sort by importance
    shap_df = pd.DataFrame(list(aggregated_shap.items()), columns=["Feature", "SHAP Importance"])
    shap_df = shap_df.sort_values("SHAP Importance", ascending=False)

    print(shap_df.head(20))

    # 6. Plot the top N SHAP aggregated feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x="SHAP Importance", y="Feature", data=shap_df.head(top_n))
    plt.title(title)
    plt.tight_layout()
    plt.show()


# --- Usage Example ---
# Assuming you have:
# - 'best_model': your fitted pipeline
# - 'X_train_transformed': the transformed training data
# - 'feature_names': the output of 'get_feature_names_out()' from the preprocessor

plot_top_shap_aggregated_features(
     model=loaded_cb,
     X_train_transformed=X_train_transformed,
     feature_names=loaded_cb.named_steps['preprocessor'].get_feature_names_out(),
     top_n=20,
     title="Top 20 SHAP Aggregated Features"
)

# %%
###################################
# SHAP Interactions
###################################

import shap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 1: Extract the classifier from the pipeline ---
catboost_estimator = loaded_cb.named_steps['classifier']

# --- Step 2: Initialize SHAP explainer ---
explainer = shap.TreeExplainer(catboost_estimator)

# %%
# Sample a subset of the training data
sample_size = 300  # Adjust based on available memory and dataset size
X_train_sample = X_train_with_indicators.sample(n=sample_size, random_state=42)

# Transform the sampled data
X_train_sample_transformed = loaded_cb.named_steps['preprocessor'].transform(X_train_sample)

# Compute SHAP interaction values for the sample
interaction_values = explainer.shap_interaction_values(X_train_sample_transformed)

# %%
# Save interaction values to a .npy file
np.save('catboost_shap_interaction_values.npy', interaction_values)

# To load it later:
# loaded_interaction_values = np.load('shap_interaction_values.npy')

# %%
import numpy as np
import pandas as pd

# List of original categorical features
original_categorical_features = [
    'tranquilizers_12m', 'high_school_program', 'likelihood_military', 'times_skipped_class', 'sedatives_barbiturates_12m', 'father_education_level', 'moving_violation_tickets', 'desire_college_grad', 'hours_worked_per_week', 'father_in_household', 
    'amphetamines_12m', 'alcohol_12m', 'days_missed_school_illness', 'desire_vocational_school', 'marital_status', 'nights_out', 'likelihood_college_grad', 'other_narcotics_12m', 'likelihood_grad_prof_school', 'crack_12m', 
    'respondent_race', 'V2907', 'V2494', 'RESPONDENT', 'mother_education_level', 'V2146', 'num_siblings', 'likelihood_two_year_grad', 'school_region', 'growing_up_location', 
    'days_missed_skipping', 'distance_driven_weekly', 'desire_two_year_grad', 'self_rated_school_ability', 'five_plus_drinks_2wks', 'V2033', 'days_missed_other', 'V2030', 'expected_graduation_time', 'V2119', 
    'V2908', 'dating_frequency', 'marijuana_12m', 'likelihood_vocational_school', 'desire_military_service', 'political_preference', 'heroin_12m', 'mother_in_household', 'desire_grad_prof_school', 'driving_accidents', 
    'V2169', 'V2122', 'missing', 'respondent_sex', 'cocaine_12m', 'average_grades', 'weekly_income_other', 'ever_smoked', 'survey_wave', 'sibling_in_household'
]

# Get the feature names from your fitted pipeline
feature_names = loaded_cb.named_steps['preprocessor'].get_feature_names_out()

# Inspect them to see how they're named
print("Transformed feature names:\n", feature_names)

# A helper function to check if a transformed column belongs to a given original feature
def belongs_to_original(col_name: str, original_feat: str) -> bool:
    parts = col_name.split("__", maxsplit=1)
    if len(parts) == 2:
        encoded_part = parts[1]  # e.g. "tranquilizers_12m_0"
    else:
        encoded_part = parts[0]
    return (encoded_part == original_feat) or encoded_part.startswith(original_feat + "_")

# Create a dictionary to map each original categorical feature to its transformed columns
feature_mapping = {}
for feature in original_categorical_features:
    matched_cols = [
        col for col in feature_names 
        if belongs_to_original(col, feature)
    ]
    feature_mapping[feature] = matched_cols

# Now do your SHAP interaction aggregation
# Parallelize the computation for better performance
def compute_single_interaction(i, j, feature_i, feature_j, interaction_values, feature_mapping, feature_names):
    """Compute interaction value for a single feature pair."""
    indices_i = [feature_names.tolist().index(c) for c in feature_mapping[feature_i]]
    indices_j = [feature_names.tolist().index(c) for c in feature_mapping[feature_j]]
    value = np.sum(np.abs(interaction_values[:, indices_i, :][:, :, indices_j]))
    return i, j, value

# Parallelize the nested loop computation
print(f"Computing {len(original_categorical_features)**2} interaction values in parallel...")
results = Parallel(n_jobs=-1, verbose=1)(
    delayed(compute_single_interaction)(
        i, j, feature_i, feature_j, interaction_values, feature_mapping, feature_names
    )
    for i, feature_i in enumerate(original_categorical_features)
    for j, feature_j in enumerate(original_categorical_features)
)

# Populate matrix from parallel results
aggregated_interaction_matrix = np.zeros((len(original_categorical_features), len(original_categorical_features)))
for i, j, value in results:
    aggregated_interaction_matrix[i, j] = value

aggregated_interaction_df = pd.DataFrame(
    aggregated_interaction_matrix,
    index=original_categorical_features,
    columns=original_categorical_features
)

print("Aggregated Interaction DataFrame:\n", aggregated_interaction_df)

# %%
# Step 1: Store interactions and their values, avoiding self-interactions AND duplicates
# Parallelize for better performance
def compute_pairwise_interaction(i, feature_i, j, feature_j, interaction_values, feature_mapping, feature_names):
    """Compute interaction for a unique feature pair."""
    # Get indices for feature_i and feature_j
    indices_i = [
        feature_names.tolist().index(col) 
        for col in feature_mapping[feature_i] 
        if col in feature_names.tolist()
    ]
    indices_j = [
        feature_names.tolist().index(col) 
        for col in feature_mapping[feature_j] 
        if col in feature_names.tolist()
    ]
    
    if not indices_i or not indices_j:
        return None
    
    # Compute interaction value
    value = np.sum(np.abs(interaction_values[:, indices_i, :][:, :, indices_j]))
    return ((feature_i, feature_j), value)

# Create list of unique pairs
print(f"Computing {len(original_categorical_features) * (len(original_categorical_features) - 1) // 2} unique pairwise interactions in parallel...")
pair_tasks = [
    (i, original_categorical_features[i], j, original_categorical_features[j])
    for i in range(len(original_categorical_features))
    for j in range(i + 1, len(original_categorical_features))
]

# Parallelize the computation
interaction_results = Parallel(n_jobs=-1, verbose=1)(
    delayed(compute_pairwise_interaction)(
        i, feature_i, j, feature_j, interaction_values, feature_mapping, feature_names
    )
    for i, feature_i, j, feature_j in pair_tasks
)

# Filter out None results
interaction_results = [r for r in interaction_results if r is not None]

# Step 2: Sort the interactions by their absolute value (descending)
sorted_interactions = sorted(interaction_results, key=lambda x: x[1], reverse=True)

# Step 3: Select the top 30 interactions
top_30_interactions = sorted_interactions[:30]

# Step 4: Display the results
print("Top 30 Feature Interactions (Excluding Self-Interactions & Duplicates):")
for (feature_pair, interaction_value) in top_30_interactions:
    print(f"Interaction ({feature_pair[0]}, {feature_pair[1]}): {interaction_value}")

# Step 5 (Optional): Visualize the top 30 interactions
import matplotlib.pyplot as plt

# Extract feature pairs and their values
feature_pairs = [f"{pair[0]} & {pair[1]}" for pair, _ in top_30_interactions]
values = [value for _, value in top_30_interactions]

# Create a bar plot
plt.figure(figsize=(12, 8))
plt.barh(feature_pairs, values, color='skyblue')
plt.xlabel('Interaction Value')
plt.ylabel('Feature Pairs')
plt.title('Top 30 Feature Interactions (Excluding Duplicates)')
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define data based on the revised table
data = {
    "Feature Combination": [
        "(Marijuana Use, Survey Wave)",
        "(Alcohol Use, Survey Wave)",
        "(Cigarette Use, Survey Wave)",
        "(Race, Survey Wave)",
        "(Driving Frequency, Survey Wave)",
        "(Binge Drinking Frequency, Survey Wave)",
        "(Fun Evenings per Week, Survey Wave)",
        "(Hometown Environment, Survey Wave)",
        "(Average Grade, Survey Wave)",
        "(Work Hours per Week, Survey Wave)",
        "(Dating Frequency, Survey Wave)",
        "(Self-Rated School Ability, Survey Wave) [self_rated_school_ability]",
        "(Political Belief, Survey Wave)",
        "(Mother’s Education, Survey Wave)",
        "(Region, Survey Wave)",
        "(Skipping School, Survey Wave)",
        "(Missing Value, Survey Wave)",
        "(Traffic Tickets in 12 Months, Survey Wave)",
        "(Accidents in 12 Months, Survey Wave)",
        "(Graduate School Prospect, Survey Wave)",
        "(Parental Supervision, Survey Wave)",
        "(Gender, Survey Wave)",
        "(Alcohol Use, Marijuana Use)",
        "(Household Income, Survey Wave)",
        "(Parental Expectations, Survey Wave)",
        "(Parental Monitoring, Survey Wave)",
        "(Self-Esteem, Survey Wave)",
        "(Alcohol Use, Hometown Environment)",
        "(Social Activities, Survey Wave)",
        "(Marijuana Use, Cigarette Use)"
    ],
    "Interaction Value": [
        171.70, 164.23, 93.08, 55.36, 30.74, 24.37, 22.64, 16.94, 15.78, 15.00,
        13.98, 13.92, 13.75, 11.71, 11.62, 9.91, 9.57, 9.42, 8.59, 8.44,
        8.19, 8.14, 6.99, 6.48, 6.09, 5.79, 5.78, 5.72, 5.49, 5.13
    ]
}

# Convert data to a DataFrame
df = pd.DataFrame(data)

# Sort by 'Interaction Value' descending
df = df.sort_values(by="Interaction Value", ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(
    x="Interaction Value",
    y="Feature Combination",
    data=df,
    color="skyblue"
)

plt.title("Top 30 Feature Interactions (Excluding Self-Interactions & Duplicates)")
plt.xlabel("Interaction Value")
plt.ylabel("Feature Combination")
plt.tight_layout()
plt.show()

# %%
import numpy as np
import pandas as pd
import shap
import seaborn as sns
import matplotlib.pyplot as plt

##############################
# 1. Pairwise Aggregation
##############################
def aggregate_shap_interactions_pairwise(interaction_values, feature_names, get_base_feature):
    """
    Aggregates pairwise SHAP interaction values back to their original (pre–one-hot) features,
    and returns a DataFrame of the mean absolute interaction for each (Feature1, Feature2).
    """
    # shap_interaction_values: [n_samples, n_features, n_features]
    # 1) Mean of absolute interaction across samples
    interaction_matrix = np.mean(np.abs(interaction_values), axis=0)  # shape: (n_features, n_features)

    # 2) Map each OHE feature to a base feature
    base_feature_names = [get_base_feature(f) for f in feature_names]
    unique_base_features = sorted(set(base_feature_names))

    # 3) Accumulate pairwise interactions
    #    We'll sum interactions for each pair (bf_i, bf_j)
    #    Because i<j is symmetric in the shap_interaction_values, we avoid double counting.
    aggregated = {}
    for bf_i in unique_base_features:
        aggregated[bf_i] = {}
        for bf_j in unique_base_features:
            aggregated[bf_i][bf_j] = 0.0

    n_features = len(feature_names)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            bf_i = base_feature_names[i]
            bf_j = base_feature_names[j]
            aggregated[bf_i][bf_j] += interaction_matrix[i, j]
            # You could also decide to add symmetrical entries if desired:
            aggregated[bf_j][bf_i] += interaction_matrix[i, j]  # for simpler referencing

    # 4) Convert to a long DataFrame
    records = []
    for bf_i in unique_base_features:
        for bf_j in unique_base_features:
            # Avoid i == j and only keep i < j in the final output (unique pairs)
            if bf_i < bf_j:
                val = aggregated[bf_i][bf_j]
                records.append({
                    "FeatureCombo": f"{bf_i} & {bf_j}",
                    "Feature1": bf_i,
                    "Feature2": bf_j,
                    "InteractionValue": val,
                    "AbsInteraction": abs(val),
                    "Order": 2  # Mark as pairwise
                })

    df = pd.DataFrame(records)
    df.sort_values("AbsInteraction", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

##############################
# 2. "Three-way" Aggregation
##############################
def aggregate_shap_interactions_three_way(interaction_values, feature_names, get_base_feature):
    """
    Naively computes a 'three-way' measure by summing the absolute pairwise interactions
    (i,j), (i,k), and (j,k) for each triple i < j < k.

    NOTE: This does not represent a true SHAP 3-way synergy. It's a proxy 
    by combining the pairwise interactions among the triple.
    """
    # 1) Mean of absolute interaction across samples
    interaction_matrix = np.mean(np.abs(interaction_values), axis=0)  # shape: (n_features, n_features)

    # 2) Map each OHE feature to a base feature
    base_feature_names = [get_base_feature(f) for f in feature_names]
    unique_base_features = sorted(set(base_feature_names))

    # 3) We will group by base features for i,j,k. Because different OHE columns might map
    #    to the same base feature, let's gather all indices that map to each base feature.
    feature_indices_by_base = {}
    for idx, bf in enumerate(base_feature_names):
        if bf not in feature_indices_by_base:
            feature_indices_by_base[bf] = []
        feature_indices_by_base[bf].append(idx)

    # 4) For each triple (bf_i, bf_j, bf_k) with i < j < k, compute sum of pairwise interactions
    records = []
    bf_list = list(unique_base_features)
    n_bf = len(bf_list)

    for i in range(n_bf):
        for j in range(i + 1, n_bf):
            for k in range(j + 1, n_bf):
                bf_i, bf_j, bf_k = bf_list[i], bf_list[j], bf_list[k]

                # For all actual columns that map to bf_i, bf_j, bf_k, sum the relevant pairwise interactions:
                indices_i = feature_indices_by_base[bf_i]
                indices_j = feature_indices_by_base[bf_j]
                indices_k = feature_indices_by_base[bf_k]

                # We'll sum the absolute interaction_matrix across all pairs (i', j'), (i', k'), (j', k')
                total_interaction = 0.0
                for ii in indices_i:
                    for jj in indices_j:
                        total_interaction += interaction_matrix[ii, jj]
                        total_interaction += interaction_matrix[jj, ii]  # symmetrical

                    for kk in indices_k:
                        total_interaction += interaction_matrix[ii, kk]
                        total_interaction += interaction_matrix[kk, ii]

                for jj in indices_j:
                    for kk in indices_k:
                        total_interaction += interaction_matrix[jj, kk]
                        total_interaction += interaction_matrix[kk, jj]

                # We'll keep it as the approximate measure
                records.append({
                    "FeatureCombo": f"{bf_i} & {bf_j} & {bf_k}",
                    "Feature1": bf_i,
                    "Feature2": bf_j,
                    "Feature3": bf_k,
                    "InteractionValue": total_interaction,
                    "AbsInteraction": abs(total_interaction),
                    "Order": 3  # Mark as three-way
                })

    df = pd.DataFrame(records)
    df.sort_values("AbsInteraction", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


##############################
# 3. Putting it all together
##############################

# Suppose you already have:
#   shap_interaction_values = explainer.shap_interaction_values(X_preprocessed)
#   feature_names = loaded_gbt.named_steps['preprocessor'].get_feature_names_out()
#   We'll reuse your simple_get_base_feature function:
import re

def simple_get_base_feature(name):
    """
    Strips off numerical one-hot encoding suffixes (e.g., 'survey_wave_2021' → 'survey_wave').
    """
    # Remove 'num__' or 'cat__' prefix if present
    if name.startswith("num__"):
        name = name[len("num__"):]
    if name.startswith("cat__"):
        name = name[len("cat__"):]

    # Remove one-hot encoding suffix (_YYYY or _X.X for categorical variables)
    name = re.sub(r"_\d+(\.\d+)?$", "", name)  # Removes _2021, _2.0, etc.

    return name

# -- 3.1 Compute pairwise and "three-way" aggregates --
df_pairwise = aggregate_shap_interactions_pairwise(
    interaction_values, 
    feature_names, 
    get_base_feature=simple_get_base_feature
)

df_three_way = aggregate_shap_interactions_three_way(
    interaction_values, 
    feature_names, 
    get_base_feature=simple_get_base_feature
)

# -- 3.2 Combine them into one table --
#    For consistency, we can unify the column names
df_pairwise["Feature3"] = None  # no third feature for pairs
combined_df = pd.concat([df_pairwise, df_three_way], ignore_index=True)

# Sort by absolute interaction
combined_df.sort_values("AbsInteraction", ascending=False, inplace=True)
combined_df.reset_index(drop=True, inplace=True)

# -- 3.3 Pick top 20 overall --
df_top_20 = combined_df.head(20).copy()

# For a nicer label that works for both pairwise and triple combos:
df_top_20["Label"] = df_top_20["FeatureCombo"]

# -- 3.4 Plot a bar chart of the top 20 combos --
plt.figure(figsize=(10, 6))
sns.barplot(data=df_top_20, y="Label", x="AbsInteraction", color="royalblue")
plt.title("Top 20 Interactions (Pairwise + Three-way) by Absolute SHAP Value")
plt.xlabel("Absolute SHAP Interaction Value")
plt.ylabel("Feature Combination")
plt.tight_layout()
plt.show()
# You now have a bar plot of the top 20 interactions (including both 2- and 3-feature combos).

# %%
df_top_20

