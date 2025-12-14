"""
Shared configuration module for vaping project modeling scripts.

This module centralizes path management and constants used across
training and analysis scripts, with support for environment variable
overrides to enable running in different environments (local, cluster, etc.).

Environment Variables:
    VAPING_DATA_PATH: Override for input data CSV path
    VAPING_MODELS_DIR: Override for model artifacts directory
    TUNING_PRESET: fast|standard|max - controls hyperparameter search budget
"""

import os
from pathlib import Path

# ================
# PATH CONFIGURATION
# ================

# Input data path - can be overridden via VAPING_DATA_PATH environment variable
DEFAULT_DATA_PATH = os.path.expanduser('~/autodl-tmp/processed_data_g12nn.csv')
DATA_PATH = os.environ.get('VAPING_DATA_PATH', DEFAULT_DATA_PATH)

# Model artifacts directory - can be overridden via VAPING_MODELS_DIR environment variable
DEFAULT_MODELS_DIR = os.path.expanduser('~/work/vaping_project_data')
MODELS_DIR = os.environ.get('VAPING_MODELS_DIR', DEFAULT_MODELS_DIR)

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# ================
# MODEL CONSTANTS
# ================

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS_CV = 5
SCORING_METRIC = 'roc_auc'
VERBOSE = 1

CPU_COUNT = os.cpu_count()

# ================
# TUNING PRESETS
# ================

TUNING_PRESET = os.environ.get("TUNING_PRESET", "standard").strip().lower()


def get_tuning_iters(preset: str = None) -> dict:
    """
    Get hyperparameter tuning budgets per model.
    
    Parameters
    ----------
    preset : str, optional
        One of 'fast', 'standard', 'max'. If None, uses TUNING_PRESET from env.
    
    Returns
    -------
    dict
        Mapping of model names to iteration counts
    """
    if preset is None:
        preset = TUNING_PRESET
    
    if preset == "fast":
        return {
            "elasticnet": 35,
            "elasticnet_interactions_deg2": 25,
            "lasso_interactions_deg23": 35,
            "rf": 35,
            "gbt": 35,
            "hgbt": 35,
            "xgb": 35,
            "catboost": 30,
        }
    elif preset == "max":
        return {
            "elasticnet": 120,
            "elasticnet_interactions_deg2": 80,
            "lasso_interactions_deg23": 120,
            "rf": 120,
            "gbt": 120,
            "hgbt": 120,
            "xgb": 120,
            "catboost": 100,
        }
    else:  # default: standard
        return {
            "elasticnet": 60,
            "elasticnet_interactions_deg2": 40,
            "lasso_interactions_deg23": 60,
            "rf": 60,
            "gbt": 60,
            "hgbt": 60,
            "xgb": 60,
            "catboost": 60,
        }


# ================
# PATH HELPERS
# ================

def get_model_path(model_name: str) -> str:
    """
    Get the full path for a saved model file.
    
    Parameters
    ----------
    model_name : str
        Model identifier (e.g., 'lasso', 'rf', 'xgb')
    
    Returns
    -------
    str
        Full path to the model joblib file
    """
    return os.path.join(MODELS_DIR, f'best_{model_name}_model.joblib')


def get_preprocessed_data_path() -> str:
    """Get the path to saved preprocessed data."""
    return os.path.join(MODELS_DIR, 'preprocessed_data.joblib')


def get_shap_output_dir(model_name: str = None) -> str:
    """
    Get directory for SHAP analysis outputs.
    
    Parameters
    ----------
    model_name : str, optional
        If provided, creates a subdirectory for this model
    
    Returns
    -------
    str
        Path to SHAP outputs directory
    """
    if model_name:
        path = os.path.join(MODELS_DIR, f'shap_{model_name}')
    else:
        path = os.path.join(MODELS_DIR, 'shap_outputs')
    os.makedirs(path, exist_ok=True)
    return path


# ================
# FEATURE CONFIGURATION
# ================

# Feature renaming map - centralizing this for consistency
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

TARGET_COL = 'nicotine_use_12m'
