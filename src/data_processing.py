"""
Data Processing Module

Functions for loading, cleaning, and preprocessing survey data for ML analysis.

Key functions:
    - load_raw_data: Load raw TSV survey files
    - preprocess_features: Clean and engineer features
    - handle_missing_data: Imputation and missing indicators
    - create_train_test_split: Stratified splitting
    - prepare_features: Encoding and scaling

Author: Siyang Ni
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_processed_data(filepath: str) -> pd.DataFrame:
    """
    Load preprocessed data from CSV file.

    Args:
        filepath: Path to processed CSV file

    Returns:
        DataFrame with preprocessed survey data

    Raises:
        FileNotFoundError: If data file doesn't exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            "Please ensure MTF data has been preprocessed using R scripts."
        )

    df = pd.read_csv(filepath)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def identify_column_types(df: pd.DataFrame,
                          target_col: str,
                          max_unique_categorical: int = 20) -> Tuple[List[str], List[str]]:
    """
    Automatically identify categorical and numerical columns.

    Args:
        df: Input DataFrame
        target_col: Name of target variable to exclude
        max_unique_categorical: Maximum unique values to consider categorical

    Returns:
        Tuple of (categorical_columns, numerical_columns)
    """
    categorical_cols = []
    numerical_cols = []

    for col in df.columns:
        if col == target_col:
            continue

        if df[col].dtype == 'object' or df[col].nunique() <= max_unique_categorical:
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)

    print(f"Identified {len(categorical_cols)} categorical and {len(numerical_cols)} numerical features")
    return categorical_cols, numerical_cols


def create_preprocessing_pipeline(categorical_cols: List[str],
                                   numerical_cols: List[str]) -> ColumnTransformer:
    """
    Create sklearn preprocessing pipeline.

    Args:
        categorical_cols: List of categorical column names
        numerical_cols: List of numerical column names

    Returns:
        ColumnTransformer with encoding and scaling steps
    """
    # Categorical preprocessing: impute + one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Numerical preprocessing: impute + scale
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols),
            ('num', numerical_transformer, numerical_cols)
        ],
        remainder='passthrough'
    )

    return preprocessor


def create_train_test_split_stratified(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create stratified train-test split.

    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Proportion of test set (default: 0.2)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Target distribution (train): {y_train.value_counts(normalize=True).to_dict()}")

    return X_train, X_test, y_train, y_test


def get_feature_names_after_preprocessing(preprocessor: ColumnTransformer,
                                          categorical_cols: List[str],
                                          numerical_cols: List[str]) -> List[str]:
    """
    Extract feature names after one-hot encoding.

    Args:
        preprocessor: Fitted ColumnTransformer
        categorical_cols: Original categorical column names
        numerical_cols: Original numerical column names

    Returns:
        List of feature names including one-hot encoded columns
    """
    # Get categorical feature names after one-hot encoding
    try:
        cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
    except:
        cat_features = []

    # Combine with numerical feature names
    feature_names = list(cat_features) + numerical_cols

    return feature_names


def encode_target_variable(y: pd.Series) -> Tuple[pd.Series, Dict[int, str]]:
    """
    Encode target variable for classification.

    Args:
        y: Target Series

    Returns:
        Tuple of (encoded_target, label_mapping)
    """
    unique_values = sorted(y.unique())
    label_mapping = {i: str(val) for i, val in enumerate(unique_values)}

    # Create reverse mapping for encoding
    reverse_mapping = {val: i for i, val in label_mapping.items()}
    y_encoded = y.map(reverse_mapping)

    print(f"Target encoding: {label_mapping}")
    return y_encoded, label_mapping


def load_and_prepare_data(
    filepath: str,
    target_col: str,
    drop_cols: Optional[List[str]] = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> Dict:
    """
    Complete data loading and preprocessing pipeline.

    Args:
        filepath: Path to processed CSV
        target_col: Name of target variable
        drop_cols: Columns to drop (e.g., IDs)
        test_size: Test set proportion
        random_state: Random seed

    Returns:
        Dictionary containing:
            - X_train, X_test, y_train, y_test
            - preprocessor (fitted)
            - feature_names
            - label_mapping
    """
    # Load data
    df = load_processed_data(filepath)

    # Drop unnecessary columns
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode target
    y_encoded, label_mapping = encode_target_variable(y)

    # Identify column types
    categorical_cols, numerical_cols = identify_column_types(X, target_col)

    # Create train-test split
    X_train, X_test, y_train, y_test = create_train_test_split_stratified(
        X, y_encoded, test_size, random_state
    )

    # Create and fit preprocessor
    preprocessor = create_preprocessing_pipeline(categorical_cols, numerical_cols)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Get feature names
    feature_names = get_feature_names_after_preprocessing(
        preprocessor, categorical_cols, numerical_cols
    )

    return {
        'X_train': X_train_processed,
        'X_test': X_test_processed,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'label_mapping': label_mapping,
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols
    }


if __name__ == "__main__":
    # Example usage
    data_path = os.path.expanduser('~/work/vaping_project_data/processed_data_g12n.csv')
    data_dict = load_and_prepare_data(
        filepath=data_path,
        target_col='nicotine12d',  # Adjust based on actual target column
        drop_cols=['id', 'V1']  # Adjust based on actual ID columns
    )
    print(f"Preprocessing complete. Training set shape: {data_dict['X_train'].shape}")
