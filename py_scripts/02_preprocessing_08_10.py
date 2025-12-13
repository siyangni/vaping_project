#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
import sys

from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score, RandomizedSearchCV, RepeatedStratifiedKFold
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    HistGradientBoostingClassifier
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Interpretability
import shap
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

# Optimization
import optuna


# In[ ]:


# ================
# 2. CONFIGURATION
# ================
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS_CV = 5
SCORING_METRIC = 'roc_auc'
VERBOSE = 1

CPU_COUNT = os.cpu_count()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# # Preprocessing

# In[ ]:


# ================
# 3. Import core waves (including 2024)
# ================

# Specify the directory path (auto-detect WSL/Linux vs Windows UNC)
directory_candidates = [
    r"\\wsl.localhost\Ubuntu\home\siyang\work\vaping_project_data\original_all_core",
    os.path.expanduser('~/work/vaping_project_data/original_all_core'),
    '/home/siyang/work/vaping_project_data/original_all_core',
]

directory_path = next((p for p in directory_candidates if os.path.exists(p)), None)
if directory_path is None:
    raise FileNotFoundError(
        "Could not find data directory. Tried: " + ", ".join(directory_candidates)
    )

# List all files ending with '0810.tsv' in the specified directory
files = [f for f in os.listdir(directory_path) if f.endswith('0810.tsv')]

# Ensure 2024 wave is included even if it wasn't picked up by directory listing
extra_file_candidates = [
    # User-provided WSL UNC path (useful when running from Windows/Jupyter that can see WSL via UNC)
    r"\\wsl.localhost\Ubuntu\home\siyang\work\vaping_project_data\original_all_core\original_core_2024_0810.tsv",
    # Native WSL/Linux path (useful when running inside WSL)
    "/home/siyang/work/vaping_project_data/original_all_core/original_core_2024_0810.tsv",
]

extra_file_path = next((p for p in extra_file_candidates if os.path.exists(p)), None)
if extra_file_path is None:
    warnings.warn(
        "Could not find original_core_2024_0810.tsv at expected paths. "
        "Proceeding with files discovered via directory listing."
    )
else:
    extra_file_name = os.path.basename(extra_file_path)
    if extra_file_name not in files:
        files.append(extra_file_name)

# Create a dictionary to store individual dataframes
dataframes = {}

# Read each file into a separate dataframe
for file in sorted(files):
    # If this is the 2024 file, prefer the explicit path we resolved above
    if file == 'original_core_2024_0810.tsv' and extra_file_path is not None:
        file_path = extra_file_path
    else:
        file_path = os.path.join(directory_path, file)

    try:
        # Use the file name (without extension) as the key
        df_name = file.replace('.tsv', '')  # Remove .tsv from the filename
        # Read the file with low_memory=False to handle mixed types
        dataframes[df_name] = pd.read_csv(file_path, sep='\t', low_memory=False)
        print(f"Successfully read: {file} into dataframe '{df_name}'")
    except Exception as e:
        print(f"Error reading {file}: {e}")

# Example: Accessing a specific dataframe
# df_2024 = dataframes['original_core_2024_0810']


# In[ ]:


# ================
# 4. Basic Info of each dataset
# ================

# Loop through each dataframe in the dictionary
for df_name, df in dataframes.items():
    print(f"=== Basic Information for {df_name} ===")
    
    # Display the first few rows
    print("\nFirst 5 Rows:")
    print(df.head())
    
    # Display the last few rows
    print("\nLast 5 Rows:")
    print(df.tail())
    
    # Get dataset shape
    print(f"\nDataset Shape: {df.shape}")
    
    # Get column names
    print(f"\nColumn Names: {df.columns.tolist()}")
    
    # Get data types
    print(f"\nData Types:\n{df.dtypes}")
    
    # Check for missing values
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    
    # Get summary statistics for numerical columns
    print(f"\nSummary Statistics:\n{df.describe(include='all')}")
    
    # Count unique values in each column
    print(f"\nUnique Values per Column:\n{df.nunique()}")
    
    # Check for duplicate rows
    print(f"\nNumber of Duplicate Rows: {df.duplicated().sum()}")
    
    # Print a separator for readability
    print("\n" + "=" * 50 + "\n")


# In[19]:


# ================
# 5. Inner Join all 7 waves
# ================

# Step 1: Find common columns across all dataframes
common_columns = set(dataframes[next(iter(dataframes))].columns)  # Initialize with columns from the first dataframe
for df in dataframes.values():
    common_columns.intersection_update(df.columns)  # Keep only columns present in all dataframes

# Convert the set of common columns to a list
common_columns = list(common_columns)
print(f"Common Columns: {common_columns}")

# Step 2: Filter each dataframe to keep only the common columns
filtered_dataframes = {}
for df_name, df in dataframes.items():
    filtered_dataframes[df_name] = df[common_columns]
    print(f"Filtered {df_name} to keep common columns.")

# Step 3: Concatenate all filtered dataframes into a single dataframe
merged_df = pd.concat(filtered_dataframes.values(), ignore_index=True)

# Display basic info of the merged dataframe
# Get dataset shape
print(f"\nDataset Shape: {df.shape}")
    
# Get column names
print(f"\nColumn Names: {df.columns.tolist()}")
    
# Get data types
print(f"\nData Types:\n{df.dtypes}")
    
# Check for missing values
print(f"\nMissing Values:\n{df.isnull().sum()}")
    
# Get summary statistics for numerical columns
print(f"\nSummary Statistics:\n{df.describe(include='all')}")
    
# Count unique values in each column
print(f"\nUnique Values per Column:\n{df.nunique()}")
    
# Check for duplicate rows
print(f"\nNumber of Duplicate Rows: {df.duplicated().sum()}")
    
# Print a separator for readability
print("\n" + "=" * 50 + "\n")

