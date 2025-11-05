"""
Vaping Project: Machine Learning Analysis Package

This package contains reusable modules for analyzing nicotine vaping trends
using machine learning classifiers.

Modules:
    - data_processing: Data loading, preprocessing, and feature engineering
    - modeling: Machine learning model implementations and training
    - interpretability: SHAP analysis and feature importance
    - visualization: Plotting functions for publication figures

Author: Siyang Ni
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Siyang Ni"

from . import data_processing
from . import modeling
from . import interpretability
from . import visualization

__all__ = [
    "data_processing",
    "modeling",
    "interpretability",
    "visualization",
]
