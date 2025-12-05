"""
ETL Package for Data Warehouse Project
Contains Extract, Transform, Load modules
"""

__version__ = "1.0.0"
__author__ = "PT"

# Import main functions for easier access
from .Extracting import extract_data
from .transform import transform_data
from .Loading import run_loading_pipeline

__all__ = [
    'extract_data',
    'transform_data', 
    'run_loading_pipeline'
]