import pandas as pd
import yaml
from pandas import CategoricalDtype
import numpy as np
import os


def parse_yaml_config(config_file):
    """
    Parse a YAML configuration file.

    Args:
        config_file (str): Path to the YAML configuration file

    Returns:
        dict: Configuration parameters
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required fields
    if 'input' not in config:
        raise ValueError("YAML configuration must include 'input' field")

    # Set default values for optional fields
    if 'output' not in config:
        config['output'] = 'profile.html'
    if 'title' not in config:
        config['title'] = 'DataFrame Characterization Report'
    if 'primary_color' not in config:
        config['primary_color'] = '#2196f3'

    # Handle LLM-related fields
    if 'llm_models' in config:
        if not isinstance(config['llm_models'], dict):
            raise ValueError("'llm_models' must be a dictionary of model configurations")
        for model_name, model_config in config['llm_models'].items():
            if not isinstance(model_config, dict):
                raise ValueError(f"Configuration for model '{model_name}' must be a dictionary")
            model_config.setdefault('token', None)
            model_config.setdefault('input_cost', None)
    else:
        # For backward compatibility, create llm_models from legacy parameters
        config['llm_models'] = {}
        config.setdefault('llm_model', None)
        config.setdefault('llm_token', None)
        config.setdefault('llm_input_cost', None)
        if config['llm_model']:
            config['llm_models'][config['llm_model']] = {
                'token': config['llm_token'],
                'input_cost': config['llm_input_cost']
            }

    return config


def load_data(file_path, schema=None):
    """
    Load data from various file formats based on file extension.
    If schema is provided, all data is initially loaded as strings and then converted.

    Args:
        file_path (str): Path to the input file
        schema (dict, optional): Custom schema mapping column names to data types

    Returns:
        pandas.DataFrame: Loaded DataFrame
    """
    file_ext = os.path.splitext(file_path)[1].lower()

    if schema:
        dtype_map = {str: 'str'} if schema else None
        if file_ext == '.csv':
            df = pd.read_csv(file_path, dtype=str)
        elif file_ext in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path, dtype=str)
        elif file_ext == '.json':
            df = pd.read_json(file_path, dtype=str)
        elif file_ext == '.parquet':
            df = pd.read_parquet(file_path)
            for col in df.columns:
                df[col] = df[col].astype(str)
        elif file_ext in ['.pickle', '.pkl']:
            df = pd.read_pickle(file_path)
            for col in df.columns:
                df[col] = df[col].astype(str)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        for column, dtype in schema.items():
            if column in df.columns:
                try:
                    dtype_lower = dtype.lower()
                    if dtype_lower in ('int', 'integer'):
                        df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
                    elif dtype_lower in ('float', 'numeric'):
                        df[column] = pd.to_numeric(df[column], errors='coerce')
                    elif dtype_lower in ('bool', 'boolean'):
                        df[column] = df[column].map({'True': True, 'true': True, '1': True,
                                                    'False': False, 'false': False, '0': False}).astype(bool)
                    elif dtype_lower in ('date', 'datetime'):
                        df[column] = pd.to_datetime(df[column], errors='coerce')
                    elif dtype_lower == 'category':
                        df[column] = df[column].astype('category')
                except Exception as e:
                    print(f"Warning: Could not convert column '{column}' to {dtype}: {str(e)}")
        return df

    # Load data normally without schema
    if file_ext == '.csv':
        return pd.read_csv(file_path)
    elif file_ext in ['.xls', '.xlsx']:
        return pd.read_excel(file_path)
    elif file_ext == '.json':
        return pd.read_json(file_path)
    elif file_ext == '.parquet':
        return pd.read_parquet(file_path)
    elif file_ext in ['.pickle', '.pkl']:
        return pd.read_pickle(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")