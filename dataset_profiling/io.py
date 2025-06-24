"""Input/output module for dataset profiling.

This module provides functions for loading data from various file formats
and parsing configuration files. It supports loading data with custom schemas
and handles different file types including CSV, Excel, JSON, Parquet, and Pickle.
"""

import os
import logging

import yaml
import pandas as pd

# Get the module logger
logger = logging.getLogger("dataset_profiling.io")


def parse_yaml_config(config_file):
    """Parse a YAML configuration file for dataset profiling.

    Reads and validates a YAML configuration file containing parameters for
    dataset profiling. The configuration must include an 'input' field specifying
    the data file path. Other fields like 'output', 'title', and 'primary_color'
    are optional and will be set to default values if not provided.

    The function also handles LLM-related configurations, supporting both a dictionary
    of model configurations and legacy single-model parameters.

    Args:
        config_file (str): Path to the YAML configuration file

    Returns:
        dict: Configuration parameters including input file path, output file path,
            report title, primary color, and LLM model configurations

    Raises:
        ValueError: If the configuration is missing required fields or has invalid values
    """
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Validate required fields
    if "input" not in config:
        raise ValueError("YAML configuration must include 'input' field")

    # Set default values for optional fields
    if "output" not in config:
        config["output"] = "profile.html"
    if "title" not in config:
        config["title"] = "DataFrame Characterization Report"
    if "primary_color" not in config:
        config["primary_color"] = "#2196f3"

    # Handle LLM-related fields
    if "llm_models" in config:
        if not isinstance(config["llm_models"], dict):
            raise ValueError("'llm_models' must be a dictionary of model configurations")
        for model_name, model_config in config["llm_models"].items():
            if not isinstance(model_config, dict):
                raise ValueError(f"Configuration for model '{model_name}' must be a dictionary")
            model_config.setdefault("token", None)
            model_config.setdefault("input_cost", None)
    else:
        # For backward compatibility, create llm_models from legacy parameters
        config["llm_models"] = {}
        config.setdefault("llm_model", None)
        config.setdefault("llm_token", None)
        config.setdefault("llm_input_cost", None)
        if config["llm_model"]:
            config["llm_models"][config["llm_model"]] = {
                "token": config["llm_token"],
                "input_cost": config["llm_input_cost"],
            }

    return config


def _load_data_as_strings(file_path, file_ext):
    """Load data from a file with all columns as strings.

    Args:
        file_path (str): Path to the input file
        file_ext (str): File extension (lowercase, with dot)

    Returns:
        pandas.DataFrame: DataFrame with all columns as strings

    Raises:
        ValueError: If the file format is not supported
    """
    if file_ext == ".csv":
        return pd.read_csv(file_path, dtype=str)
    if file_ext in [".xls", ".xlsx"]:
        return pd.read_excel(file_path, dtype=str)
    if file_ext == ".json":
        return pd.read_json(file_path, dtype=str)
    if file_ext == ".parquet":
        df = pd.read_parquet(file_path)
        for col in df.columns:
            df[col] = df[col].astype(str)
        return df
    if file_ext in [".pickle", ".pkl"]:
        df = pd.read_pickle(file_path)
        for col in df.columns:
            df[col] = df[col].astype(str)
        return df
    raise ValueError(f"Unsupported file format: {file_ext}")


def _load_data_normal(file_path, file_ext):
    """Load data from a file with default data types.

    Args:
        file_path (str): Path to the input file
        file_ext (str): File extension (lowercase, with dot)

    Returns:
        pandas.DataFrame: DataFrame with default data types

    Raises:
        ValueError: If the file format is not supported
    """
    if file_ext == ".csv":
        return pd.read_csv(file_path)
    if file_ext in [".xls", ".xlsx"]:
        return pd.read_excel(file_path)
    if file_ext == ".json":
        return pd.read_json(file_path)
    if file_ext == ".parquet":
        return pd.read_parquet(file_path)
    if file_ext in [".pickle", ".pkl"]:
        return pd.read_pickle(file_path)
    raise ValueError(f"Unsupported file format: {file_ext}")


def _apply_schema(df, schema):
    """Apply a schema to a DataFrame, converting columns to specified data types.

    Args:
        df (pandas.DataFrame): DataFrame to convert
        schema (dict): Schema mapping column names to data types

    Returns:
        pandas.DataFrame: DataFrame with columns converted to specified data types
    """
    for column, dtype in schema.items():
        if column in df.columns:
            try:
                dtype_lower = dtype.lower()
                if dtype_lower in ("int", "integer"):
                    df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
                elif dtype_lower in ("float", "numeric"):
                    df[column] = pd.to_numeric(df[column], errors="coerce")
                elif dtype_lower in ("bool", "boolean"):
                    df[column] = (
                        df[column]
                        .map({"True": True, "true": True, "1": True, "False": False, "false": False, "0": False})
                        .astype(bool)
                    )
                elif dtype_lower in ("date", "datetime"):
                    df[column] = pd.to_datetime(df[column], errors="coerce")
                elif dtype_lower == "category":
                    df[column] = df[column].astype("category")
            except Exception as e:
                logger.warning(f"Could not convert column '{column}' to {dtype}: {str(e)}")
    return df


def load_data(file_path, schema=None):
    """Load data from various file formats based on file extension.

    Loads data from different file formats including CSV, Excel, JSON, Parquet, and Pickle.
    The file format is determined by the file extension. If a schema is provided,
    the function will attempt to convert columns to the specified data types.

    When a schema is provided, all data is initially loaded as strings and then
    converted to the specified types. This helps handle type conversion issues
    that might occur with direct loading.

    Args:
        file_path (str): Path to the input file
        schema (dict, optional): Custom schema mapping column names to data types.
            Supported types include: 'int'/'integer', 'float'/'numeric', 'bool'/'boolean',
            'date'/'datetime', and 'category'. Defaults to None.

    Returns:
        pandas.DataFrame: Loaded DataFrame with data types converted according to schema if provided

    Raises:
        ValueError: If the file format is not supported
    """
    file_ext = os.path.splitext(file_path)[1].lower()

    if schema:
        df = _load_data_as_strings(file_path, file_ext)
        return _apply_schema(df, schema)
    return _load_data_normal(file_path, file_ext)
