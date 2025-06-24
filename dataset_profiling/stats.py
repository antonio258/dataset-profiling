"""Statistics module for dataset profiling.

This module provides functions to calculate various statistics for DataFrames and their columns,
including basic statistics, type-specific statistics, and text analysis with LLM tokenization.
"""

import logging

import pandas as pd

from .text_analysis import analyze_text_with_llm_tokenizer, analyze_text_with_countvectorizer

# Get the module logger
logger = logging.getLogger("dataset_profiling.stats")


def get_basic_stats(df):
    """Calculate basic statistics for the DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to analyze

    Returns:
        dict: Dictionary containing basic statistics including row count, column count,
            missing cells, duplicate rows, memory usage, and data type distribution
    """
    stats = {
        "rows": len(df),
        "columns": len(df.columns),
        "total_cells": df.size,
        "missing_cells": df.isna().sum().sum(),
        "missing_percent": round(df.isna().sum().sum() / df.size * 100, 2) if df.size > 0 else 0,
        "duplicate_rows": df.duplicated().sum(),
        "duplicate_percent": round(df.duplicated().sum() / len(df) * 100, 2) if len(df) > 0 else 0,
        "memory_usage": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),  # MB
        "dtypes": {str(k): int(v) for k, v in df.dtypes.value_counts().to_dict().items()},
    }
    return stats


def analyze_column(df, column, llm_models=None, llm_model=None, llm_token=None, llm_input_cost=None):
    """Perform detailed analysis of a single column.

    Analyzes a column from the DataFrame and calculates various statistics based on its data type.
    For text columns, performs additional analysis including word counts and LLM tokenization.
    Tokenizes once using llm_model, then calculates costs for all models in llm_models.

    Args:
        df (pandas.DataFrame): The DataFrame containing the column to analyze
        column (str): The name of the column to analyze
        llm_models (dict, optional): Dictionary mapping model names to their configurations. Defaults to None.
        llm_model (str, optional): The name of the model to use for tokenization. Defaults to None.
        llm_token (str, optional): The token to use for authentication with the LLM API. Defaults to None.
        llm_input_cost (float, optional): The cost per token for the LLM model. Defaults to None.

    Returns:
        dict: Dictionary containing analysis results including data type, statistics,
            and for text columns, word and token statistics
    """
    if llm_models is None:
        llm_models = {}
        if llm_model:
            llm_models[llm_model] = {"token": llm_token, "input_cost": llm_input_cost}

    series = df[column]
    dtype = str(series.dtype)
    is_numeric = pd.api.types.is_numeric_dtype(series)
    is_datetime = pd.api.types.is_datetime64_any_dtype(series)

    # Improved categorical/text detection from original script
    non_null_series = series.dropna()
    unique_count = series.nunique()
    is_categorical = (
        isinstance(series.dtype, pd.CategoricalDtype)
        or (unique_count / len(non_null_series) < 0.05 and unique_count < 50)
        if len(non_null_series) > 0
        else True
    )

    is_text = False
    if "object" in dtype and not is_categorical:
        if len(non_null_series) > 0 and all(isinstance(x, str) for x in non_null_series):
            avg_length = non_null_series.str.len().mean()
            if avg_length > 15:  # Heuristic for identifying text columns
                is_text = True

    analysis = {
        "name": column,
        "dtype": dtype,
        "count": len(series),
        "missing": series.isna().sum(),
        "missing_percent": round(series.isna().sum() / len(series) * 100, 2) if len(series) > 0 else 0,
        "unique": unique_count,
        "unique_percent": round(unique_count / len(series) * 100, 2) if len(series) > 0 else 0,
        "is_numeric": is_numeric,
        "is_datetime": is_datetime,
        "is_categorical": is_categorical,
        "is_text": is_text,
    }

    # Add type-specific stats
    if is_numeric:
        analysis.update(
            {
                "min": float(series.min()) if series.notna().any() else None,
                "max": float(series.max()) if series.notna().any() else None,
                "mean": float(series.mean()) if series.notna().any() else None,
                "median": float(series.median()) if series.notna().any() else None,
                "std": float(series.std()) if series.notna().any() else None,
            },
        )
    elif is_datetime:
        analysis.update(
            {
                "min": series.min().strftime("%Y-%m-%d %H:%M:%S") if series.notna().any() else None,
                "max": series.max().strftime("%Y-%m-%d %H:%M:%S") if series.notna().any() else None,
                "range_days": (series.max() - series.min()).days if series.notna().any() else None,
            },
        )
    elif "object" in dtype or is_categorical:
        if len(non_null_series) > 0 and all(isinstance(x, str) for x in non_null_series):
            analysis.update(
                {
                    "avg_length": round(non_null_series.str.len().mean(), 2),
                    "min_length": non_null_series.str.len().min(),
                    "max_length": non_null_series.str.len().max(),
                },
            )

    # --- Corrected Text and LLM Analysis Logic ---
    if is_text:
        # Word analysis with CountVectorizer
        text_stats = analyze_text_with_countvectorizer(series)
        analysis.update(text_stats)

        # LLM analysis: Tokenize ONCE, calculate costs for all.
        tokenization_results = None
        # The 'llm_model' from config is the designated tokenizer.
        base_tokenizer_model = llm_model

        if base_tokenizer_model and llm_models:
            # 1. Tokenize using the base model
            base_model_config = llm_models.get(base_tokenizer_model, {})
            tokenization_results = analyze_text_with_llm_tokenizer(
                series,
                base_tokenizer_model,
                base_model_config.get("token"),
            )

            # Add base tokenization stats to the top level for the legacy report section
            analysis.update(tokenization_results)
            # Add the model name to identify which model was used for tokenization
            analysis["model_name"] = base_tokenizer_model
            # Add cost for the base model itself to the top level
            base_input_cost = base_model_config.get("input_cost")
            if base_input_cost is not None and tokenization_results.get("total_tokens", 0) > 0:
                total_cost = tokenization_results["total_tokens"] * float(base_input_cost)
                analysis.update(
                    {
                        "total_token_cost": round(total_cost, 4),
                        "avg_cost_per_doc": (
                            round(total_cost / len(non_null_series), 4) if len(non_null_series) > 0 else 0
                        ),
                    },
                )

        if llm_models and tokenization_results:
            analysis["model_token_stats"] = {}

            # 2. Loop through all models to calculate costs using the stored token counts
            for model_name, model_config in llm_models.items():
                model_stats = tokenization_results.copy()  # Reuse the same token counts

                input_cost = model_config.get("input_cost")
                if input_cost is not None and model_stats.get("total_tokens", 0) > 0:
                    try:
                        total_cost = model_stats["total_tokens"] * float(input_cost)
                        model_stats.update(
                            {
                                "total_token_cost": round(total_cost, 4),
                                "avg_cost_per_doc": (
                                    round(total_cost / len(non_null_series), 4) if len(non_null_series) > 0 else 0
                                ),
                            },
                        )
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid input cost for model '{model_name}'.")

                analysis["model_token_stats"][model_name] = model_stats

    return analysis
