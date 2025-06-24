"""Dataset profiling package for comprehensive data analysis and visualization.

This package provides tools for analyzing datasets and generating interactive HTML reports
with detailed statistics and visualizations. It supports various data types and file formats,
and includes features for text analysis with word statistics and LLM tokenization.

The package consists of several modules:
- io: Functions for loading data from various file formats and parsing configuration files
- stats: Functions for calculating statistics for DataFrames and their columns
- report: Functions for generating HTML reports with analyses and visualizations
- text_analysis: Functions for analyzing text data with word statistics and LLM tokenization
- visualize: Functions for creating various visualizations for the report
- cli: Command-line interface for the package

The main entry point is the generate_profile_report function, which analyzes a DataFrame
and generates a comprehensive HTML report.
"""

from .io import load_data, parse_yaml_config
from .stats import analyze_column, get_basic_stats
from .report import create_html_report, generate_profile_report
from .text_analysis import analyze_text_with_llm_tokenizer, analyze_text_with_countvectorizer
from .visualize import (
    create_box_plot,
    create_bar_chart,
    create_histogram,
    create_wordcloud,
    create_time_series,
    create_data_types_chart,
    create_sample_data_table,
    create_value_counts_table,
    create_missing_values_chart,
)

__version__ = "0.1.0"

__all__ = [
    "load_data",
    "parse_yaml_config",
    "get_basic_stats",
    "analyze_column",
    "create_histogram",
    "create_bar_chart",
    "create_box_plot",
    "create_time_series",
    "create_missing_values_chart",
    "create_data_types_chart",
    "create_wordcloud",
    "create_value_counts_table",
    "create_sample_data_table",
    "analyze_text_with_countvectorizer",
    "analyze_text_with_llm_tokenizer",
    "create_html_report",
    "generate_profile_report",
]
