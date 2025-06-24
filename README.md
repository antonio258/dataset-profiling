# Dataset Profiling

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://raw.githubusercontent.com/antonio258/dataset-profiling/main/interrogate_badge.svg)](https://github.com/antonio258p/dataset-profiling)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A powerful Python package for comprehensive dataset analysis and visualization. Generate interactive HTML reports with detailed statistics, visualizations, and insights for your datasets.

## Features

- **Comprehensive Data Analysis**: Analyze datasets with detailed statistics for each column
- **Interactive HTML Reports**: Generate beautiful, interactive HTML reports with visualizations
- **Multiple Data Types Support**: Handle numeric, categorical, datetime, and text data types
- **Text Analysis**: Analyze text data with word statistics and word clouds
- **LLM Token Analysis**: Calculate token counts and costs for LLM models (with optional transformers dependency)
- **Flexible Input Formats**: Support for CSV, Excel, JSON, Parquet, and Pickle file formats
- **Custom Schema Support**: Define custom schemas for data type conversion
- **Command-line Interface**: Easy-to-use CLI for quick analysis
- **Programmatic API**: Use as a Python library in your own code

## Installation

### Basic Installation

```bash
pip install dataset-profiling
```

### With Transformers Support (for LLM token analysis)

```bash
pip install "dataset-profiling[transformers]"
```

### Development Installation

```bash
git clone https://github.com/antonio258p/dataset-profiling.git
cd dataset-profiling
pip install -e ".[dev,transformers]"
```

## Usage

### Command Line Interface

Generate a basic report:

```bash
dataset-profiling --input data.csv --output report.html
```

Use a configuration file for advanced options:

```bash
dataset-profiling --config config.yaml
```

### Python API

```python
import pandas as pd
from dataset_profiling import generate_profile_report

# Load your data
df = pd.read_csv("data.csv")

# Generate a report
generate_profile_report(
    df,
    title="My Dataset Report",
    output_file="report.html"
)
```

With LLM token analysis:

```python
generate_profile_report(
    df,
    title="Text Analysis Report",
    output_file="text_report.html",
    llm_model="gpt2",
    llm_input_cost=0.0001
)
```

## Configuration

You can configure the report generation using a YAML file:

```yaml
# Basic configuration
input: data.csv
output: report.html
title: My Dataset Report
primary_color: "#4CAF50"  # Custom primary color

# Custom schema for data type conversion
schema:
  age: int
  income: float
  date_column: date
  category_column: category

# LLM token analysis configuration
llm_models:
  gpt2:
    token: null  # API token if needed
    input_cost: 0.0001  # Cost per token
  gpt2-medium:
    input_cost: 0.0002
```

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `input` | Path to the input data file | (Required) |
| `output` | Path to save the HTML report | `profile.html` |
| `title` | Title for the report | `DataFrame Characterization Report` |
| `primary_color` | Primary color for the report | `#2196f3` |
| `schema` | Custom schema for data type conversion | `null` |
| `llm_models` | Dictionary of LLM model configurations | `{}` |

## Report Features

The generated HTML report includes:

### Dataset Overview
- Basic statistics (rows, columns, missing cells, etc.)
- Data type distribution
- Missing values analysis
- Sample data preview

### Column Analysis
- Type-specific statistics for each column
- Visualizations based on data type:
  - Numeric: Histograms and box plots
  - Categorical: Bar charts and value distributions
  - Datetime: Time series plots
  - Text: Word clouds

### Text Analysis
- Word statistics (unique words, total words, etc.)
- Word count distributions
- Word clouds
- LLM token analysis (if enabled)
- Token cost calculations (if enabled)

## Dependencies

- pandas: Data manipulation and analysis
- numpy: Numerical operations
- plotly: Interactive visualizations
- matplotlib: Static visualizations
- wordcloud: Word cloud generation
- scikit-learn: Text analysis
- typer: Command-line interface
- rich: Terminal formatting
- pyyaml: YAML configuration parsing
- transformers (optional): LLM token analysis

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Install development dependencies (`pip install -e ".[dev]"`)
4. Make your changes
5. Run the tests (`pytest`)
6. Commit your changes (`git commit -m 'Add some amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
