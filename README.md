# Dataset-Profiling

![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A comprehensive Python package for profiling and analyzing datasets, generating detailed HTML reports with visualizations and statistics.

## 🚀 Features

- Generate comprehensive HTML reports for datasets
- Analyze various data types (numeric, categorical, text, datetime, boolean)
- Create visualizations for each column type (histograms, bar charts, box plots, etc.)
- Calculate detailed statistics for each column
- Analyze text data with word frequency analysis and word clouds
- Support for LLM tokenization analysis (optional)
- Customizable report styling
- Command-line interface for easy use
- YAML configuration support for advanced options

## 📋 Requirements

- Python 3.11+
- Dependencies:
  - pandas
  - numpy
  - plotly
  - matplotlib
  - wordcloud
  - scikit-learn
  - pyyaml
  - rich
  - typer

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/anahealth-app/dataset-profiling.git
cd dataset-profiling
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install the package:
```bash
pip install -e .
```

## 🏃‍♂️ Usage

### Command Line Interface

Generate a profile report for a dataset:

```bash
dataset-profiling profile --input your_data.csv --output report.html
```

For more options:

```bash
dataset-profiling profile --help
```

## 🧪 Development

### Project Structure

```
├── dataset_profiling/          # Package source code
│   ├── __init__.py             # Package initialization and exports
│   ├── __main__.py             # Entry point for running as a module
│   ├── cli.py                  # Command-line interface
│   ├── io.py                   # Data loading functions
│   ├── stats.py                # Statistical analysis functions
│   ├── visualize.py            # Visualization functions
│   ├── text_analysis.py        # Text analysis functions
│   └── report.py               # Report generation functions
├── tests/                      # Unit tests
│   ├── __init__.py
│   └── dataset_profiling_test.py
├── docs/                       # Documentation
├── README.md                   # Project documentation
├── pyproject.toml              # Project configuration
└── custom_dataframe_profiler.py # Original source code (for reference)
```

### Using as a Library

You can also use Dataset Profiling as a library in your Python code:

```python
import pandas as pd
from dataset_profiling import generate_profile_report

# Load your data
df = pd.read_csv("your_data.csv")

# Generate a profile report
generate_profile_report(
    df, 
    title="My Dataset Analysis",
    output_file="profile_report.html"
)
```

### Advanced Configuration

For more advanced options, you can use a YAML configuration file:

```yaml
# config.yaml
input: data/my_dataset.csv
output: reports/profile_report.html
title: Comprehensive Dataset Analysis

# Schema definition for data types
schema:
  age:
    type: int
  income:
    type: float
  category:
    type: categorical
    categories: ["A", "B", "C"]
    ordered: true

# LLM configuration for text analysis
llm_models:
  gpt2:
    input_cost: 0.0001
  bert-base-uncased:
    input_cost: 0.00005

# Visual customization
primary_color: "#4CAF50"
```

Then use it with:

```bash
dataset-profiling profile --config config.yaml
```

### Tests

To run the tests:

```bash
pytest
```

## 🧹 Code Quality

The project uses several tools to ensure code quality:

- Black for formatting
- Flake8 for linting
- isort for import sorting
- pre-commit for git hooks

To set up pre-commit hooks:

```bash
pre-commit install
```

To run pre-commit for all files:
```bash
pre-commit run --all-files
```

## 📝 Documentation

Documentation can be generated using:

```bash
bash generate_docs.sh
```

## 📜 License

This project is licensed under the MIT License.

---

**Made with ❤️ by the Dataset Profiling Team**

[Report Bug](mailto:antonio258p@gmail.com) • [Request Feature](mailto:antonio258p@gmail.com)
