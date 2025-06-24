"""Main entry point for the dataset_profiling package.

This module allows the package to be run as a script using either:
- `python -m dataset_profiling`
- Direct execution of the module

It imports and runs the CLI application from the cli module.
"""

from .cli import app

if __name__ == "__main__":
    app()
else:
    # This allows the module to be run with `python -m dataset_profiling`
    app()
