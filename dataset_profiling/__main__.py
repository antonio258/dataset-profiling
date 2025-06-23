"""dataset_profiling main module."""

from .cli import app

if __name__ == "__main__":
    app()
else:
    # This allows the module to be run with `python -m dataset_profiling`
    app()
