"""Command-line interface module for dataset profiling.

This module provides a command-line interface for the dataset profiling tool,
allowing users to generate comprehensive reports on datasets through the command line.
It supports loading data from various file formats and configuring the report
through command-line options or a YAML configuration file.
"""

from pathlib import Path

import typer
from rich.console import Console

from .report import generate_profile_report
from .io import load_data, parse_yaml_config

app = typer.Typer(help="Dataset profiling tool for generating comprehensive reports on datasets.")
console = Console()


@app.command()
def profile(
    input_file: Path = typer.Option(
        None,
        "--input",
        "-i",
        help="Path to the input data file (csv, excel, json, parquet, pickle)",
    ),
    config_file: Path = typer.Option(None, "--config", "-c", help="Path to YAML configuration file"),
    output_file: Path = typer.Option("profile.html", "--output", "-o", help="Path to save the output HTML report"),
    title: str = typer.Option("DataFrame Characterization Report", "--title", "-t", help="Title for the report"),
):
    """Generate a comprehensive profile report for a dataset.

    This command analyzes a dataset and generates an HTML report with detailed statistics
    and visualizations. The report includes basic statistics, data type distribution,
    missing values analysis, and column-specific visualizations based on data types.

    The command can be used in two ways:
    1. Directly with command-line options (--input, --output, --title)
    2. With a YAML configuration file (--config) that provides more advanced options
       including LLM tokenization and custom schema definitions

    Args:
        input_file: Path to the input data file (CSV, Excel, JSON, Parquet, Pickle)
        config_file: Path to YAML configuration file with advanced options
        output_file: Path to save the output HTML report
        title: Title for the report

    Raises:
        typer.Exit: If required arguments are missing or if an error occurs during processing
    """
    if not input_file and not config_file:
        console.print("[red]Error: Either --input or --config must be provided.[/red]")
        raise typer.Exit(code=1)

    if input_file and config_file:
        console.print("[red]Error: Cannot provide both --input and --config.[/red]")
        raise typer.Exit(code=1)

    try:
        if config_file:
            console.print(f"Loading configuration from [cyan]{config_file}[/cyan]...")
            config = parse_yaml_config(str(config_file))

            input_path = config["input"]
            output_path = config.get("output", str(output_file))
            report_title = config.get("title", title)
            schema = config.get("schema", None)
            llm_models = config.get("llm_models", {})
            llm_model = config.get("llm_model")
            llm_token = config.get("llm_token")
            llm_input_cost = config.get("llm_input_cost")
            primary_color = config.get("primary_color", "#2196f3")

            if llm_models:
                console.print(f"Using {len(llm_models)} LLM models for token analysis:")
                for model_name, model_config in llm_models.items():
                    cost_info = (
                        f" (cost: ${model_config.get('input_cost')} per token)"
                        if model_config.get("input_cost")
                        else ""
                    )
                    console.print(f"  - [cyan]{model_name}[/cyan]{cost_info}")

            console.print(f"Loading data from [cyan]{input_path}[/cyan]...")
            df = load_data(input_path, schema)
        else:
            # Use command line arguments
            input_path = str(input_file)
            output_path = str(output_file)
            report_title = title

            # No LLM parameters when using command line arguments directly in this simplified CLI
            llm_models, llm_model, llm_token, llm_input_cost, primary_color = None, None, None, None, "#2196f3"

            console.print(f"Loading data from [cyan]{input_path}[/cyan]...")
            df = load_data(input_path)

        console.print(
            f"DataFrame loaded successfully with [green]{df.shape[0]}[/green] rows and [green]{df.shape[1]}[/green] columns.",
        )

        with console.status("[bold green]Generating profile report...", spinner="dots"):
            generate_profile_report(
                df,
                report_title,
                output_path,
                llm_models,
                llm_model,
                llm_token,
                llm_input_cost,
                primary_color,
            )

        console.print(
            f"\n[bold green]Profile report successfully generated and saved to[/bold green] [cyan]{output_path}[/cyan]",
        )

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        import traceback

        console.print(traceback.format_exc())
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
