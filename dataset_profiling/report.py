from datetime import datetime
from .stats import get_basic_stats, analyze_column
from .visualize import (
    create_histogram, create_bar_chart, create_box_plot, create_time_series,
    create_missing_values_chart, create_data_types_chart, create_wordcloud,
    create_value_counts_table, create_sample_data_table,
    create_word_count_histogram, create_word_count_boxplot,
    create_token_count_histogram, create_token_count_boxplot, create_token_cost_histogram
)

def generate_profile_report(df, title="DataFrame Characterization Report", output_file="profile.html", 
                     llm_models=None, llm_model=None, llm_token=None, llm_input_cost=None, primary_color="#2196f3"):
    """
    Generate a comprehensive profile report for the DataFrame.
    """
    print("Analyzing basic statistics...")
    basic_stats = get_basic_stats(df)

    print("Analyzing columns...")
    column_analyses = []
    for column in df.columns:
        print(f"  - Analyzing column: {column}")
        analysis = analyze_column(df, column, llm_models, llm_model, llm_token, llm_input_cost)
        column_analyses.append(analysis)

    print("Generating HTML report...")
    html_report = create_html_report(df, title, column_analyses, basic_stats, primary_color)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_report)

    print(f"Profile report successfully generated and saved to {output_file}")
    return output_file


def create_html_report(df, title, column_analyses, basic_stats, primary_color="#2196f3"):
    """
    Create an HTML report with all the analyses and visualizations.
    """
    missing_chart = create_missing_values_chart(df, "Missing Values by Column", primary_color)
    dtypes_chart = create_data_types_chart(df, "Data Types Distribution")

    column_html = ""
    text_columns_html = ""
    for analysis in column_analyses:
        column_name = analysis["name"]
        series = df[column_name]

        # Common part of the column card
        card_header = f"""
        <div class="column-card">
            <h3>{column_name}</h3>
            <div class="column-metadata">
                <p><strong>Type:</strong> {analysis["dtype"]}</p>
                <p><strong>Missing:</strong> {analysis["missing"]} ({analysis["missing_percent"]}%)</p>
                <p><strong>Unique:</strong> {analysis["unique"]} ({analysis["unique_percent"]}%)</p>
        """

        # Type-specific metadata
        if analysis["is_numeric"]:
            card_header += f"""
                <p><strong>Min:</strong> {analysis.get("min")}</p> <p><strong>Max:</strong> {analysis.get("max")}</p> <p><strong>Mean:</strong> {analysis.get("mean")}</p>
                <p><strong>Median:</strong> {analysis.get("median")}</p> <p><strong>Std Dev:</strong> {analysis.get("std")}</p>
            """
        elif analysis["is_datetime"]:
            card_header += f"""
                <p><strong>Start:</strong> {analysis.get("min")}</p> <p><strong>End:</strong> {analysis.get("max")}</p> <p><strong>Range (days):</strong> {analysis.get("range_days")}</p>
            """
        elif "avg_length" in analysis:
            card_header += f"""
                 <p><strong>Avg Length:</strong> {analysis.get("avg_length")}</p>
                 <p><strong>Min Length:</strong> {analysis.get("min_length")}</p>
                 <p><strong>Max Length:</strong> {analysis.get("max_length")}</p>
            """

        card_header += "</div>"

        visualizations = ""
        # Type-specific visualizations for the "Column Analysis" tab
        if analysis["is_numeric"]:
            visualizations += f'<div class="viz-container">{create_histogram(series, f"Distribution of {column_name}", primary_color)}</div>'
            visualizations += f'<div class="viz-container">{create_box_plot(series, f"Box Plot of {column_name}", primary_color)}</div>'
        elif analysis["is_datetime"]:
            visualizations += f'<div class="viz-container">{create_time_series(series, f"Time Series of {column_name}", primary_color)}</div>'
        elif analysis["is_text"]:
            visualizations += f'<div class="viz-container">{create_wordcloud(series, f"Word Cloud of {column_name}")}</div>'
        else: # Categorical
             visualizations += f'<div class="viz-container">{create_bar_chart(series, f"Value Counts of {column_name}", primary_color)}</div>'
             visualizations += f'''
                <div class="expandable-card">
                    <div class="expandable-header" onclick="toggleExpandable(this)"><h4>Full Value Distribution</h4><span class="expand-icon">+</span></div>
                    <div class="expandable-content"><div class="table-container">{create_value_counts_table(series)}</div></div>
                </div>'''

        column_html += f"{card_header}<div class='column-visualization'>{visualizations}</div></div>"

        # Build detailed analysis for the "Text Analysis" tab if it's a text column
        if analysis["is_text"]:
            text_card_content = f"{card_header}"
            text_card_content += f'<div class="viz-container">{create_wordcloud(series, f"Word Cloud of {column_name}")}</div>'

            # Word stats
            text_card_content += f"""
                <div class="text-stats-container">
                    <h4>Word Statistics</h4>
                    <div class="text-stats-grid">
                        <div class="text-stat-card"><div class="text-stat-title">Unique Words</div><div class="text-stat-value">{analysis.get("unique_words", 0):,}</div></div>
                        <div class="text-stat-card"><div class="text-stat-title">Total Words</div><div class="text-stat-value">{analysis.get("total_words", 0):,}</div></div>
                        <div class="text-stat-card"><div class="text-stat-title">Avg Words/Doc</div><div class="text-stat-value">{analysis.get("avg_words_per_doc", 0)}</div></div>
                    </div>
                </div>
            """
            text_card_content += f'<div class="viz-container">{create_word_count_histogram(analysis.get("word_counts", []), f"Word Count Distribution", primary_color)}</div>'
            text_card_content += f'<div class="viz-container">{create_word_count_boxplot(analysis.get("word_counts", []), f"Word Count Boxplot", primary_color)}</div>'

            # LLM stats
            if "model_token_stats" in analysis and analysis["model_token_stats"]:
                 text_card_content += '<div class="alert alert-info"><strong>Note:</strong> Calculated costs are based on input costs only.</div>'
                 # Get the model used for tokenization (if available)
                 tokenizer_model = analysis.get("model_name", "")

                 for model_name, model_stats in analysis["model_token_stats"].items():
                     if model_stats.get("total_tokens", 0) > 0:
                        # Start expandable card
                        text_card_content += f'''
                        <div class="expandable-card">
                            <div class="expandable-header" onclick="toggleExpandable(this)"><h4>LLM Analysis: {model_name}</h4><span class="expand-icon">+</span></div>
                            <div class="expandable-content">
                                <div class="text-stats-container">
                        '''

                        # Only show token statistics for the model used to generate tokens
                        if model_name == tokenizer_model:
                            text_card_content += f'''
                                    <div class="text-stats-grid">
                                        <div class="text-stat-card"><div class="text-stat-title">Unique Tokens</div><div class="text-stat-value">{model_stats.get("unique_tokens", 0):,}</div></div>
                                        <div class="text-stat-card"><div class="text-stat-title">Total Tokens</div><div class="text-stat-value">{model_stats.get("total_tokens", 0):,}</div></div>
                                        <div class="text-stat-card"><div class="text-stat-title">Avg Tokens/Doc</div><div class="text-stat-value">{model_stats.get("avg_tokens_per_doc", 0)}</div></div>
                                    </div>
                            '''

                        # Show cost statistics for all models
                        if "total_token_cost" in model_stats:
                            text_card_content += f'''
                                <h4 class="mt-4">Cost Statistics</h4>
                                <div class="text-stats-grid">
                                    <div class="text-stat-card"><div class="text-stat-title">Total Cost</div><div class="text-stat-value">${model_stats.get("total_token_cost", 0):,.4f}</div></div>
                                    <div class="text-stat-card"><div class="text-stat-title">Avg Cost/Doc</div><div class="text-stat-value">${model_stats.get("avg_cost_per_doc", 0):,.4f}</div></div>
                                </div>'''

                        # Close the stats container
                        text_card_content += '''
                                </div>
                        '''

                        # Only show token visualizations for the model used to generate tokens
                        if model_name == tokenizer_model:
                            text_card_content += f'''
                                <div class="viz-container">{create_token_count_histogram(model_stats.get("token_counts", []), f"Token Count Distribution ({model_name})")}</div>
                                <div class="viz-container">{create_token_count_boxplot(model_stats.get("token_counts", []), f"Token Count Boxplot ({model_name})")}</div>
                            '''

                        # Close the expandable card
                        text_card_content += '''
                            </div>
                        </div>'''

            text_columns_html += f'<div class="column-card">{text_card_content}</div>'

    # The full HTML template is very large. It is being copied from the original file.
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;700&display=swap" rel="stylesheet">
        <script>
            function toggleExpandable(element) {{
                const content = element.nextElementSibling;
                const icon = element.querySelector('.expand-icon');
                const isDisplayed = content.style.display === "block";
                content.style.display = isDisplayed ? "none" : "block";
                icon.textContent = isDisplayed ? "+" : "−";
                if (!isDisplayed) setTimeout(() => window.dispatchEvent(new Event('resize')), 100);
            }}
            function openTab(evt, tabName) {{
                document.querySelectorAll(".tab-content").forEach(tc => tc.style.display = "none");
                document.querySelectorAll(".tab-button").forEach(tb => tb.classList.remove("active"));
                document.getElementById(tabName).style.display = "block";
                evt.currentTarget.classList.add("active");
                setTimeout(() => window.dispatchEvent(new Event('resize')), 100);
            }}
            window.addEventListener('load', () => setTimeout(() => window.dispatchEvent(new Event('resize')), 100));
        </script>
        <style>
            :root {{ --primary-color: {primary_color}; --card-background: #FFFFFF; --background-color: #FAFAFA; --text-primary: #212121; --text-secondary: #757575; --divider-color: #EEEEEE; --elevation-1: 0 2px 1px -1px rgba(0,0,0,0.2), 0 1px 1px 0 rgba(0,0,0,0.14), 0 1px 3px 0 rgba(0,0,0,0.12); }}
            body {{ font-family: 'Inter', sans-serif; margin: 0; background-color: var(--background-color); color: var(--text-primary); }}
            .container {{ width: 100%; max-width: 100%; margin: 0 auto; padding: 24px; }}
            header {{ background: var(--primary-color); color: white; padding: 24px; text-align: center; margin-bottom: 32px; box-shadow: var(--elevation-4); }}
            header h1 {{ margin: 0; font-size: 1.75rem; }} header p {{ opacity: 0.9; }}
            h2, h3, h4 {{ color: var(--text-primary); }}
            .tab-container {{ width: 100%; margin-bottom: 32px; }}
            .tab-navigation {{ display: flex; background-color: var(--card-background); box-shadow: var(--elevation-1); position: sticky; top: 0; z-index: 100; }}
            .tab-button {{ background: transparent; border: none; cursor: pointer; padding: 16px 24px; font-size: 0.875rem; font-weight: 500; color: var(--text-secondary); position: relative; }}
            .tab-button.active {{ color: var(--primary-color); }}
            .tab-button.active::after {{ content: ''; position: absolute; bottom: 0; left: 0; width: 100%; height: 2px; background-color: var(--primary-color); }}
            .tab-content {{ display: none; padding-top: 24px; }} #dataset-overview {{ display: block; }}
            .stats-container, .text-stats-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }}
            .stat-card, .text-stat-card {{ background-color: var(--card-background); border-radius: 4px; padding: 16px; box-shadow: var(--elevation-1); text-align: center; }}
            .stat-card h3, .text-stat-title {{ font-size: 0.875rem; color: var(--text-secondary); margin-bottom: 8px; font-weight: 500; }}
            .stat-card p, .text-stat-value {{ font-size: 1.75rem; font-weight: 500; color: var(--primary-color); }}
            .overview-card, .column-card {{ background-color: var(--card-background); border-radius: 4px; box-shadow: var(--elevation-1); padding: 24px; margin-bottom: 32px; overflow: hidden; }}
            .overview-card h3, .column-card h3 {{ margin-top: 0; color: var(--primary-color); border-bottom: 1px solid var(--divider-color); padding-bottom: 8px; font-size: 1.25rem; font-weight: 500; letter-spacing: 0.15px; }}
            .viz-container, .text-stats-container {{ padding: 16px; }}
            .column-metadata {{ display: flex; flex-wrap: wrap; margin-bottom: 16px; padding: 16px 0; border-bottom: 1px solid var(--divider-color); }} .column-metadata p {{ margin: 0 16px 16px 0; font-size: 0.875rem; }} .column-metadata p strong {{ color: var(--text-primary); }}
            .alert {{ padding: 16px; border-radius: 4px; margin: 16px; border: 1px solid; }} .alert-info {{ border-color: #2196F3; background-color: #E3F2FD; color: #1E88E5; }}
            .expandable-card {{ border: 1px solid var(--divider-color); margin: 16px; border-radius: 4px; }}
            .expandable-header {{ padding: 12px 16px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; background-color: #f5f5f5; }}
            .expandable-content {{ display: none; padding: 16px; max-height: 500px; overflow-y: auto; }}
            .table-container {{ overflow-x: auto; }}
            table.custom-table {{ width: 100%; border-collapse: collapse; }}
            table.custom-table th {{ background-color: var(--primary-color); color: white; padding: 12px; text-align: left; }}
            table.custom-table td {{ padding: 10px; border-bottom: 1px solid var(--divider-color); }}
            table.custom-table tr:hover {{ background-color: #f0f0f0; }}
            .mt-4 {{ margin-top: 16px; }}
            footer {{ text-align: center; margin-top: 30px; padding: 20px; color: #666; font-size: 14px; border-top: 1px solid var(--divider-color); }}
        </style>
    </head>
    <body>
        <div class="container">
            <header><h1>{title}</h1><p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p></header>
            <div class="tab-container">
                <div class="tab-navigation">
                    <button class="tab-button active" onclick="openTab(event, 'dataset-overview')">Dataset Overview</button>
                    <button class="tab-button" onclick="openTab(event, 'column-analysis')">Column Analysis</button>
                    <button class="tab-button" onclick="openTab(event, 'text-analysis')">Text Analysis</button>
                </div>
                <div id="dataset-overview" class="tab-content">
                    <div class="stats-container">
                        <div class="stat-card"><h3>Rows</h3><p>{basic_stats["rows"]:,}</p></div>
                        <div class="stat-card"><h3>Columns</h3><p>{basic_stats["columns"]}</p></div>
                        <div class="stat-card"><h3>Missing Cells</h3><p>{basic_stats["missing_cells"]:,} ({basic_stats["missing_percent"]}%)</p></div>
                        <div class="stat-card"><h3>Duplicate Rows</h3><p>{basic_stats["duplicate_rows"]:,} ({basic_stats["duplicate_percent"]}%)</p></div>
                        <div class="stat-card"><h3>Memory Usage</h3><p>{basic_stats["memory_usage"]} MB</p></div>
                    </div>
                    <div class="overview-card"><h3>Data Types</h3><div class="viz-container">{dtypes_chart}</div></div>
                    <div class="overview-card"><h3>Missing Values</h3><div class="viz-container">{missing_chart}</div></div>
                    <div class="overview-card"><h3>Sample Data</h3><div class="table-container">{create_sample_data_table(df)}</div></div>
                </div>
                <div id="column-analysis" class="tab-content">{column_html}</div>
                <div id="text-analysis" class="tab-content">
                    <h2>Text Columns Analysis</h2>
                    <p>This tab contains detailed analysis of text columns including word statistics, LLM token statistics, and more.</p>
                    <div id="text-columns-container">{text_columns_html if text_columns_html else "<p>No text columns found in the dataset.</p>"}</div>
                </div>
            </div>
            <footer><p>DataFrame Profiler</p></footer>
        </div>
    </body>
    </html>
    """
    return html_template
