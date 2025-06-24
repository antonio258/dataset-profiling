"""Visualization module for dataset profiling.

This module provides functions to create various visualizations for dataset profiling,
including histograms, bar charts, box plots, time series, wordclouds, and tables.
All visualization functions return HTML that can be embedded in the final report.
"""

import base64
from io import BytesIO

import matplotlib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def create_histogram(series, title, primary_color="#2196f3"):
    """Create a histogram visualization for a numeric series.

    Args:
        series (pandas.Series): The numeric series to visualize
        title (str): The title of the histogram
        primary_color (str, optional): The color to use for the histogram bars. Defaults to "#2196f3".

    Returns:
        str: HTML representation of the histogram
    """
    fig = px.histogram(
        series.dropna(),
        x=series.name,
        title=title,
        template="plotly_white",
        color_discrete_sequence=[primary_color],
    )
    fig.update_layout(
        xaxis_title=series.name,
        yaxis_title="Count",
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        autosize=True,
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})


def create_bar_chart(series, title, primary_color="#2196f3"):
    """Create a bar chart visualization for categorical data.

    Creates a bar chart showing value counts for categorical data.
    If there are more than 20 unique values, only the top 20 are shown
    individually and the rest are grouped as "Others".

    Args:
        series (pandas.Series): The categorical series to visualize
        title (str): The title of the bar chart
        primary_color (str, optional): The color to use for the bars. Defaults to "#2196f3".

    Returns:
        str: HTML representation of the bar chart
    """
    value_counts = series.value_counts().sort_values(ascending=False)
    if len(value_counts) > 20:
        top_20 = value_counts.head(20)
        others_sum = value_counts[20:].sum()
        value_counts = pd.concat([top_20, pd.Series({"Others (by profiler)": others_sum})])

    fig = px.bar(
        x=value_counts.index,
        y=value_counts.values,
        title=title,
        template="plotly_white",
        color_discrete_sequence=[primary_color],
    )
    fig.update_traces(texttemplate="%{y}", textposition="outside")
    fig.update_layout(
        xaxis_title=series.name,
        yaxis_title="Count",
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        autosize=True,
    )
    if len(value_counts) > 5:
        fig.update_layout(xaxis_tickangle=-45)
    return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})


def create_box_plot(series, title, primary_color="#2196f3"):
    """Create a box plot visualization for a numeric series.

    Args:
        series (pandas.Series): The numeric series to visualize
        title (str): The title of the box plot
        primary_color (str, optional): The color to use for the box plot. Defaults to "#2196f3".

    Returns:
        str: HTML representation of the box plot
    """
    fig = px.box(y=series.dropna(), title=title, template="plotly_white", color_discrete_sequence=[primary_color])
    fig.update_layout(yaxis_title=series.name, showlegend=False, margin=dict(l=40, r=40, t=40, b=40), autosize=True)
    return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})


def create_time_series(series, title, primary_color="#2196f3"):
    """Create a time series visualization for a datetime series.

    Creates a line chart showing the count of values by date.

    Args:
        series (pandas.Series): The datetime series to visualize
        title (str): The title of the time series chart
        primary_color (str, optional): The color to use for the line. Defaults to "#2196f3".

    Returns:
        str: HTML representation of the time series chart
    """
    date_counts = series.dt.date.value_counts().sort_index()
    fig = px.line(
        x=date_counts.index,
        y=date_counts.values,
        title=title,
        template="plotly_white",
        color_discrete_sequence=[primary_color],
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Count",
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        autosize=True,
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})


def create_missing_values_chart(df, title, primary_color="#2196f3"):
    """Create a chart showing missing values by column.

    Creates a dual-axis chart with bars showing the count of missing values
    and a line showing the percentage of missing values for each column.
    Only columns with missing values are displayed.

    Args:
        df (pandas.DataFrame): The DataFrame to analyze for missing values
        title (str): The title of the chart
        primary_color (str, optional): The color to use for the bars. Defaults to "#2196f3".

    Returns:
        str: HTML representation of the missing values chart or an alert message if no missing values
    """
    missing = df.isna().sum().sort_values(ascending=False)
    missing_percent = (missing / len(df) * 100).round(2)
    # Filter to only show columns with missing values
    missing = missing[missing > 0]
    missing_percent = missing_percent[missing_percent > 0]
    if len(missing) == 0:
        return "<div class='alert alert-success'>No missing values found in the dataset.</div>"
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=missing.index,
            y=missing.values,
            name="Missing Count",
            marker_color=primary_color,
            text=missing.values,
            textposition="outside",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=missing.index,
            y=missing_percent[missing.index].values,
            name="Missing Percent",
            marker_color="#FF9900",
            mode="lines+markers",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="",
        template="plotly_white",
        xaxis_title="Column",
        margin=dict(l=40, r=40, t=10, b=40),
        autosize=True,
    )
    fig.update_yaxes(title_text="Count", secondary_y=False)
    fig.update_yaxes(title_text="Percent (%)", secondary_y=True)
    if len(missing) > 5:
        fig.update_layout(xaxis_tickangle=-45)
    return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})


def create_data_types_chart(df, title):
    """Create a pie chart showing the distribution of data types in the DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to analyze
        title (str): The title of the chart

    Returns:
        str: HTML representation of the data types pie chart
    """
    dtype_counts = df.dtypes.value_counts()
    fig = px.pie(
        values=dtype_counts.values,
        names=dtype_counts.index.astype(str),
        title=title,
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )
    fig.update_layout(showlegend=True, margin=dict(l=40, r=40, t=40, b=40), autosize=True)
    return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})


def create_wordcloud(series, title):
    """Create a wordcloud visualization for text data.

    Generates a wordcloud image from the text in the series, with word size
    proportional to frequency.

    Args:
        series (pandas.Series): The text series to visualize
        title (str): The title of the wordcloud

    Returns:
        str: HTML representation of the wordcloud image or an alert message if generation fails
    """
    try:
        text = " ".join(series.dropna().astype(str))
        if not text.strip() or len(text.split()) == 0:
            return "<div class='alert alert-info'>Not enough text data to generate a wordcloud.</div>"
        wordcloud = WordCloud(width=1920, height=1080, background_color="white", max_words=200).generate(text)
        buffer = BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(title)
        plt.tight_layout(pad=0)
        plt.savefig(buffer, format="png", dpi=150)
        plt.close()
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode("utf-8")
        return f'<div class="wordcloud-container"><h4>{title}</h4><img src="data:image/png;base64,{img_str}" alt="Wordcloud" style="width:100%;"></div>'
    except Exception as e:
        return f"<div class='alert alert-warning'>Could not generate wordcloud: {str(e)}</div>"


def create_value_counts_table(series):
    """Create an HTML table showing value counts for a series.

    Creates a table with columns for value, count, and percentage.

    Args:
        series (pandas.Series): The series to analyze

    Returns:
        str: HTML representation of the value counts table
    """
    value_counts = series.value_counts().sort_values(ascending=False)
    df_counts = pd.DataFrame(
        {
            "Value": value_counts.index,
            "Count": value_counts.values,
            "Percentage": (value_counts.values / len(series) * 100).round(2),
        },
    )
    html = '<table class="custom-table value-counts-table"><thead><tr><th>Value</th><th>Count</th><th>Percentage (%)</th></tr></thead><tbody>'
    for _, row in df_counts.iterrows():
        value = str(row["Value"])[:100].replace("<", "&lt;").replace(">", "&gt;")
        html += f"<tr><td>{value}</td><td>{int(row['Count']):,}</td><td>{row['Percentage']:.2f}%</td></tr>"
    html += "</tbody></table>"
    return html


def create_sample_data_table(df, max_rows=10):
    """Create an HTML table showing a sample of the DataFrame.

    Creates a table with the first few rows of the DataFrame for preview purposes.

    Args:
        df (pandas.DataFrame): The DataFrame to sample
        max_rows (int, optional): The maximum number of rows to include. Defaults to 10.

    Returns:
        str: HTML representation of the sample data table
    """
    sample_df = df.head(max_rows)
    html = '<table class="custom-table"><thead><tr>'
    for col in sample_df.columns:
        html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"
    for _, row in sample_df.iterrows():
        html += "<tr>"
        for col in sample_df.columns:
            value = row[col]
            if pd.isna(value):
                formatted_value = '<span style="color: #999; font-style: italic;">NA</span>'
            elif isinstance(value, (int, float)):
                formatted_value = f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
            else:
                formatted_value = str(value)[:100].replace("<", "&lt;").replace(">", "&gt;")
            html += f"<td>{formatted_value}</td>"
        html += "</tr>"
    html += "</tbody></table>"
    return html


def create_word_count_histogram(word_counts, title, primary_color="#2196f3"):
    """Create a histogram of word counts for text data.

    Args:
        word_counts (list): List of word counts for each document
        title (str): The title of the histogram
        primary_color (str, optional): The color to use for the histogram bars. Defaults to "#2196f3".

    Returns:
        str: HTML representation of the word count histogram or empty string if no word counts
    """
    if not word_counts:
        return ""
    df_counts = pd.DataFrame({"word_count": word_counts})
    fig = px.histogram(
        df_counts,
        x="word_count",
        title=title,
        template="plotly_white",
        color_discrete_sequence=[primary_color],
    )
    fig.update_layout(
        xaxis_title="Words per Document",
        yaxis_title="Number of Documents",
        showlegend=False,
        autosize=True,
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})


def create_word_count_boxplot(word_counts, title, primary_color="#2196f3"):
    """Create a box plot of word counts for text data.

    Args:
        word_counts (list): List of word counts for each document
        title (str): The title of the box plot
        primary_color (str, optional): The color to use for the box plot. Defaults to "#2196f3".

    Returns:
        str: HTML representation of the word count box plot or empty string if no word counts
    """
    if not word_counts:
        return ""
    df_counts = pd.DataFrame({"word_count": word_counts})
    fig = px.box(
        df_counts,
        y="word_count",
        title=title,
        template="plotly_white",
        color_discrete_sequence=[primary_color],
    )
    fig.update_layout(yaxis_title="Words per Document", showlegend=False, autosize=True)
    return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})


def create_token_count_histogram(token_counts, title):
    """Create a histogram of token counts for text data.

    Args:
        token_counts (list): List of token counts for each document
        title (str): The title of the histogram

    Returns:
        str: HTML representation of the token count histogram or empty string if no token counts
    """
    if not token_counts:
        return ""
    df_counts = pd.DataFrame({"token_count": token_counts})
    fig = px.histogram(
        df_counts,
        x="token_count",
        title=title,
        template="plotly_white",
        color_discrete_sequence=["#FF9900"],
    )
    fig.update_layout(
        xaxis_title="Tokens per Document",
        yaxis_title="Number of Documents",
        showlegend=False,
        autosize=True,
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})


def create_token_count_boxplot(token_counts, title):
    """Create a box plot of token counts for text data.

    Args:
        token_counts (list): List of token counts for each document
        title (str): The title of the box plot

    Returns:
        str: HTML representation of the token count box plot or empty string if no token counts
    """
    if not token_counts:
        return ""
    df_counts = pd.DataFrame({"token_count": token_counts})
    fig = px.box(df_counts, y="token_count", title=title, template="plotly_white", color_discrete_sequence=["#FF9900"])
    fig.update_layout(yaxis_title="Tokens per Document", showlegend=False, autosize=True)
    return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})


def create_token_cost_histogram(token_counts, cost_per_token, title):
    """Create a histogram of token costs for text data.

    Calculates the cost for each document based on token count and cost per token,
    then creates a histogram of these costs.

    Args:
        token_counts (list): List of token counts for each document
        cost_per_token (float): Cost per token in currency units
        title (str): The title of the histogram

    Returns:
        str: HTML representation of the token cost histogram or empty string if no token counts
    """
    if not token_counts or not cost_per_token:
        return ""
    costs = [count * cost_per_token for count in token_counts]
    df_costs = pd.DataFrame({"cost": costs})
    fig = px.histogram(df_costs, x="cost", title=title, template="plotly_white", color_discrete_sequence=["#FF5733"])
    fig.update_layout(
        xaxis_title="Cost per Document",
        yaxis_title="Number of Documents",
        showlegend=False,
        autosize=True,
    )
    fig.update_xaxes(tickprefix="$", tickformat=",.4f")
    return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})
