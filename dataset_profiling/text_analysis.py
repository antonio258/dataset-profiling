import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

try:
    from transformers import AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def analyze_text_with_countvectorizer(series):
    """
    Analyze text data using CountVectorizer to calculate word statistics.

    Args:
        series (pandas.Series): The series containing text data

    Returns:
        dict: Dictionary containing text statistics
    """
    try:
        non_null_series = series.dropna().astype(str)
        if len(non_null_series) == 0:
            return {"unique_words": 0, "total_words": 0, "avg_words_per_doc": 0, "word_counts": []}

        vectorizer = CountVectorizer(strip_accents='unicode', lowercase=True, token_pattern=r'\b\w+\b')
        X = vectorizer.fit_transform(non_null_series)
        vocabulary = vectorizer.get_feature_names_out()
        word_counts_matrix = X.sum(axis=1)

        total_words = int(X.sum())
        avg_words_per_doc = total_words / len(non_null_series) if len(non_null_series) > 0 else 0

        return {
            "unique_words": len(vocabulary),
            "total_words": total_words,
            "avg_words_per_doc": round(avg_words_per_doc, 2),
            "word_counts": [int(count[0, 0]) for count in word_counts_matrix]
        }
    except Exception as e:
        print(f"Error analyzing text with CountVectorizer: {str(e)}")
        return {"unique_words": 0, "total_words": 0, "avg_words_per_doc": 0, "word_counts": []}


def analyze_text_with_llm_tokenizer(series, model_name, token=None):
    """
    Analyze text data using an LLM tokenizer to calculate token statistics.

    Args:
        series (pandas.Series): The series containing text data
        model_name (str): Name of the LLM model to use for tokenization
        token (str, optional): Token for accessing the LLM model

    Returns:
        dict: Dictionary containing token statistics
    """
    if not TRANSFORMERS_AVAILABLE:
        print("Warning: transformers library not available. Skipping LLM token analysis.")
        return {"unique_tokens": 0, "total_tokens": 0, "avg_tokens_per_doc": 0, "token_counts": []}

    try:
        non_null_series = series.dropna().astype(str)
        if len(non_null_series) == 0:
            return {"unique_tokens": 0, "total_tokens": 0, "avg_tokens_per_doc": 0, "token_counts": []}

        print(f"Loading tokenizer for model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

        token_counts = []
        all_token_ids = set()
        for text in non_null_series:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            token_counts.append(len(tokens))
            all_token_ids.update(tokens)

        total_tokens = sum(token_counts)
        avg_tokens_per_doc = total_tokens / len(non_null_series) if len(non_null_series) > 0 else 0

        return {
            "unique_tokens": len(all_token_ids),
            "total_tokens": total_tokens,
            "avg_tokens_per_doc": round(avg_tokens_per_doc, 2),
            "token_counts": token_counts
        }
    except Exception as e:
        print(f"Error analyzing text with LLM tokenizer: {str(e)}")
        return {"unique_tokens": 0, "total_tokens": 0, "avg_tokens_per_doc": 0, "token_counts": []}