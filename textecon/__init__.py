"""
TextEcon -- NLP for Economics.

A lightweight toolkit for applying natural language processing techniques
to economic text corpora. Provides preprocessing, sentiment analysis,
topic modeling, word embeddings, and built-in economic text corpora.
"""

from textecon.preprocess import (
    tokenize,
    lemmatize,
    remove_stopwords,
    clean_text,
    build_vocab,
    text_to_bow,
)
from textecon.sentiment import (
    economic_sentiment,
    sentiment_timeseries,
    policy_uncertainty_index,
)
from textecon.topic import LDA
from textecon.embeddings import (
    cooccurrence_matrix,
    ppmi,
    svd_embeddings,
    economic_similarity,
    document_embedding,
)
from textecon.corpora import (
    load_fomc_statements,
    load_economic_news,
    FOMC_STATEMENTS,
    ECONOMIC_NEWS,
    FED_MINUTES_SUMMARIES,
)

__version__ = "0.1.0"
__all__ = [
    # Preprocessing
    "tokenize",
    "lemmatize",
    "remove_stopwords",
    "clean_text",
    "build_vocab",
    "text_to_bow",
    # Sentiment
    "economic_sentiment",
    "sentiment_timeseries",
    "policy_uncertainty_index",
    # Topic modeling
    "LDA",
    # Embeddings
    "cooccurrence_matrix",
    "ppmi",
    "svd_embeddings",
    "economic_similarity",
    "document_embedding",
    # Corpora
    "load_fomc_statements",
    "load_economic_news",
    "FOMC_STATEMENTS",
    "ECONOMIC_NEWS",
    "FED_MINUTES_SUMMARIES",
]
