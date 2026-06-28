"""
Preprocessing module for economic text.

Provides a complete text preprocessing pipeline including tokenization,
rule-based lemmatization, stopword removal, vocabulary building, and
bag-of-words conversion -- all without external NLP library dependencies.
"""

import re
from collections import Counter
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Built-in economic stopwords
# ---------------------------------------------------------------------------
# Standard English stopwords extended with economic-domain terms that carry
# little semantic signal in monetary policy / macroeconomic documents.

STANDARD_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how", "all",
    "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "don", "should", "now", "d", "ll",
    "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn",
    "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn",
    "shan", "shouldn", "wasn", "weren", "won", "wouldn",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "would", "could", "should", "shall", "will", "might", "must",
}

ECONOMIC_STOPWORDS = {
    "committee", "members", "meeting", "federal", "reserve", "board",
    "governors", "chairman", "chair", "vice", "president", "mr", "ms",
    "dr", "said", "would", "also", "may", "one", "two", "three",
    "first", "second", "third", "well", "much", "many", "however",
    "therefore", "thus", "accordingly", "nevertheless", "furthermore",
    "indeed", "moreover", "noted", "stated", "added", "remarked",
    "observed", "commented", "emphasized", "recognized", "acknowledged",
    "suggested", "indicated", "pointed", "regarding", "concerning",
    "per", "via", "within", "without", "upon", "toward", "towards",
    "since", "although", "though", "whereas", "despite",
    "january", "february", "march", "april", "june", "july",
    "august", "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "fomc", "percent", "rate", "rates", "basis", "point", "points",
    "made", "make", "makes", "making", "see", "saw", "seen",
}

ALL_STOPWORDS = STANDARD_STOPWORDS | ECONOMIC_STOPWORDS


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """
    Tokenize text into lowercase word tokens.

    Splits on non-alphabetic characters, filters out empty tokens,
    and converts all tokens to lowercase.

    Args:
        text: Input text string.

    Returns:
        List of lowercase word tokens.

    Example:
        >>> tokenize("The Committee decided to raise rates.")
        ['the', 'committee', 'decided', 'to', 'raise', 'rates']
    """
    # Replace common punctuation with spaces
    text = re.sub(r"[^\w\s]", " ", text)
    # Replace digits with spaces (handle standalone numbers, not word-internal)
    text = re.sub(r"\b\d+\b", " ", text)
    # Split on whitespace and keep only alphabetic tokens of length >= 2
    tokens = [t.lower() for t in text.split() if t.isalpha() and len(t) >= 2]
    return tokens


# ---------------------------------------------------------------------------
# Rule-based lemmatization
# ---------------------------------------------------------------------------

def lemmatize(word: str) -> str:
    """
    Simple rule-based lemmatization for English words.

    Strips common inflectional suffixes (-ing, -ed, -s, -es, -ly, -er, -est,
    -tion endings, -ment). This is a heuristic approach suitable for
    bag-of-words models where exact linguistic correctness is not required.

    Args:
        word: Input word token (lowercase).

    Returns:
        Lemmatized form of the word.

    Example:
        >>> lemmatize("increasing")
        'increas'
        >>> lemmatize("rates")
        'rate'
        >>> lemmatize("tightened")
        'tighten'
    """
    if len(word) <= 3:
        return word

    # -ing: running -> runn, hiking -> hik
    if word.endswith("ing") and len(word) > 4:
        stem = word[:-3]
        # Double consonant fixing: "running" -> "run"
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem

    # -ed: walked -> walk, relied -> reli (acceptable for BoW)
    if word.endswith("ed") and len(word) > 4:
        stem = word[:-2]
        if stem.endswith("i"):
            stem = stem[:-1] + "y"
        return stem

    # -es: boxes -> box, rates -> rate
    if word.endswith("es") and len(word) > 3:
        return word[:-2]

    # -s (but not -ss): rates -> rate
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        stem = word[:-1]
        # Don't strip if it creates a very short word
        if len(stem) >= 3:
            return stem

    # -ly: quickly -> quick
    if word.endswith("ly") and len(word) > 4:
        return word[:-2]

    # -er: higher -> high
    if word.endswith("er") and len(word) > 4:
        stem = word[:-2]
        if stem.endswith("i"):
            stem = stem[:-1] + "y"
        return stem

    # -est: highest -> high
    if word.endswith("est") and len(word) > 5:
        return word[:-3]

    # -tion: inflation -> inflate
    if word.endswith("tion") and len(word) > 6:
        return word[:-4] + "te"

    # -ment: employment -> employ
    if word.endswith("ment") and len(word) > 6:
        return word[:-4]

    return word


# ---------------------------------------------------------------------------
# Stopword removal
# ---------------------------------------------------------------------------

def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Remove stopwords from a list of tokens.

    Uses a combined list of standard English stopwords and economic-domain
    stopwords (terms that appear frequently in monetary policy documents
    but carry little semantic signal).

    Args:
        tokens: List of word tokens.

    Returns:
        Filtered list with stopwords removed.

    Example:
        >>> remove_stopwords(['the', 'committee', 'raised', 'rates'])
        ['raised', 'rates']
    """
    return [t for t in tokens if t not in ALL_STOPWORDS]


# ---------------------------------------------------------------------------
# Full cleaning pipeline
# ---------------------------------------------------------------------------

def clean_text(text: str, lemmatize_tokens: bool = True) -> List[str]:
    """
    Apply the full text preprocessing pipeline.

    Steps: tokenize -> (lemmatize) -> remove stopwords.

    Args:
        text: Input text string.
        lemmatize_tokens: If True, apply rule-based lemmatization to each token.

    Returns:
        List of cleaned, processed tokens.

    Example:
        >>> clean_text("The unemployment rate increased sharply.")
        ['unemployment', 'rate', 'increased', 'sharply']
    """
    tokens = tokenize(text)
    if lemmatize_tokens:
        tokens = [lemmatize(t) for t in tokens]
    tokens = remove_stopwords(tokens)
    return tokens


# ---------------------------------------------------------------------------
# Vocabulary building
# ---------------------------------------------------------------------------

def build_vocab(documents: List[List[str]],
                min_freq: int = 2) -> Dict[str, int]:
    """
    Build a vocabulary mapping words to integer indices.

    Words are sorted by frequency (descending) then alphabetically.
    Only words appearing at least ``min_freq`` times across all documents
    are included.

    Args:
        documents: List of preprocessed token lists (one per document).
        min_freq: Minimum frequency threshold for inclusion.

    Returns:
        Dictionary mapping word -> integer index.

    Example:
        >>> docs = [['growth', 'inflation', 'growth'], ['inflation', 'policy']]
        >>> vocab = build_vocab(docs, min_freq=1)
        >>> vocab
        {'growth': 0, 'inflation': 1, 'policy': 2}
    """
    counter = Counter()
    for doc in documents:
        counter.update(doc)
    # Filter by min_freq
    filtered = [(w, c) for w, c in counter.items() if c >= min_freq]
    # Sort by frequency (desc), then alphabetically
    filtered.sort(key=lambda x: (-x[1], x[0]))
    vocab = {word: idx for idx, (word, _) in enumerate(filtered)}
    return vocab


# ---------------------------------------------------------------------------
# Bag-of-words conversion
# ---------------------------------------------------------------------------

def text_to_bow(text: str,
                vocab: Dict[str, int],
                lemmatize_tokens: bool = True) -> Tuple[List[int], List[float]]:
    """
    Convert text into a bag-of-words vector.

    Preprocesses the text using the full pipeline, then builds a sparse
    representation as (indices, counts) tuples relative to the provided
    vocabulary. Words not in the vocabulary are ignored.

    Args:
        text: Input text string.
        vocab: Vocabulary dictionary (word -> index).
        lemmatize_tokens: If True, lemmatize tokens before matching.

    Returns:
        Tuple of (indices, counts) as parallel lists.
        Indices correspond to positions in the vocabulary;
        counts are the frequencies of those words in the text.

    Example:
        >>> vocab = {'growth': 0, 'inflation': 1}
        >>> idx, cnt = text_to_bow("strong growth and low inflation", vocab)
        >>> list(zip(idx, cnt))
        [(0, 1), (1, 1)]
    """
    tokens = clean_text(text, lemmatize_tokens=lemmatize_tokens)
    counter = Counter(tokens)
    indices = []
    counts = []
    for word, count in counter.items():
        if word in vocab:
            indices.append(vocab[word])
            counts.append(count)
    return indices, counts
