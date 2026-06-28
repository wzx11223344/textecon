"""
Word and document embeddings module for economic text.

Provides methods for building co-occurrence matrices, computing
Positive Pointwise Mutual Information (PPMI), performing truncated SVD
for dense embeddings, and measuring economic-domain cosine similarity.
All methods are implemented with NumPy/SciPy -- no deep learning
frameworks required.
"""

from collections import Counter
from typing import Dict, List, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from textecon.preprocess import clean_text, build_vocab


def cooccurrence_matrix(
    documents: List[List[str]],
    vocab: Dict[str, int] = None,
    window: int = 5,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Build a word co-occurrence matrix from a collection of documents.

    For each word token, increments the co-occurrence count with every
    other word within a symmetric window of ``window`` tokens.

    Args:
        documents: List of preprocessed token lists (one per document).
        vocab: Vocabulary dict ``word -> index``. If None, builds one
            from the documents including all unique words.
        window: Number of context tokens on each side of the target word.

    Returns:
        Tuple of ``(matrix, vocab)`` where ``matrix`` is a numpy array
        of shape ``(V, V)`` and ``vocab`` is the word-to-index mapping.

    Example:
        >>> docs = [['growth', 'inflation', 'policy'],
        ...         ['inflation', 'growth', 'rate']]
        >>> mat, vocab = cooccurrence_matrix(docs, window=2)
        >>> mat.shape
        (4, 4)
    """
    if vocab is None:
        vocab = build_vocab(documents, min_freq=1)
    V = len(vocab)
    matrix = np.zeros((V, V), dtype=np.float64)

    for doc in documents:
        tokens = [vocab[w] for w in doc if w in vocab]
        n = len(tokens)
        for i, wi in enumerate(tokens):
            start = max(0, i - window)
            end = min(n, i + window + 1)
            for j in range(start, end):
                if i != j:
                    wj = tokens[j]
                    matrix[wi, wj] += 1.0

    return matrix, vocab


def ppmi(matrix: np.ndarray, k: float = None, shift: float = 0.0) -> np.ndarray:
    """
    Compute Positive Pointwise Mutual Information (PPMI) from a
    co-occurrence matrix.

    PPMI(w, c) = max(0, log(P(w, c) / (P(w) * P(c))) - log(k))

    where k > 1 discounts frequent word pairs and shift adds a constant
    offset to all PMI values before clamping (Shifted PPMI, Levy &
    Goldberg 2014).

    Args:
        matrix: Square co-occurrence matrix of shape (V, V).
        k: Context distribution smoothing parameter (k=1 for standard PPMI).
            Typically k=5 or k=10 for word embeddings. If None, uses k=1.
        shift: Shift constant for Shifted PPMI (default 0 = standard PPMI).

    Returns:
        PPMI matrix of shape (V, V).

    Example:
        >>> cooc = np.array([[5, 2, 0],
        ...                  [2, 7, 3],
        ...                  [0, 3, 4]], dtype=float)
        >>> result = ppmi(cooc)
        >>> (result >= 0).all()
        np.True_
    """
    if k is None:
        k = 1.0
    total = matrix.sum()
    if total == 0:
        return np.zeros_like(matrix)

    # P(w) = sum_c #(w,c) / total
    row_sums = matrix.sum(axis=1)
    p_w = row_sums / total

    # P_alpha(c) = sum_w #(w,c)^alpha / sum_{wc} #(w,c)^alpha
    col_sums_alpha = np.power(matrix, k).sum(axis=0)
    total_alpha = col_sums_alpha.sum()
    if total_alpha == 0:
        return np.zeros_like(matrix)
    p_c_alpha = col_sums_alpha / total_alpha

    V = matrix.shape[0]
    ppmi_mat = np.zeros((V, V), dtype=np.float64)

    for i in range(V):
        for j in range(V):
            count = matrix[i, j]
            if count == 0:
                continue
            p_wc = count / total
            denom = p_w[i] * p_c_alpha[j]
            if denom == 0:
                continue
            pmi = np.log(p_wc / denom) - np.log(k)
            ppmi_mat[i, j] = max(0.0, pmi - shift)

    return ppmi_mat


def svd_embeddings(
    ppmi_matrix: np.ndarray,
    dim: int = 100,
) -> np.ndarray:
    """
    Compute dense word embeddings via truncated SVD of a PPMI matrix.

    Performs singular value decomposition M = U * S * V^T and returns
    U * sqrt(S) as the word embeddings, corresponding to the factorised
    PPMI matrix M \approx W * C^T where W are the word embeddings.

    Args:
        ppmi_matrix: PPMI matrix of shape (V, V).
        dim: Desired embedding dimensionality. Must be <= min(V, V) - 1.

    Returns:
        Array of shape (V, dim) with dense word embeddings.

    Example:
        >>> ppmi_mat = np.eye(100)
        >>> emb = svd_embeddings(ppmi_mat, dim=50)
        >>> emb.shape
        (100, 50)
    """
    V = ppmi_matrix.shape[0]
    dim = min(dim, V - 1)
    if dim < 1:
        return np.zeros((V, 1))

    U, s, Vt = svds(ppmi_matrix.astype(np.float64), k=dim)
    # svds returns singular values in ascending order; reverse to descending
    U = U[:, ::-1]
    s = s[::-1]
    Vt = Vt[::-1, :]

    # Word embeddings: U * sqrt(S)
    embeddings = U * np.sqrt(s)[np.newaxis, :]
    embeddings[np.isnan(embeddings)] = 0.0
    embeddings[np.isinf(embeddings)] = 0.0

    return embeddings


def economic_similarity(
    word1: str,
    word2: str,
    embeddings: np.ndarray,
    vocab: Dict[str, int],
) -> float:
    """
    Compute cosine similarity between two words in embedding space.

    Args:
        word1: First word (string).
        word2: Second word (string).
        embeddings: Embedding matrix of shape (V, dim).
        vocab: Vocabulary mapping word -> index.

    Returns:
        Cosine similarity in [-1, 1]. Returns 0.0 if either word is
        not in the vocabulary.

    Example:
        >>> import numpy as np
        >>> vocab = {'inflation': 0, 'growth': 1}
        >>> emb = np.array([[0.5, 0.2], [0.1, 0.8]])
        >>> economic_similarity('inflation', 'growth', emb, vocab)
        0.5388...
    """
    if word1 not in vocab or word2 not in vocab:
        return 0.0

    v1 = embeddings[vocab[word1]]
    v2 = embeddings[vocab[word2]]

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0

    sim = np.dot(v1, v2) / (norm1 * norm2)
    return float(np.clip(sim, -1.0, 1.0))


def document_embedding(
    text: str,
    word_embeddings: np.ndarray,
    vocab: Dict[str, int],
    method: str = "average",
) -> np.ndarray:
    """
    Compute a document embedding by aggregating its word embeddings.

    Args:
        text: Input text string.
        word_embeddings: Embedding matrix of shape (V, dim).
        vocab: Vocabulary mapping word -> index.
        method: Aggregation method. Currently only ``"average"`` is
            supported (simple mean of word embeddings in the document).

    Returns:
        Array of shape (dim,) representing the document embedding.
        Returns a zero vector if no in-vocabulary words are found.

    Example:
        >>> import numpy as np
        >>> vocab = {'inflation': 0, 'growth': 1}
        >>> emb = np.array([[0.5, 0.2], [0.1, 0.8]])
        >>> doc_emb = document_embedding("inflation growth", emb, vocab)
        >>> doc_emb.shape
        (2,)
    """
    tokens = clean_text(text)
    dim = word_embeddings.shape[1]

    vectors = []
    for token in tokens:
        if token in vocab:
            vectors.append(word_embeddings[vocab[token]])

    if not vectors:
        return np.zeros(dim)

    if method == "average":
        return np.mean(vectors, axis=0)

    raise ValueError(f"Unknown aggregation method: {method}")
