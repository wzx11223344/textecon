"""
Topic modeling module for economic documents.

Implements Latent Dirichlet Allocation (LDA) using manual Gibbs sampling
-- no external NLP/ML library dependencies.  Suitable for discovering
thematic structures in FOMC statements, central bank communications,
and other economic text corpora.
"""

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np


class LDA:
    """
    Latent Dirichlet Allocation via collapsed Gibbs sampling.

    Implements the standard LDA model (Blei, Ng & Jordan, 2003) with
    symmetric Dirichlet priors.  The Gibbs sampler assigns each word token
    to a topic conditioned on all other assignments.

    Parameters:
        n_topics: Number of topics.
        alpha: Dirichlet prior on document-topic distributions.
        beta: Dirichlet prior on topic-word distributions.
        n_iter: Number of Gibbs sampling iterations.
        random_state: Seed for reproducibility.
    """

    def __init__(self,
                 n_topics: int = 10,
                 alpha: float = 0.1,
                 beta: float = 0.01,
                 n_iter: int = 1000,
                 random_state: int = 42):
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.n_iter = n_iter
        self.random_state = random_state

        # Fitted state
        self.vocab_: Dict[str, int] = {}          # word -> id
        self.inv_vocab_: Dict[int, str] = {}       # id -> word
        self.n_words_: int = 0
        self.n_docs_: int = 0

        # Count matrices
        self.topic_word_: np.ndarray = None        # topic x word
        self.doc_topic_: np.ndarray = None         # doc x topic
        self.topic_totals_: np.ndarray = None      # topic totals

        # Assignment arrays (per-document word-topic assignments)
        self.assignments_: List[np.ndarray] = []

        # Derived
        self.phi_: np.ndarray = None               # topic-word distributions
        self.theta_: np.ndarray = None              # doc-topic distributions

    def _initialize(self, documents: List[List[str]]):
        """Build vocabulary and randomly assign initial topic labels."""
        # Build vocabulary
        word_counts = Counter()
        for doc in documents:
            word_counts.update(doc)
        # Keep all words; order by frequency desc for determinism
        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
        self.vocab_ = {w: i for i, (w, _) in enumerate(sorted_words)}
        self.inv_vocab_ = {i: w for w, i in self.vocab_.items()}
        self.n_words_ = len(self.vocab_)
        self.n_docs_ = len(documents)

        # Initialize count matrices
        self.topic_word_ = np.zeros((self.n_topics, self.n_words_),
                                     dtype=np.int64)
        self.doc_topic_ = np.zeros((self.n_docs_, self.n_topics),
                                    dtype=np.int64)
        self.topic_totals_ = np.zeros(self.n_topics, dtype=np.int64)

        rng = np.random.RandomState(self.random_state)

        # Random initial topic assignments
        self.assignments_ = []
        for d, doc in enumerate(documents):
            doc_assignments = np.zeros(len(doc), dtype=np.int32)
            for i, word in enumerate(doc):
                if word in self.vocab_:
                    w = self.vocab_[word]
                    z = rng.randint(0, self.n_topics)
                    doc_assignments[i] = z
                    self.topic_word_[z, w] += 1
                    self.doc_topic_[d, z] += 1
                    self.topic_totals_[z] += 1
            self.assignments_.append(doc_assignments)

    def fit(self,
            documents: List[List[str]],
            n_topics: int = None,
            n_iter: int = None,
            show_progress: bool = True) -> "LDA":
        """
        Fit the LDA model to a collection of documents.

        Args:
            documents: List of preprocessed token lists, one per document.
            n_topics: Override number of topics (uses ``__init__`` value
                if None).
            n_iter: Override number of iterations.
            show_progress: If True, print progress every 200 iterations.

        Returns:
            self (the fitted model).
        """
        if n_topics is not None:
            self.n_topics = n_topics
        if n_iter is not None:
            self.n_iter = n_iter

        self._initialize(documents)
        rng = np.random.RandomState(self.random_state)

        for it in range(self.n_iter):
            for d, doc in enumerate(documents):
                z_d = self.assignments_[d]
                for i, word in enumerate(doc):
                    if word not in self.vocab_:
                        continue
                    w = self.vocab_[word]
                    old_z = z_d[i]

                    # Decrement counts for this token
                    self.topic_word_[old_z, w] -= 1
                    self.doc_topic_[d, old_z] -= 1
                    self.topic_totals_[old_z] -= 1

                    # Compute conditional probabilities
                    # p(z | *) ∝ (n_dz + alpha) * (n_zw + beta) / (n_z + W*beta)
                    scores = np.zeros(self.n_topics)
                    for z in range(self.n_topics):
                        doc_term = self.doc_topic_[d, z] + self.alpha
                        word_term = (self.topic_word_[z, w] + self.beta) / \
                                    (self.topic_totals_[z] +
                                     self.n_words_ * self.beta)
                        scores[z] = doc_term * word_term

                    # Sample new topic
                    scores /= scores.sum()
                    new_z = np.searchsorted(
                        np.cumsum(scores),
                        rng.random()
                    )
                    new_z = min(new_z, self.n_topics - 1)

                    # Increment counts
                    z_d[i] = new_z
                    self.topic_word_[new_z, w] += 1
                    self.doc_topic_[d, new_z] += 1
                    self.topic_totals_[new_z] += 1

            if show_progress and (it + 1) % 200 == 0:
                print(f"  LDA iteration {it + 1}/{self.n_iter}")

        # Compute final distributions
        self._compute_phi()
        self._compute_theta()
        return self

    def _compute_phi(self):
        """Compute topic-word distributions (phi)."""
        self.phi_ = np.zeros((self.n_topics, self.n_words_))
        for z in range(self.n_topics):
            denom = self.topic_totals_[z] + self.n_words_ * self.beta
            self.phi_[z, :] = (self.topic_word_[z, :] + self.beta) / denom

    def _compute_theta(self):
        """Compute document-topic distributions (theta)."""
        self.theta_ = np.zeros((self.n_docs_, self.n_topics))
        for d in range(self.n_docs_):
            denom = self.doc_topic_[d, :].sum() + self.n_topics * self.alpha
            self.theta_[d, :] = (self.doc_topic_[d, :] + self.alpha) / denom

    def get_topics(self, n_words: int = 10) -> List[List[str]]:
        """
        Return the top words for each topic.

        Args:
            n_words: Number of top words to return per topic.

        Returns:
            List of lists, where each inner list contains the top words
            for one topic, ordered by probability.
        """
        if self.phi_ is None:
            raise RuntimeError("Model must be fitted before calling get_topics().")
        topics = []
        for z in range(self.n_topics):
            top_indices = np.argsort(self.phi_[z, :])[::-1][:n_words]
            top_words = [self.inv_vocab_[i] for i in top_indices]
            topics.append(top_words)
        return topics

    def transform(self,
                  documents: List[List[str]],
                  n_iter: int = 200) -> np.ndarray:
        """
        Infer topic distributions for new documents.

        Uses Gibbs sampling with the learned topic-word distributions held
        fixed.

        Args:
            documents: List of preprocessed token lists.
            n_iter: Number of Gibbs sampling iterations for inference.

        Returns:
            Array of shape ``(n_docs, n_topics)`` with topic proportions.
        """
        if self.phi_ is None:
            raise RuntimeError("Model must be fitted before calling transform().")

        rng = np.random.RandomState(self.random_state)
        n_new = len(documents)
        theta_new = np.zeros((n_new, self.n_topics))

        for d, doc in enumerate(documents):
            # Filter to in-vocabulary words
            valid_words = []
            valid_ids = []
            for word in doc:
                if word in self.vocab_:
                    valid_words.append(word)
                    valid_ids.append(self.vocab_[word])

            if len(valid_words) == 0:
                # Uniform prior
                theta_new[d, :] = 1.0 / self.n_topics
                continue

            N = len(valid_words)
            z = rng.randint(0, self.n_topics, size=N)
            n_dz = np.zeros(self.n_topics)

            for i, w_idx in enumerate(valid_ids):
                n_dz[z[i]] += 1

            for _ in range(n_iter):
                for i, w_idx in enumerate(valid_ids):
                    old_z = z[i]
                    n_dz[old_z] -= 1

                    scores = np.zeros(self.n_topics)
                    for k in range(self.n_topics):
                        doc_term = n_dz[k] + self.alpha
                        word_term = self.phi_[k, w_idx]
                        scores[k] = doc_term * word_term

                    scores /= scores.sum()
                    new_z = np.searchsorted(np.cumsum(scores), rng.random())
                    new_z = min(new_z, self.n_topics - 1)

                    z[i] = new_z
                    n_dz[new_z] += 1

            theta_new[d, :] = (n_dz + self.alpha) / (N + self.n_topics * self.alpha)

        return theta_new

    def topic_evolution(self,
                        documents: List[List[str]],
                        timestamps: List[str]) -> Dict[str, np.ndarray]:
        """
        Track how topic prevalence changes over time.

        For each unique timestamp, computes the average topic distribution
        across all documents belonging to that time period.

        Args:
            documents: List of preprocessed token lists.
            timestamps: List of corresponding timestamps.

        Returns:
            Dictionary with keys:
            - ``timestamps``: list of unique timestamps (sorted).
            - ``prevalence``: array of shape ``(n_timestamps, n_topics)``
              with mean topic proportions per time period.
        """
        theta = self.transform(documents)
        unique_ts = sorted(set(timestamps))
        ts_to_idx = {t: i for i, t in enumerate(unique_ts)}
        n_ts = len(unique_ts)

        prevalence = np.zeros((n_ts, self.n_topics))
        counts = np.zeros(n_ts)

        for i, ts in enumerate(timestamps):
            idx = ts_to_idx[ts]
            prevalence[idx] += theta[i]
            counts[idx] += 1

        for i in range(n_ts):
            if counts[i] > 0:
                prevalence[i] /= counts[i]

        return {
            "timestamps": unique_ts,
            "prevalence": prevalence,
        }
