# TextEcon: NLP for Economics

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Text-as-data methods for economic research.** A lightweight, dependency-minimal toolkit for applying natural language processing techniques to economic text corpora.

## Features

- **Preprocessing** -- Tokenization, lemmatization, stopword removal, vocabulary building, and bag-of-words conversion (zero NLTK dependency).
- **Economic Sentiment** -- Domain-specific sentiment lexicons for hawkish/dovish monetary policy stance detection and positive/negative tone measurement. Includes an Economic Policy Uncertainty (EPU) index approximation.
- **Topic Modeling** -- Manual Gibbs sampling implementation of Latent Dirichlet Allocation (LDA) for discovering thematic structures in economic documents.
- **Word Embeddings** -- Co-occurrence matrix construction, PPMI weighting, truncated SVD embeddings, and economic-domain cosine similarity.
- **Built-in Corpora** -- FOMC statements, Fed minutes summaries, and economic news headlines for immediate experimentation.

## Installation

```bash
git clone https://github.com/your-org/textecon.git
cd textecon
pip install -e .
```

## Quick Start

```python
from textecon import load_fomc_statements, economic_sentiment, LDA

# Load built-in FOMC corpus
corpus = load_fomc_statements()

# Analyze monetary policy sentiment
text = "The Committee decided to raise the target range for the federal funds rate."
result = economic_sentiment(text)
print(f"Hawkish: {result['hawkish']:.3f}, Dovish: {result['dovish']:.3f}")

# Run topic modeling
lda = LDA(n_topics=5, n_iter=1000)
lda.fit([t for t, _, _ in corpus])
topics = lda.get_topics(n_words=10)
for i, words in enumerate(topics):
    print(f"Topic {i}: {words}")
```

## Dependencies

- Python 3.8+
- numpy
- scipy

Optional (for examples/demo.py):
- matplotlib

## Project Structure

```
textecon/
├── README.md
├── setup.py
├── requirements.txt
├── LICENSE
├── .gitignore
├── textecon/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── sentiment.py
│   ├── topic.py
│   ├── embeddings.py
│   └── corpora.py
└── examples/
    └── demo.py
```

## Citation

If you use TextEcon in your research, please cite:

```
@software{textecon2026,
  title = {TextEcon: NLP for Economics},
  year = {2026},
  url = {https://github.com/your-org/textecon}
}
```

## License

MIT License -- see [LICENSE](LICENSE) for details.
