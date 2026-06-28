"""
TextEcon Demo -- NLP for Economics Toolkit
==========================================

This script demonstrates the core capabilities of TextEcon:
1. Load built-in economic corpora (FOMC statements, economic news)
2. Preprocess and clean economic text
3. Analyze monetary policy sentiment (hawkish/dovish)
4. Run LDA topic modeling on FOMC statements
5. Build word embeddings and compute economic similarity
6. Compute an EPU index approximation
"""

import sys
import os

# Add parent directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from textecon.preprocess import tokenize, lemmatize, clean_text, build_vocab, text_to_bow
from textecon.sentiment import economic_sentiment, sentiment_timeseries, policy_uncertainty_index
from textecon.topic import LDA
from textecon.embeddings import (
    cooccurrence_matrix, ppmi, svd_embeddings,
    economic_similarity, document_embedding,
)
from textecon.corpora import load_fomc_statements, load_economic_news

SEPARATOR = "=" * 70


def demo_preprocessing():
    """Demonstrate text preprocessing pipeline."""
    print(SEPARATOR)
    print("1. TEXT PREPROCESSING")
    print(SEPARATOR)

    sample = (
        "The Committee decided to raise the target range for the federal "
        "funds rate by 75 basis points to combat persistent inflationary "
        "pressures. The labor market remains extremely tight."
    )

    print(f"\n  Original text:\n    \"{sample}\"\n")

    # Tokenization
    tokens = tokenize(sample)
    print(f"  Tokens ({len(tokens)}): {tokens[:15]}...")

    # Lemmatization
    lemmas = [lemmatize(t) for t in tokens]
    print(f"  Lemmatized: {lemmas[:15]}...")

    # Full pipeline
    cleaned = clean_text(sample)
    print(f"  Cleaned ({len(cleaned)}): {cleaned}")

    # Vocabulary building
    docs = [
        clean_text("The economy is growing strongly with low inflation."),
        clean_text("Inflation remains elevated despite aggressive tightening."),
        clean_text("Growth moderated as the labor market shows signs of cooling."),
    ]
    vocab = build_vocab(docs, min_freq=1)
    print(f"\n  Vocabulary size: {len(vocab)}")
    print(f"  Top 10 words: {list(vocab.keys())[:10]}")

    # Text-to-bow
    idx, cnt = text_to_bow("growth inflation policy", vocab)
    print(f"\n  BoW for 'growth inflation policy': {list(zip(idx, cnt))}")


def demo_sentiment():
    """Demonstrate economic sentiment analysis."""
    print(f"\n{SEPARATOR}")
    print("2. ECONOMIC SENTIMENT ANALYSIS")
    print(SEPARATOR)

    # Load FOMC corpus
    corpus = load_fomc_statements()
    print(f"\n  Loaded {len(corpus)} FOMC statements")

    # Analyze each statement
    print("\n  Sentiment results:")
    for text, date, label in corpus:
        result = economic_sentiment(text)
        print(
            f"    [{date:>12}] Stance: {result['stance']:>8} "
            f"(H:{result['hawkish']:+.2f}, D:{result['dovish']:+.2f}) "
            f"| Tone: {result['tone']:>8} "
            f"(P:{result['positive']:+.2f}, N:{result['negative']:+.2f}) "
            f"| True: {label}"
        )

    # Evaluate accuracy
    correct = 0
    for text, date, label in corpus:
        result = economic_sentiment(text)
        if result["stance"] == label:
            correct += 1
    print(f"\n  Stance classification accuracy: {correct}/{len(corpus)} "
          f"({100*correct/len(corpus):.1f}%)")

    # Sentiment timeseries
    texts = [t for t, _, _ in corpus]
    dates = [d for _, d, _ in corpus]
    ts_results = sentiment_timeseries(texts, dates)

    hawkish_scores = [r["hawkish"] for r in ts_results]
    avg_hawkish = sum(hawkish_scores) / len(hawkish_scores)
    print(f"  Average hawkish score: {avg_hawkish:+.3f}")

    # EPU index
    epu = policy_uncertainty_index(texts)
    print(f"  EPU Index (proportion of EPU-relevant docs): {epu[0]:.4f}")


def demo_topic_modeling():
    """Demonstrate LDA topic modeling on economic text."""
    print(f"\n{SEPARATOR}")
    print("3. LDA TOPIC MODELING")
    print(SEPARATOR)

    # Load and preprocess corpus
    corpus = load_fomc_statements()
    texts = [t for t, _, _ in corpus]
    dates = [d for _, d, _ in corpus]

    documents = [clean_text(t) for t in texts]
    print(f"\n  Preprocessed {len(documents)} documents")
    vocab_size = len(set(w for doc in documents for w in doc))
    print(f"  Vocabulary size: {vocab_size}")

    # Fit LDA
    print("\n  Fitting LDA with 5 topics...")
    lda = LDA(n_topics=5, alpha=0.5, beta=0.1, n_iter=1500, random_state=42)
    lda.fit(documents, show_progress=True)

    # Show topics
    print("\n  Discovered topics (top 8 words each):")
    topics = lda.get_topics(n_words=8)
    for i, words in enumerate(topics):
        print(f"    Topic {i+1}: {', '.join(words)}")

    # Show document-topic distributions for first 3 docs
    print("\n  Document-topic distributions (first 3 docs):")
    for d in range(min(3, len(documents))):
        top_topics = sorted(
            enumerate(lda.theta_[d]),
            key=lambda x: -x[1]
        )[:3]
        topic_str = ", ".join(
            f"T{t} ({p:.3f})" for t, p in top_topics
        )
        print(f"    Doc {d+1} ({dates[d]}): {topic_str}")


def demo_embeddings():
    """Demonstrate word embeddings and economic similarity."""
    print(f"\n{SEPARATOR}")
    print("4. WORD EMBEDDINGS")
    print(SEPARATOR)

    # Load and preprocess
    corpus = load_fomc_statements()
    documents = [clean_text(t) for t, _, _ in corpus]
    vocab = build_vocab(documents, min_freq=2)
    print(f"\n  Vocabulary size (min_freq=2): {len(vocab)}")

    if len(vocab) < 5:
        print("  [SKIP] Not enough vocabulary for embedding demo")
        return

    # Build co-occurrence matrix
    print("  Building co-occurrence matrix (window=5)...")
    cooc, vocab_cooc = cooccurrence_matrix(documents, vocab=vocab, window=5)
    print(f"  Co-occurrence matrix: {cooc.shape}")

    # PPMI
    print("  Computing PPMI...")
    ppmi_mat = ppmi(cooc)
    nonzero = (ppmi_mat > 0).sum()
    print(f"  PPMI non-zero entries: {nonzero} ({100*nonzero/ppmi_mat.size:.1f}%)")

    # SVD embeddings
    emb_dim = min(50, len(vocab) - 1)
    print(f"  Computing SVD embeddings (dim={emb_dim})...")
    try:
        embeddings = svd_embeddings(ppmi_mat, dim=emb_dim)
        print(f"  Embeddings shape: {embeddings.shape}")
    except Exception as e:
        print(f"  [WARNING] SVD failed: {e}")
        return

    # Compute similarities (use lemmatized forms matching the vocabulary)
    print("\n  Economic word similarities:")
    print("    (Note: words must match lemmatized vocabulary forms)")
    # Sample of actual vocab entries for reference
    sample_vocab = list(vocab.keys())[:15]
    print(f"    Sample vocab: {sample_vocab}")
    word_pairs = [
        ("inflate", "economic"),
        ("employ", "gain"),
        ("policy", "range"),
        ("activity", "job"),
        ("inflate", "remain"),
    ]
    for w1, w2 in word_pairs:
        sim = economic_similarity(w1, w2, embeddings, vocab)
        status = "OK" if sim != 0.0 else "(OOV)"
        print(f"    {w1:>12} <-> {w2:<12}: {sim:+.4f} {status}")

    # Document embedding
    print("\n  Document embeddings (first 3 docs):")
    inv_vocab = {v: k for k, v in vocab.items()}
    for d in range(min(3, len(documents))):
        doc_emb = document_embedding(
            corpus[d][0], embeddings, vocab
        )
        norm = sum(abs(doc_emb))
        print(f"    Doc {d+1}: norm={norm:.4f}")


def main():
    """Run the full TextEcon demo."""
    print("\n" + " " * 15 + "TEXTCON DEMO -- NLP for Economics")
    print(" " * 20 + "=" * 30)

    try:
        demo_preprocessing()
        demo_sentiment()
        demo_topic_modeling()
        demo_embeddings()

        print(f"\n{SEPARATOR}")
        print("DEMO COMPLETE")
        print(SEPARATOR)
        print("\nAll modules loaded and executed successfully.")
        print("TextEcon is ready for economic NLP research!")
    except Exception as e:
        print(f"\n[ERROR] Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
