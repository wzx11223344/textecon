"""
Sentiment analysis module for economic text.

Provides domain-specific sentiment lexicons and scoring functions for
monetary policy stance (hawkish/dovish), general economic tone
(positive/negative), and an Economic Policy Uncertainty (EPU) index
approximation.
"""

import re
from collections import Counter
from typing import Dict, List, Tuple

from textecon.preprocess import tokenize


# ===========================================================================
# Sentiment Lexicons
# ===========================================================================

# Hawkish terms: signal tightening monetary policy (rate hikes, inflation
# control, reducing accommodation).
HAWKISH_TERMS = {
    "tighten", "tightening", "tightened", "tight",
    "hike", "hiking", "hiked", "hikes",
    "hawkish", "hawks", "hawk",
    "inflation", "inflationary",
    "overheat", "overheating",
    "restrictive", "restriction", "restrict",
    "withdraw", "withdrawal", "withdrawing",
    "reduce", "reduction", "reducing", "reduced",
    "unwind", "unwinding",
    "normalize", "normalization", "normalizing",
    "firm", "firming", "firmed",
    "rise", "rising", "rose",
    "accelerate", "accelerating", "acceleration",
    "upside", "upward", "upwardly",
    "pressure", "pressures",
    "wage", "wages",
    "vigilant", "vigilance",
    "contain", "containing", "containment",
    "anchor", "anchoring", "anchored",
    "robust", "robustly",
    "strong", "strengthen", "strengthening",
    "solid", "solidly",
    "broaden", "broadening", "broadened",
    "elevated", "elevate",
    "persistent", "persistence",
    "above", "exceed", "exceeding",
    "target", "targets",
}

# Dovish terms: signal easing monetary policy (rate cuts, stimulus,
# accommodative stance).
DOVISH_TERMS = {
    "ease", "easing", "eased",
    "dovish", "doves", "dove",
    "accommodative", "accommodation",
    "stimulus", "stimulate", "stimulative", "stimulating",
    "support", "supportive", "supporting", "supported",
    "cut", "cuts", "cutting",
    "lower", "lowering", "lowered",
    "below", "undershoot",
    "slack", "sluggish", "sluggishly",
    "subdued", "subduedly",
    "moderate", "moderating", "moderation",
    "slowdown", "slowing", "slowed", "slow",
    "weak", "weakness", "weakening", "weakened",
    "soft", "softening", "softened",
    "dovishness",
    "disinflation", "disinflationary",
    "deflation", "deflationary",
    "patient", "patience", "patiently",
    "gradual", "gradually", "gradualism",
    "measured", "cautious", "cautiously",
    "accommodate",
    "deteriorate", "deterioration", "deteriorating",
    "contraction", "contractionary",
    "recession", "recessionary",
    "headwind", "headwinds",
    "drag", "dragging",
    "downside", "downward", "downwardly",
    "decline", "declining", "declined",
    "drop", "dropping", "dropped",
    "fall", "falling", "fell",
}

# Positive tone terms for economic context.
POSITIVE_TERMS = {
    "growth", "growing", "grew", "grow",
    "expand", "expansion", "expanding", "expanded",
    "improve", "improvement", "improving", "improved",
    "gain", "gains", "gaining", "gained",
    "recovery", "recover", "recovering", "recovered",
    "strength", "strengthen", "strong",
    "positive", "positively",
    "progress", "advance", "advancing", "advanced",
    "resilient", "resilience",
    "stable", "stability", "stabilize", "stabilizing",
    "balanced", "balance", "balancing",
    "favorable", "favorably",
    "increase", "increasing", "increased",
    "boost", "boosting", "boosted",
    "momentum",
    "optimistic", "optimism",
    "confident", "confidence",
    "sustain", "sustained", "sustainable",
    "prosperity", "prosperous",
    "employment", "employ",
    "output",
    "productivity", "productive",
    "innovation", "innovative",
    "investment", "invest", "investing",
    "demand",
}

# Negative tone terms for economic context.
NEGATIVE_TERMS = {
    "crisis", "crises",
    "recession", "recessionary",
    "depression",
    "contraction", "contract", "contracting", "contracted",
    "decline", "declining", "declined",
    "drop", "dropping", "dropped",
    "downturn",
    "shock", "shocks",
    "risk", "risks", "risky",
    "uncertainty", "uncertain", "uncertainties",
    "volatile", "volatility",
    "worry", "worries", "worrying", "worried",
    "concern", "concerns", "concerning", "concerned",
    "fear", "fears",
    "fragile", "fragility",
    "weak", "weakness", "weakening", "weakened",
    "loss", "losses",
    "negative", "negatively",
    "worse", "worsen", "worsening",
    "adverse", "adversely",
    "disruption", "disruptive", "disrupt",
    "turbulence", "turbulent",
    "strain", "strains",
    "distress", "distressed",
    "default", "defaults",
    "unemployment", "unemployed",
    "deficit", "deficits",
    "debt", "debts", "indebted",
    "burden", "burdensome",
    "imbalance", "imbalances",
    "distort", "distortion", "distortions",
    "tight", "tightening",
    "squeeze", "squeezing",
    "spike", "spikes", "spiking",
    "surge", "surging", "surged",
    "bubble", "bubbles",
    "correction",
    "sell", "selloff",
    "plunge", "plunging", "plunged",
    "collapse", "collapsing", "collapsed",
    "failure", "failures", "failing", "failed",
    "panic",
    "frozen", "freeze",
    "emergency",
}

# Economic Policy Uncertainty terms (Baker, Bloom & Davis, 2016)
EPU_TERMS_ECONOMY = {
    "economic", "economics", "economy", "economies",
}

EPU_TERMS_UNCERTAINTY = {
    "uncertain", "uncertainty", "uncertainties",
    "uncertainly",
}

EPU_TERMS_POLICY = {
    "regulation", "regulations", "regulatory",
    "legislation", "legislative",
    "congress", "congressional",
    "senate", "senator",
    "house", "representative",
    "president", "presidential",
    "administration",
    "white house",
    "federal reserve", "fed", "fomc",
    "treasury",
    "deficit", "deficits",
    "fiscal",
    "monetary",
    "tax", "taxes", "taxation",
    "spending", "expenditure",
    "budget", "budgets", "budgetary",
    "subsidy", "subsidies",
    "tariff", "tariffs",
    "trade policy", "trade policies",
    "central bank", "central banks",
    "governor", "governors",
    "policy", "policies",
    "policymaker", "policymakers",
    "reform", "reforms",
    "stimulus",
    "bailout", "bailouts",
    "quantitative easing",
    "interest rate", "interest rates",
    "minimum wage",
    "entitlement", "entitlements",
    "mandate",
}

# Set of EPU policy terms decomposed into individual words for token matching.
_EPU_POLICY_TOKENS = set()
for term in EPU_TERMS_POLICY:
    for w in term.split():
        _EPU_POLICY_TOKENS.add(w)

# Ensure single-word variants cover key policy tokens
_EPU_POLICY_TOKENS.update({
    "policy", "policies", "regulation", "regulations", "regulatory",
    "legislation", "legislative", "congress", "fiscal", "monetary",
    "tax", "taxes", "taxation", "spending", "expenditure", "budget",
    "deficit", "deficits", "reform", "reforms", "stimulus", "tariff",
    "tariffs", "subsidy", "subsidies", "bailout", "mandate",
    "treasury", "central", "fed", "fomc",
})


# ===========================================================================
# Scoring functions
# ===========================================================================

def _count_terms(tokens: List[str], lexicon: set) -> int:
    """Count how many tokens appear in the given lexicon."""
    return sum(1 for t in tokens if t in lexicon)


def _score(tokens: List[str], pos_lex: set, neg_lex: set) -> float:
    """
    Compute a normalized sentiment score.

    Returns a value in [-1, 1] where positive values indicate dominance
    of the positive lexicon and negative values indicate dominance of
    the negative lexicon.
    """
    pos_count = _count_terms(tokens, pos_lex)
    neg_count = _count_terms(tokens, neg_lex)
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    return (pos_count - neg_count) / total


def _predict_label(score: float, threshold: float = 0.1) -> str:
    """Convert a continuous score to a discrete label."""
    if score > threshold:
        return "positive"
    elif score < -threshold:
        return "negative"
    return "neutral"


# ===========================================================================
# Public API
# ===========================================================================

def economic_sentiment(text: str) -> Dict[str, object]:
    """
    Analyze the economic sentiment of a text.

    Computes hawkish/dovish stance scores (monetary policy orientation)
    and positive/negative tone scores (general economic sentiment).

    Args:
        text: Input text string (e.g., FOMC statement, speech).

    Returns:
        Dictionary with the following keys:
        - ``hawkish`` (float): Hawkish score in [-1, 1].
        - ``dovish`` (float): Dovish score in [-1, 1] (inverse of hawkish).
        - ``stance`` (str): ``"hawkish"``, ``"dovish"``, or ``"neutral"``.
        - ``positive`` (float): Positive tone score in [-1, 1].
        - ``negative`` (float): Negative tone score in [-1, 1].
        - ``tone`` (str): ``"positive"``, ``"negative"``, or ``"neutral"``.
        - ``hawkish_count`` (int): Number of hawkish terms found.
        - ``dovish_count`` (int): Number of dovish terms found.
        - ``positive_count`` (int): Number of positive terms found.
        - ``negative_count`` (int): Number of negative terms found.

    Example:
        >>> result = economic_sentiment("The Committee decided to raise rates "
        ...                             "to combat persistent inflation.")
        >>> result['stance']
        'hawkish'
        >>> result['hawkish'] > 0
        True
    """
    tokens = tokenize(text)

    hawk_count = _count_terms(tokens, HAWKISH_TERMS)
    dove_count = _count_terms(tokens, DOVISH_TERMS)
    pos_count = _count_terms(tokens, POSITIVE_TERMS)
    neg_count = _count_terms(tokens, NEGATIVE_TERMS)

    # Hawkish/dovish score: higher = more hawkish
    total_hd = hawk_count + dove_count
    if total_hd > 0:
        hawkish_score = (hawk_count - dove_count) / total_hd
    else:
        hawkish_score = 0.0
    dovish_score = -hawkish_score

    # Stance label
    if hawkish_score > 0.1:
        stance = "hawkish"
    elif hawkish_score < -0.1:
        stance = "dovish"
    else:
        stance = "neutral"

    # Positive/negative tone
    total_pn = pos_count + neg_count
    if total_pn > 0:
        positive_score = (pos_count - neg_count) / total_pn
    else:
        positive_score = 0.0
    negative_score = -positive_score

    if positive_score > 0.1:
        tone = "positive"
    elif positive_score < -0.1:
        tone = "negative"
    else:
        tone = "neutral"

    return {
        "hawkish": round(hawkish_score, 4),
        "dovish": round(dovish_score, 4),
        "stance": stance,
        "positive": round(positive_score, 4),
        "negative": round(negative_score, 4),
        "tone": tone,
        "hawkish_count": hawk_count,
        "dovish_count": dove_count,
        "positive_count": pos_count,
        "negative_count": neg_count,
    }


def sentiment_timeseries(
    documents: List[str],
    timestamps: List[str]
) -> List[Dict[str, object]]:
    """
    Compute economic sentiment for a time-ordered list of documents.

    Args:
        documents: List of text documents.
        timestamps: List of corresponding timestamps (strings, e.g. dates).

    Returns:
        List of dictionaries, each containing ``timestamp`` and the
        sentiment scores from :func:`economic_sentiment`.

    Example:
        >>> docs = ["Economy is growing strongly.",
        ...         "Recession fears mount as markets plunge."]
        >>> ts = ["2024-01", "2024-03"]
        >>> results = sentiment_timeseries(docs, ts)
        >>> results[0]['tone']
        'positive'
    """
    results = []
    for doc, ts in zip(documents, timestamps):
        sent = economic_sentiment(doc)
        sent["timestamp"] = ts
        results.append(sent)
    return results


def policy_uncertainty_index(texts: List[str],
                              normalize: bool = True) -> List[float]:
    """
    Compute an Economic Policy Uncertainty (EPU) index approximation.

    Based on the methodology of Baker, Bloom & Davis (2016). An article is
    counted as EPU-relevant if it contains at least one term from each of
    three categories: economy, uncertainty, and policy.

    Args:
        texts: List of text documents (articles, speeches, etc.).
        normalize: If True, returns the fraction of documents that are
            EPU-relevant (proportion). If False, returns raw counts.

    Returns:
        If ``normalize=True``, a single float in [0, 1] representing the
        proportion of EPU-relevant documents.
        If ``normalize=False``, a list of 0/1 values per document.

    Example:
        >>> articles = [
        ...     "Economic uncertainty amid new fiscal policy debates.",
        ...     "The weather is nice today.",
        ...     "Uncertainty surrounds the economic outlook as Congress "
        ...     "debates tax reform.",
        ... ]
        >>> policy_uncertainty_index(articles)
        0.6667
    """
    results = []
    for text in texts:
        tokens = tokenize(text)
        tokens_set = set(tokens)

        has_economy = bool(tokens_set & EPU_TERMS_ECONOMY)
        has_uncertainty = bool(tokens_set & EPU_TERMS_UNCERTAINTY)
        has_policy = bool(tokens_set & _EPU_POLICY_TOKENS)

        is_epu = 1 if (has_economy and has_uncertainty and has_policy) else 0
        results.append(is_epu)

    if normalize:
        if len(results) == 0:
            return [0.0]
        return [round(sum(results) / len(results), 4)]
    return results
