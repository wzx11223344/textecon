"""
Built-in economic text corpora for TextEcon.

Provides curated datasets for immediate experimentation with economic
NLP methods, including FOMC statements, Fed minutes summaries, and
economic news headlines.  All data is synthetic / illustrative and
designed for educational and methodological research purposes.
"""

from typing import List, Tuple


# ===========================================================================
# FOMC Statements
# ===========================================================================
# Illustrative excerpts from Federal Open Market Committee statements
# spanning hawkish, dovish, and neutral policy stances.

FOMC_STATEMENTS: List[Tuple[str, str, str]] = [
    # (text, date, sentiment_label)
    # ---- Hawkish statements ----
    (
        "The Committee decided to raise the target range for the federal "
        "funds rate by 75 basis points to combat persistent inflationary "
        "pressures. Economic activity has been expanding at a robust pace, "
        "with strong job gains and elevated inflation reflecting supply "
        "and demand imbalances. The Committee remains highly attentive to "
        "inflation risks and is prepared to adjust the stance of monetary "
        "policy as appropriate if risks emerge.",
        "2022-09-21",
        "hawkish",
    ),
    (
        "Inflation remains well above the Committee's longer-run objective "
        "of 2 percent. The labor market remains extremely tight, putting "
        "upward pressure on wages and prices. The Committee judges that "
        "ongoing increases in the target range will be appropriate to "
        "return inflation to 2 percent over time. The Committee is "
        "strongly committed to restoring price stability.",
        "2022-11-02",
        "hawkish",
    ),
    (
        "Recent indicators point to solid growth in spending and "
        "production. Job gains have been robust and the unemployment "
        "rate has remained low. Inflation has eased somewhat but "
        "remains elevated. The Committee anticipates that ongoing "
        "increases in the target range will be appropriate to attain "
        "a stance of monetary policy that is sufficiently restrictive.",
        "2023-02-01",
        "hawkish",
    ),
    (
        "The Committee seeks to achieve maximum employment and inflation "
        "at the rate of 2 percent over the longer run. In support of "
        "these goals, the Committee decided to raise the target range. "
        "The Committee will continue to monitor the implications of "
        "incoming information for the economic outlook and would be "
        "prepared to adjust the stance of monetary policy if risks "
        "emerge that could impede attainment of Committee goals.",
        "2018-06-13",
        "hawkish",
    ),
    # ---- Dovish statements ----
    (
        "The Committee decided to lower the target range for the federal "
        "funds rate to support the economy during this challenging period. "
        "The coronavirus outbreak is causing tremendous hardship and "
        "disruption across the global economy. The Committee expects to "
        "maintain this accommodative stance until the economy has "
        "weathered recent events and is on track to achieve its maximum "
        "employment and price stability goals.",
        "2020-03-15",
        "dovish",
    ),
    (
        "The Committee is committed to using its full range of tools to "
        "support the economy in this challenging time, thereby promoting "
        "its maximum employment and price stability goals. The path of "
        "the economy will depend significantly on the course of the virus. "
        "The ongoing public health crisis will continue to weigh on "
        "economic activity, employment, and inflation in the near term.",
        "2020-09-16",
        "dovish",
    ),
    (
        "The Committee decided to maintain the target range at its current "
        "level. Indicators of spending and production have softened. Job "
        "gains have moderated but remain solid, and the unemployment rate "
        "has moved up but remains low. Inflation has eased over the past "
        "year but remains elevated. The Committee will carefully assess "
        "incoming data in determining the extent of additional policy "
        "firming that may be appropriate.",
        "2024-06-12",
        "dovish",
    ),
    (
        "The Committee will continue to assess the economic outlook in "
        "light of incoming information. Global economic and financial "
        "developments warrant a patient approach to determining future "
        "adjustments in the stance of monetary policy. The Committee "
        "stands ready to use its tools as appropriate to sustain the "
        "expansion and promote a strong labor market.",
        "2019-03-20",
        "dovish",
    ),
    # ---- Neutral / balanced statements ----
    (
        "The Committee decided to keep the target range for the federal "
        "funds rate unchanged. Economic activity has been expanding at "
        "a moderate pace. Job gains have been solid, and the unemployment "
        "rate has remained low. Inflation continues to run below 2 "
        "percent. The Committee will continue to monitor economic and "
        "financial developments and assess their implications for the "
        "economic outlook.",
        "2017-05-03",
        "neutral",
    ),
    (
        "Information received since the Committee met in January indicates "
        "that the labor market has continued to strengthen and that "
        "economic activity has continued to expand at a moderate pace. "
        "Job gains remained solid and the unemployment rate stayed near "
        "its recent low. On a 12-month basis, inflation has continued to "
        "run somewhat below the Committee's longer-run objective.",
        "2017-03-15",
        "neutral",
    ),
    (
        "The Committee seeks to foster maximum employment and price "
        "stability. The Committee expects that with appropriate "
        "firming in the stance of monetary policy, economic activity "
        "will expand at a moderate pace and labor market conditions "
        "will remain strong. Inflation on a 12-month basis is expected "
        "to move toward the Committee's symmetric 2 percent objective.",
        "2023-05-03",
        "neutral",
    ),
    (
        "The Committee is maintaining the target range at the current "
        "level while continuing to assess the cumulative effects of "
        "past policy actions. The economic outlook is uncertain and "
        "the Committee remains highly attentive to risks on both "
        "sides of its dual mandate. The timing of any adjustment to "
        "the stance of policy will depend on the evolution of the data.",
        "2024-03-20",
        "neutral",
    ),
]


# ===========================================================================
# Fed Minutes Summaries
# ===========================================================================
# Abbreviated summaries of Federal Reserve meeting minutes.

FED_MINUTES_SUMMARIES: List[str] = [
    (
        "Participants noted that inflation had eased over the past year "
        "but remained elevated. Many participants emphasized the need to "
        "maintain a restrictive policy stance until inflation was clearly "
        "on a path toward the 2 percent objective. Some participants "
        "expressed concern about the lagged effects of cumulative monetary "
        "tightening on economic activity."
    ),
    (
        "Staff presented an analysis suggesting that labor market tightness "
        "was gradually easing, with job openings declining and wage growth "
        "moderating. Participants discussed the risks to the outlook, with "
        "several noting that upside risks to inflation had diminished while "
        "downside risks to employment had increased."
    ),
    (
        "Discussion turned to the appropriate path of monetary policy. "
        "Several participants noted the importance of communicating that "
        "decisions would remain data-dependent. A number of participants "
        "observed that maintaining the current target range for some time "
        "would likely be appropriate if the economy evolved as expected."
    ),
    (
        "Participants generally assessed that the financial system remained "
        "sound and resilient. They noted that credit conditions had "
        "tightened somewhat for households and businesses, but that the "
        "financial sector overall remained well capitalized with ample "
        "liquidity."
    ),
    (
        "The staff provided an update on global economic developments, "
        "noting that growth abroad had been modest but that risks remained "
        "tilted to the downside. Trade policy uncertainty and geopolitical "
        "tensions were cited as potential headwinds to the global outlook."
    ),
]


# ===========================================================================
# Economic News Headlines
# ===========================================================================
# Illustrative economic news headlines with dates.

ECONOMIC_NEWS: List[Tuple[str, str]] = [
    (
        "US Economy Adds 275,000 Jobs in February, Beating Expectations",
        "2024-02-15",
    ),
    (
        "Federal Reserve Holds Rates Steady, Signals Three Cuts This Year",
        "2024-03-20",
    ),
    (
        "Inflation Cools to 3.1% as Consumer Spending Moderates",
        "2024-01-11",
    ),
    (
        "GDP Growth Accelerates to 4.9% in Third Quarter",
        "2023-10-26",
    ),
    (
        "Housing Market Shows Signs of Recovery as Mortgage Rates Ease",
        "2024-04-05",
    ),
    (
        "Treasury Yields Surge on Strong Labor Market Data",
        "2024-01-05",
    ),
    (
        "Supply Chain Pressures Ease to Pre-Pandemic Levels",
        "2023-11-15",
    ),
    (
        "Consumer Confidence Index Rises for Third Straight Month",
        "2024-03-28",
    ),
    (
        "Fed Chair Powell Says Rate Cuts Likely Appropriate This Year",
        "2024-02-01",
    ),
    (
        "Manufacturing Sector Contracts for Fifth Consecutive Month",
        "2024-02-29",
    ),
    (
        "Trade Deficit Narrows as Exports Hit Record High",
        "2023-12-20",
    ),
    (
        "Retail Sales Unexpectedly Decline Amid Consumer Caution",
        "2024-01-17",
    ),
    (
        "Oil Prices Climb on Geopolitical Tensions and Supply Concerns",
        "2024-04-10",
    ),
    (
        "US Dollar Strengthens as Markets Reassess Rate Cut Timeline",
        "2024-02-14",
    ),
    (
        "Small Business Optimism Index Declines on Inflation Worries",
        "2024-03-12",
    ),
]


# ===========================================================================
# Corpus loaders
# ===========================================================================

def load_fomc_statements() -> List[Tuple[str, str, str]]:
    """
    Load the built-in FOMC statements corpus.

    Returns a list of (text, date, sentiment_label) tuples covering
    hawkish, dovish, and neutral monetary policy stances. The corpus
    contains illustrative excerpts from Federal Open Market Committee
    statements spanning multiple policy regimes.

    Returns:
        List of ``(text, date, label)`` tuples where ``label`` is one of
        ``"hawkish"``, ``"dovish"``, or ``"neutral"``.

    Example:
        >>> corpus = load_fomc_statements()
        >>> len(corpus)
        12
        >>> corpus[0][2]
        'hawkish'
    """
    return list(FOMC_STATEMENTS)


def load_economic_news() -> List[Tuple[str, str]]:
    """
    Load the built-in economic news corpus.

    Returns a list of (headline, date) tuples covering a range of
    economic topics including employment, inflation, GDP, trade, and
    financial markets.

    Returns:
        List of ``(headline, date)`` tuples.

    Example:
        >>> news = load_economic_news()
        >>> len(news)
        15
    """
    return list(ECONOMIC_NEWS)
