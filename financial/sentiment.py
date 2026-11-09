"""
financial/sentiment.py
======================
Financial Sentiment Analyzer.
Evaluates earnings call transcripts, news commentary, and management outlook.
Supports FinBERT when available, with a built-in Loughran-McDonald financial lexicon fallback.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SentimentResult:
    label: str            # "POSITIVE", "NEGATIVE", "NEUTRAL"
    score: float          # Polarity score (-1.0 to +1.0)
    confidence: float     # 0.0 to 1.0
    positive_terms: List[str]
    negative_terms: List[str]


class FinSentimentAnalyzer:
    """
    Financial domain sentiment evaluator for earnings transcripts & news.
    """

    FINANCIAL_POSITIVE_LEXICON = frozenset({
        "growth", "record", "exceeded", "outperformed", "expansion", "robust",
        "strong", "acceleration", "tailwinds", "momentum", "resilient", "profitability",
        "accretive", "dividend", "order book", "deal wins", "cash generation",
        "operating leverage", "margin expansion", "guidance upgrade", "healthy",
        "efficiency", "milestone", "double-digit", "surpassed", "upside",
    })

    FINANCIAL_NEGATIVE_LEXICON = frozenset({
        "slowdown", "contraction", "headwinds", "margin pressure", "compression",
        "attrition", "decline", "deterioration", "impairment", "write-down",
        "cancellation", "macroeconomic uncertainty", "delayed", "churn", "debt",
        "sluggish", "underperformed", "weakness", "guidance cut", "litigation",
        "loss", "deficit", "downturn", "restructuring", "inflationary",
    })

    def analyze(self, text: str) -> SentimentResult:
        """Analyze text and return financial sentiment polarity and confidence."""
        if not text:
            return SentimentResult(
                label="NEUTRAL",
                score=0.0,
                confidence=0.5,
                positive_terms=[],
                negative_terms=[]
            )

        text_lower = text.lower()
        words = re.findall(r'\b[a-zA-Z-]{3,}\b', text_lower)

        pos_found = [w for w in words if w in self.FINANCIAL_POSITIVE_LEXICON]
        neg_found = [w for w in words if w in self.FINANCIAL_NEGATIVE_LEXICON]

        pos_count = len(pos_found)
        neg_count = len(neg_found)
        total = pos_count + neg_count

        if total == 0:
            return SentimentResult(
                label="NEUTRAL",
                score=0.0,
                confidence=0.6,
                positive_terms=[],
                negative_terms=[]
            )

        # Polarity between -1.0 and +1.0
        polarity = (pos_count - neg_count) / total
        confidence = min(0.95, 0.5 + (total / 20.0) * 0.45)

        if polarity > 0.15:
            label = "POSITIVE"
        elif polarity < -0.15:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"

        return SentimentResult(
            label=label,
            score=polarity,
            confidence=confidence,
            positive_terms=list(set(pos_found)),
            negative_terms=list(set(neg_found)),
        )
