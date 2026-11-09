"""
financial/config.py
===================
Configuration, enums, metric dictionaries, currency mappings,
and tolerance parameters for the Financial Hallucination Detector
and AI Financial Research Agent.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Optional


class ClaimStatus(str, Enum):
    """Verification outcome of a financial claim."""
    SUPPORTED = "SUPPORTED"          # Claim matches audited source within tolerance
    CONTRADICTED = "CONTRADICTED"    # Claim directly conflicts with audited source (> tolerance)
    UNKNOWN = "UNKNOWN"              # Claim cannot be verified from available reports/data


class FinancialRiskLevel(str, Enum):
    """Aggregate financial risk tier."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ResearchPillar(str, Enum):
    """Core pillars of financial research agent evaluation."""
    FINANCIAL_HEALTH = "FINANCIAL_HEALTH"
    GROWTH = "GROWTH"
    PROFITABILITY = "PROFITABILITY"
    RISK = "RISK"


@dataclass
class FinancialDetectorConfig:
    """
    Configuration parameters for financial claim verification.
    
    relative_tolerance (float):
        Acceptable relative percentage variance before flagging a number
        as contradicted (e.g. 0.02 = 2.0% variance allowed for rounding).
        Example: ₹61,237 Cr vs ₹61,200 Cr is 0.06% variance -> SUPPORTED.
        Example: ₹65,000 Cr vs ₹61,237 Cr is 6.14% variance -> CONTRADICTED.
        
    margin_tolerance_pts (float):
        Absolute percentage point variance allowed for margin and ratio metrics.
        Example: 26.0% vs 25.8% -> 0.2 pt diff -> SUPPORTED.
        Example: 26.0% vs 29.5% -> 3.5 pt diff -> CONTRADICTED.
        
    contradiction_weight (float):
        Base hallucination probability contribution for a single contradicted claim.
        
    unknown_weight (float):
        Hallucination probability contribution for an unverifiable claim.
        
    enable_yfinance_live (bool):
        Whether to attempt live market data retrieval if network allows.
    """
    relative_tolerance: float = 0.02           # 2% rounding tolerance
    margin_tolerance_pts: float = 0.5          # 0.5 percentage points for margins/ratios
    contradiction_weight: float = 0.85         # base probability for contradiction
    unknown_weight: float = 0.40               # probability weight for unverified claims
    enable_yfinance_live: bool = False         # default to instant offline cached fundamentals; toggle True for live yfinance
    auto_heal_answers: bool = True


# Currency scales and multiplier maps
CURRENCY_SCALES: Dict[str, float] = {
    # Indian numbering
    "crore": 1e7,
    "crores": 1e7,
    "cr": 1e7,
    "lakh": 1e5,
    "lakhs": 1e5,
    "lac": 1e5,
    "lacs": 1e5,
    # Western numbering
    "trillion": 1e12,
    "trillions": 1e12,
    "t": 1e12,
    "billion": 1e9,
    "billions": 1e9,
    "b": 1e9,
    "bn": 1e9,
    "million": 1e6,
    "millions": 1e6,
    "m": 1e6,
    "mn": 1e6,
    "thousand": 1e3,
    "thousands": 1e3,
    "k": 1e3,
}

# Currency symbol and code mappings
CURRENCY_SYMBOLS: Dict[str, str] = {
    "₹": "INR",
    "rs": "INR",
    "rs.": "INR",
    "inr": "INR",
    "$": "USD",
    "usd": "USD",
    "dollar": "USD",
    "dollars": "USD",
    "€": "EUR",
    "eur": "EUR",
    "euro": "EUR",
    "euros": "EUR",
    "£": "GBP",
    "gbp": "GBP",
    "pound": "GBP",
    "pounds": "GBP",
    "¥": "JPY",
    "jpy": "JPY",
    "yen": "JPY",
}

# Standardized Financial Metric Categories and Lexical Aliases
FINANCIAL_METRICS_MAP: Dict[str, List[str]] = {
    "revenue": [
        "revenue", "revenues", "sales", "turnover", "top-line", "topline",
        "revenue from operations", "total revenue", "net sales", "gross sales"
    ],
    "revenue_growth": [
        "revenue growth", "sales growth", "top-line growth", "revenue grew"
    ],
    "net_profit": [
        "net profit", "net income", "pat", "profit after tax", "bottom-line",
        "bottomline", "net earnings", "profit for the period", "reported profit"
    ],
    "operating_profit": [
        "operating profit", "operating income", "ebit", "operating earnings"
    ],
    "ebitda": [
        "ebitda", "operating ebitda", "operating cash flow"
    ],
    "operating_margin": [
        "operating margin", "ebit margin", "operating profit margin", "ebitda margin"
    ],
    "net_margin": [
        "net margin", "net profit margin", "pat margin"
    ],
    "gross_margin": [
        "gross margin", "gross profit margin"
    ],
    "eps": [
        "eps", "earnings per share", "diluted eps", "basic eps"
    ],
    "pe_ratio": [
        "pe ratio", "p/e", "p/e ratio", "price to earnings", "price-to-earnings"
    ],
    "market_cap": [
        "market cap", "market capitalization", "market value"
    ],
    "order_book": [
        "order book", "tcv", "deal wins", "total contract value", "deal pipeline"
    ],
    "dividend": [
        "dividend", "dividend per share", "dps", "final dividend", "interim dividend"
    ],
    "debt": [
        "total debt", "net debt", "borrowings", "debt-to-equity", "d/e ratio"
    ],
    "cash_reserves": [
        "cash reserves", "cash and cash equivalents", "free cash flow", "cash balance"
    ],
    "capex_rd": [
        "r&d", "research and development", "capex", "capital expenditure", "research"
    ],
    "expenditure": [
        "spent", "expenditure", "expenses", "spending", "operating cost", "costs", "invested", "investment"
    ],
}
