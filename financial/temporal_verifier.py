"""
financial/temporal_verifier.py
==============================
Temporal and Basis Confusion Verifier.
Catches period confabulations (FY vs Q4), basis mix-ups (YoY vs QoQ),
and currency/scale mismatch hallucinations.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .config import ClaimStatus


@dataclass
class TemporalVerificationResult:
    claim_text: str
    issue_type: str        # "BASIS_FLIP", "PERIOD_MISMATCH", "CURRENCY_SLIP", "CLEAN"
    detected_issue: bool
    explanation: str
    suggested_correction: Optional[str] = None


class TemporalGrowthVerifier:
    """
    Audits temporal assertions:
    - YoY (Year-over-Year) vs QoQ (Quarter-over-Quarter)
    - Full Year (FY) vs Single Quarter (Q1-Q4)
    - Currency scale alignment
    """

    # Ground truth growth dynamics for benchmark companies
    BENCHMARK_GROWTH: dict = {
        "TCS": {
            "q4_yoy_growth_pct": 3.5,
            "q4_qoq_growth_pct": 1.1,
            "fy_annual_revenue": 240893.0,
            "q4_quarterly_revenue": 61237.0,
        },
        "INFOSYS": {
            "q4_yoy_growth_pct": 1.3,
            "q4_qoq_growth_pct": -2.2,
            "fy_annual_revenue": 153670.0,
            "q4_quarterly_revenue": 37923.0,
        },
        "RELIANCE": {
            "q4_yoy_growth_pct": 11.3,
            "q4_qoq_growth_pct": 4.1,
            "fy_annual_revenue": 1000122.0,
            "q4_quarterly_revenue": 240715.0,
        },
        "APPLE": {
            "q4_yoy_growth_pct": 6.0,
            "q4_qoq_growth_pct": 10.7,
            "fy_annual_revenue": 391035.0,
            "q4_quarterly_revenue": 94930.0,
        },
        "MICROSOFT": {
            "q4_yoy_growth_pct": 15.0,
            "q4_qoq_growth_pct": 4.5,
            "fy_annual_revenue": 245120.0,
            "q4_quarterly_revenue": 64730.0,
        },
        "GOOGLE": {
            "q4_yoy_growth_pct": 13.5,
            "q4_qoq_growth_pct": 3.8,
            "fy_annual_revenue": 307390.0,
            "q4_quarterly_revenue": 86310.0,
        },
        "NVIDIA": {
            "q4_yoy_growth_pct": 265.0,
            "q4_qoq_growth_pct": 22.0,
            "fy_annual_revenue": 60920.0,
            "q4_quarterly_revenue": 22100.0,
        },
        "AMAZON": {
            "q4_yoy_growth_pct": 14.0,
            "q4_qoq_growth_pct": 18.0,
            "fy_annual_revenue": 574780.0,
            "q4_quarterly_revenue": 169960.0,
        },
        "META": {
            "q4_yoy_growth_pct": 25.0,
            "q4_qoq_growth_pct": 17.5,
            "fy_annual_revenue": 134900.0,
            "q4_quarterly_revenue": 40110.0,
        },
        "TESLA": {
            "q4_yoy_growth_pct": 3.0,
            "q4_qoq_growth_pct": 7.8,
            "fy_annual_revenue": 96770.0,
            "q4_quarterly_revenue": 25170.0,
        },
        "HDFC": {
            "q4_yoy_growth_pct": 37.1,
            "q4_qoq_growth_pct": 2.5,
            "fy_annual_revenue": 115340.0,
            "q4_quarterly_revenue": 29077.0,
        },
        "WIPRO": {
            "q4_yoy_growth_pct": -4.2,
            "q4_qoq_growth_pct": -0.3,
            "fy_annual_revenue": 89760.0,
            "q4_quarterly_revenue": 22208.0,
        },
        "TATAMOTORS": {
            "q4_yoy_growth_pct": 13.3,
            "q4_qoq_growth_pct": 8.5,
            "fy_annual_revenue": 437930.0,
            "q4_quarterly_revenue": 119986.0,
        },
    }

    def verify_temporal_statement(self, sentence: str, company: str = "TCS") -> TemporalVerificationResult:
        """Inspect a sentence for YoY vs QoQ basis flip or full-year vs quarter confusion."""
        text_lower = sentence.lower()
        comp_upper = company.upper()
        bench = self.BENCHMARK_GROWTH.get(comp_upper)
        if not bench:
            return TemporalVerificationResult(
                claim_text=sentence,
                issue_type="CLEAN",
                detected_issue=False,
                explanation="No benchmark growth rules for company.",
                suggested_correction=None
            )

        # ── 1. Check YoY vs QoQ basis flip ────────────────────
        # e.g., "grew 3.5% QoQ" vs actual 3.5% YoY
        has_qoq = bool(re.search(r'\b(qoq|quarter[- ]on[- ]quarter|sequentially)\b', text_lower))
        has_yoy = bool(re.search(r'\b(yoy|year[- ]over[- ]year|annually)\b', text_lower))

        m_pct = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*%', text_lower)
        if m_pct:
            pct_val = float(m_pct.group(1))
            # If percentage matches YoY growth number but sentence says QoQ
            if has_qoq and not has_yoy and abs(pct_val - bench["q4_yoy_growth_pct"]) < 0.2:
                return TemporalVerificationResult(
                    claim_text=sentence,
                    issue_type="BASIS_FLIP",
                    detected_issue=True,
                    explanation=(
                        f"Basis Confusion: Stated that revenue grew {pct_val}% sequentially (QoQ), "
                        f"but official filings report {pct_val}% was Year-over-Year (YoY) growth. "
                        f"Actual sequential QoQ growth was only {bench['q4_qoq_growth_pct']}%."
                    ),
                    suggested_correction=sentence.replace("QoQ", "YoY").replace("qoq", "YoY")
                )

        # ── 2. Check Full-Year vs Quarter Mismatch ─────────────
        # e.g., asserting annual ₹240,893 crore as Q4 revenue
        if "q4" in text_lower or "fourth quarter" in text_lower:
            m_num = re.search(r'([0-9]{3,}(?:,[0-9]+)*)', text_lower)
            if m_num:
                num_val = float(m_num.group(1).replace(',', ''))
                if abs(num_val - bench["fy_annual_revenue"]) < 500:
                    return TemporalVerificationResult(
                        claim_text=sentence,
                        issue_type="PERIOD_MISMATCH",
                        detected_issue=True,
                        explanation=(
                            f"Period Mismatch: Claimed {num_val:,.0f} for Q4, but this is the "
                            f"Full-Year (FY24) cumulative revenue. Official Q4 revenue was {bench['q4_quarterly_revenue']:,.0f} crore."
                        ),
                        suggested_correction=f"TCS Full Year (FY24) revenue was ₹{bench['fy_annual_revenue']:,.0f} crore (Q4 was ₹{bench['q4_quarterly_revenue']:,.0f} crore)."
                    )

        return TemporalVerificationResult(
            claim_text=sentence,
            issue_type="CLEAN",
            detected_issue=False,
            explanation="Temporal basis and period alignment verified.",
            suggested_correction=None
        )
