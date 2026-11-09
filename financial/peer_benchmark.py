"""
financial/peer_benchmark.py
===========================
Peer benchmarking, industry ranking, and comparative claim verification.
Compares companies against industry baskets and audits relative claims
(e.g., "Company A has lower margins than Company B").
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

from .config import ClaimStatus


@dataclass
class PeerMetricProfile:
    company: str
    ticker: str
    sector: str
    operating_margin: float       # %
    net_margin: float             # %
    pe_ratio: float               # x
    roe: float                    # %
    dividend_yield: float         # %
    debt_to_equity: float         # ratio
    market_cap_str: str


@dataclass
class ComparativeClaimResult:
    claim_text: str
    company_a: str
    company_b: str
    metric: str
    relation: str                 # "higher", "lower", "equal"
    val_a: float
    val_b: float
    status: ClaimStatus
    explanation: str


class PeerBenchmarkEngine:
    """
    Industry peer comparison and relative comparative claim auditor.
    """

    PEER_PROFILES: Dict[str, PeerMetricProfile] = {
        "TCS": PeerMetricProfile(
            company="TCS",
            ticker="TCS.NS",
            sector="IT Services",
            operating_margin=26.0,
            net_margin=19.3,
            pe_ratio=29.8,
            roe=51.5,
            dividend_yield=1.95,
            debt_to_equity=0.08,
            market_cap_str="₹13.95 Lakh Cr"
        ),
        "INFOSYS": PeerMetricProfile(
            company="INFOSYS",
            ticker="INFY.NS",
            sector="IT Services",
            operating_margin=20.1,
            net_margin=16.8,
            pe_ratio=28.2,
            roe=31.8,
            dividend_yield=2.30,
            debt_to_equity=0.10,
            market_cap_str="₹7.55 Lakh Cr"
        ),
        "WIPRO": PeerMetricProfile(
            company="WIPRO",
            ticker="WIPRO.NS",
            sector="IT Services",
            operating_margin=16.4,
            net_margin=12.2,
            pe_ratio=23.5,
            roe=14.2,
            dividend_yield=0.85,
            debt_to_equity=0.22,
            market_cap_str="₹2.85 Lakh Cr"
        ),
        "RELIANCE": PeerMetricProfile(
            company="RELIANCE",
            ticker="RELIANCE.NS",
            sector="Conglomerate & Energy",
            operating_margin=17.6,
            net_margin=7.9,
            pe_ratio=25.4,
            roe=9.8,
            dividend_yield=0.78,
            debt_to_equity=0.42,
            market_cap_str="₹17.38 Lakh Cr"
        ),
        "APPLE": PeerMetricProfile(
            company="APPLE",
            ticker="AAPL",
            sector="Consumer Electronics",
            operating_margin=31.2,
            net_margin=24.0,
            pe_ratio=36.4,
            roe=147.2,
            dividend_yield=0.45,
            debt_to_equity=1.52,
            market_cap_str="$3.41 Trillion"
        ),
        "MICROSOFT": PeerMetricProfile(
            company="MICROSOFT",
            ticker="MSFT",
            sector="Global Big Tech",
            operating_margin=44.6,
            net_margin=35.3,
            pe_ratio=34.8,
            roe=38.5,
            dividend_yield=0.72,
            debt_to_equity=0.41,
            market_cap_str="$3.09 Trillion"
        ),
        "GOOGLE": PeerMetricProfile(
            company="GOOGLE",
            ticker="GOOGL",
            sector="Global Big Tech",
            operating_margin=27.5,
            net_margin=24.0,
            pe_ratio=24.8,
            roe=31.4,
            dividend_yield=0.45,
            debt_to_equity=0.11,
            market_cap_str="$2.21 Trillion"
        ),
        "AMAZON": PeerMetricProfile(
            company="AMAZON",
            ticker="AMZN",
            sector="Global Big Tech",
            operating_margin=7.8,
            net_margin=6.2,
            pe_ratio=42.1,
            roe=22.3,
            dividend_yield=0.00,
            debt_to_equity=0.58,
            market_cap_str="$2.07 Trillion"
        ),
        "META": PeerMetricProfile(
            company="META",
            ticker="META",
            sector="Global Big Tech",
            operating_margin=41.0,
            net_margin=35.0,
            pe_ratio=28.6,
            roe=36.8,
            dividend_yield=0.34,
            debt_to_equity=0.24,
            market_cap_str="$1.48 Trillion"
        ),
        "NVIDIA": PeerMetricProfile(
            company="NVIDIA",
            ticker="NVDA",
            sector="Semiconductors & AI",
            operating_margin=61.5,
            net_margin=55.6,
            pe_ratio=54.2,
            roe=115.8,
            dividend_yield=0.03,
            debt_to_equity=0.22,
            market_cap_str="$3.15 Trillion"
        ),
        "TESLA": PeerMetricProfile(
            company="TESLA",
            ticker="TSLA",
            sector="Automotive & Mobility",
            operating_margin=8.2,
            net_margin=6.8,
            pe_ratio=62.4,
            roe=19.8,
            dividend_yield=0.00,
            debt_to_equity=0.15,
            market_cap_str="$698.00 Billion"
        ),
        "TATAMOTORS": PeerMetricProfile(
            company="TATAMOTORS",
            ticker="TATAMOTORS.NS",
            sector="Automotive & Mobility",
            operating_margin=14.2,
            net_margin=8.6,
            pe_ratio=12.8,
            roe=34.6,
            dividend_yield=0.65,
            debt_to_equity=0.85,
            market_cap_str="₹3.25 Lakh Cr"
        ),
        "HDFC": PeerMetricProfile(
            company="HDFC",
            ticker="HDFCBANK.NS",
            sector="Banking & Financial Services",
            operating_margin=42.0,
            net_margin=24.5,
            pe_ratio=18.5,
            roe=16.8,
            dividend_yield=1.15,
            debt_to_equity=1.10,
            market_cap_str="₹12.75 Lakh Cr"
        ),
    }

    SECTORS: Dict[str, List[str]] = {
        "IT Services": ["TCS", "INFOSYS", "WIPRO"],
        "Global Big Tech": ["APPLE", "MICROSOFT", "GOOGLE", "AMAZON", "META"],
        "Semiconductors & AI": ["NVIDIA"],
        "Automotive & Mobility": ["TESLA", "TATAMOTORS"],
        "Banking & Financial Services": ["HDFC"],
        "Conglomerate & Energy": ["RELIANCE"],
    }

    def get_peer_comparison(self, company: str) -> List[PeerMetricProfile]:
        """Return list of peer profiles in the same sector."""
        comp_upper = company.upper()
        target = self.PEER_PROFILES.get(comp_upper)
        if not target:
            # Fallback peer basket
            return [self.PEER_PROFILES["TCS"], self.PEER_PROFILES["INFOSYS"], self.PEER_PROFILES["MICROSOFT"]]

        sector_peers = self.SECTORS.get(target.sector, [comp_upper])
        return [self.PEER_PROFILES[p] for p in sector_peers if p in self.PEER_PROFILES]

    def audit_comparative_claim(self, claim_sentence: str) -> Optional[ComparativeClaimResult]:
        """
        Check comparative assertions between two companies.
        e.g., 'TCS has a lower operating margin than Infosys' -> Contradicted!
        """
        text_lower = claim_sentence.lower()

        # Identify two companies in the claim
        found_companies = []
        for comp in self.PEER_PROFILES.keys():
            if re.search(rf'\b{re.escape(comp.lower())}\b', text_lower):
                found_companies.append(comp)

        if len(found_companies) < 2:
            return None

        comp_a, comp_b = found_companies[0], found_companies[1]
        prof_a = self.PEER_PROFILES[comp_a]
        prof_b = self.PEER_PROFILES[comp_b]

        # Identify metric
        metric_key = None
        if "operating margin" in text_lower or "margin" in text_lower:
            metric_key = "operating_margin"
            val_a = prof_a.operating_margin
            val_b = prof_b.operating_margin
        elif "roe" in text_lower or "return on equity" in text_lower:
            metric_key = "roe"
            val_a = prof_a.roe
            val_b = prof_b.roe
        elif "p/e" in text_lower or "pe ratio" in text_lower or "valuation" in text_lower:
            metric_key = "pe_ratio"
            val_a = prof_a.pe_ratio
            val_b = prof_b.pe_ratio
        elif "dividend" in text_lower:
            metric_key = "dividend_yield"
            val_a = prof_a.dividend_yield
            val_b = prof_b.dividend_yield
        else:
            return None

        # Identify claimed direction
        claimed_relation = None
        if re.search(r'\b(lower|below|worse|inferior|less|smaller|trailing)\b', text_lower):
            claimed_relation = "lower"
        elif re.search(r'\b(higher|above|better|superior|more|greater|exceeds|surpasses|outperformed)\b', text_lower):
            claimed_relation = "higher"
        else:
            return None

        # Verify against ground truth
        actual_is_higher = val_a > val_b
        actual_is_lower = val_a < val_b

        if (claimed_relation == "higher" and actual_is_higher) or (claimed_relation == "lower" and actual_is_lower):
            status = ClaimStatus.SUPPORTED
            exp = (
                f"Accurate peer comparison: {comp_a} ({val_a}) is indeed {claimed_relation} "
                f"than {comp_b} ({val_b}) for {metric_key.replace('_', ' ')}."
            )
        else:
            status = ClaimStatus.CONTRADICTED
            opp = "higher" if actual_is_higher else "lower"
            exp = (
                f"Comparative Contradiction: Stated that {comp_a} has {claimed_relation} {metric_key.replace('_', ' ')} "
                f"than {comp_b}, but audited figures show {comp_a} ({val_a}) is actually {opp} than {comp_b} ({val_b})."
            )

        return ComparativeClaimResult(
            claim_text=claim_sentence,
            company_a=comp_a,
            company_b=comp_b,
            metric=metric_key,
            relation=claimed_relation,
            val_a=val_a,
            val_b=val_b,
            status=status,
            explanation=exp
        )
