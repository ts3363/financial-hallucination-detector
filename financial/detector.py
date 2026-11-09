"""
financial/detector.py
=====================
Financial Hallucination Detector.
Extracts financial assertions, verifies them against audited filings and
market data, and computes claim-level verification and aggregate hallucination probability.
"""

from __future__ import annotations
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from .config import (
    ClaimStatus,
    FinancialRiskLevel,
    FinancialDetectorConfig,
)
from .claim_extractor import FinancialClaimExtractor, FinancialClaim
from .data_sources import FinancialReportStore, MarketDataProvider, AuditedMetricRecord
from .temporal_verifier import TemporalGrowthVerifier
from .peer_benchmark import PeerBenchmarkEngine


@dataclass
class VerifiedClaim:
    """The audit outcome for an individual financial claim."""
    claim: FinancialClaim
    status: ClaimStatus
    claimed_value_str: str
    evidence_value_str: Optional[str]
    discrepancy_pct: Optional[float]
    evidence_source: Optional[str]
    evidence_quote: Optional[str]
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.claim.metric_label,
            "period": self.claim.period or "Latest",
            "status": self.status.value,
            "claimed_value": self.claimed_value_str,
            "evidence_value": self.evidence_value_str or "N/A",
            "discrepancy_pct": f"{self.discrepancy_pct:+.2f}%" if self.discrepancy_pct is not None else "N/A",
            "evidence_source": self.evidence_source or "Unverified",
            "evidence_quote": self.evidence_quote or "No matching filing passage found.",
            "explanation": self.explanation,
        }


@dataclass
class FinancialVerificationReport:
    """Complete audit report for a financial LLM response."""
    question: str
    answer: str
    company: Optional[str]
    verified_claims: List[VerifiedClaim] = field(default_factory=list)
    hallucination_probability: float = 0.0
    quality_score: float = 1.0
    risk_level: FinancialRiskLevel = FinancialRiskLevel.LOW
    is_hallucinating: bool = False
    supported_count: int = 0
    contradicted_count: int = 0
    unknown_count: int = 0
    latency_ms: float = 0.0

    def summary(self) -> str:
        lines = [
            f"  Entity       : {self.company or 'Auto-detected'}",
            f"  Risk Level   : {self.risk_level.value}",
            f"  Hallucination: {self.hallucination_probability:.1%}",
            f"  Quality Score: {self.quality_score:.2f} / 1.0",
            f"  Audit Counts : {self.supported_count} Supported | {self.contradicted_count} Contradicted | {self.unknown_count} Unknown",
            f"  Latency      : {self.latency_ms:.1f}ms",
            "\n  Claim Verification Breakdown:",
        ]
        if not self.verified_claims:
            lines.append("    (No specific quantitative financial claims detected)")
        for idx, vc in enumerate(self.verified_claims, 1):
            badge = "✅" if vc.status == ClaimStatus.SUPPORTED else ("❌" if vc.status == ClaimStatus.CONTRADICTED else "⚠️")
            lines.append(
                f"    {idx}. {badge} [{vc.status.value}] Claim: {vc.claim.metric_label} ({vc.claim.period or 'Latest'}) = {vc.claimed_value_str}"
            )
            if vc.status == ClaimStatus.CONTRADICTED:
                disc_str = f" ({vc.discrepancy_pct:+.2f}% discrepancy)" if vc.discrepancy_pct is not None else ""
                lines.append(f"       Audited Ground Truth: {vc.evidence_value_str}{disc_str}")
                lines.append(f"       Source              : {vc.evidence_source}")
                lines.append(f"       Filing Excerpt      : \"{vc.evidence_quote}\"")
            elif vc.status == ClaimStatus.SUPPORTED:
                lines.append(f"       Audited Ground Truth: {vc.evidence_value_str}")
                lines.append(f"       Source              : {vc.evidence_source}")
            else:
                lines.append(f"       Note                : {vc.explanation}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "company": self.company,
            "risk_level": self.risk_level.value,
            "hallucination_probability": round(self.hallucination_probability, 3),
            "quality_score": round(self.quality_score, 3),
            "is_hallucinating": self.is_hallucinating,
            "supported_count": self.supported_count,
            "contradicted_count": self.contradicted_count,
            "unknown_count": self.unknown_count,
            "latency_ms": round(self.latency_ms, 2),
            "claims": [c.to_dict() for c in self.verified_claims],
        }


class FinancialHallucinationDetector:
    """
    Production-grade Financial Hallucination Detector.
    
    Validates LLM answers against regulatory filings and market data.
    Catches numeric confabulations, period slips, currency mismatches,
    and ungrounded quantitative assertions.
    """

    def __init__(
        self,
        config: Optional[FinancialDetectorConfig] = None,
        report_store: Optional[FinancialReportStore] = None,
        market_data: Optional[MarketDataProvider] = None,
    ):
        self.config = config or FinancialDetectorConfig()
        self.report_store = report_store or FinancialReportStore()
        self.market_data = market_data or MarketDataProvider(enable_live=self.config.enable_yfinance_live)
        self.claim_extractor = FinancialClaimExtractor()
        self.temporal_verifier = TemporalGrowthVerifier()
        self.peer_engine = PeerBenchmarkEngine()

    def inspect(
        self,
        question: str,
        answer: str,
        company: Optional[str] = None,
        context_chunks: Optional[List[str]] = None,
    ) -> FinancialVerificationReport:
        """
        Analyze and audit financial claims in an LLM answer.
        """
        t0 = time.perf_counter()

        # Step 1: Detect company
        detected_company = company or self.claim_extractor._detect_company(f"{question} {answer}")
        
        # Step 2: Extract structured claims
        claims = self.claim_extractor.extract_claims(answer, default_company=detected_company)

        # Step 3: Verify each claim against filings & market data
        verified_claims: List[VerifiedClaim] = []
        supported_cnt = 0
        contradicted_cnt = 0
        unknown_cnt = 0

        # Retrieve relevant filing chunks for entity
        filing_chunks = (context_chunks or [])
        if detected_company:
            filing_chunks.extend(self.report_store.get_filing_chunks(detected_company))

        for claim in claims:
            target_comp = claim.company or detected_company
            vc = self._verify_claim(claim, target_comp, filing_chunks)
            verified_claims.append(vc)
            if vc.status == ClaimStatus.SUPPORTED:
                supported_cnt += 1
            elif vc.status == ClaimStatus.CONTRADICTED:
                contradicted_cnt += 1
            else:
                unknown_cnt += 1

        # Step 3b: Audit temporal basis and comparative peer claims
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if len(s.strip()) > 10]
        for sent in sentences:
            temp_res = self.temporal_verifier.verify_temporal_statement(sent, company=detected_company or "TCS")
            if temp_res.detected_issue:
                vc_temp = VerifiedClaim(
                    claim=FinancialClaim(
                        raw_text=sent,
                        company=detected_company,
                        metric=temp_res.issue_type.lower(),
                        metric_label=temp_res.issue_type.replace('_', ' ').title(),
                        period="Q4",
                        numeric_value=0.0,
                        raw_number_str=temp_res.issue_type,
                        currency=None,
                        scale=None,
                        normalized_value=0.0
                    ),
                    status=ClaimStatus.CONTRADICTED,
                    claimed_value_str=temp_res.issue_type,
                    evidence_value_str=temp_res.suggested_correction or "Audited Filing",
                    discrepancy_pct=100.0,
                    evidence_source="Regulatory Filings & Statements",
                    evidence_quote=temp_res.explanation,
                    explanation=temp_res.explanation
                )
                verified_claims.append(vc_temp)
                contradicted_cnt += 1

            peer_res = self.peer_engine.audit_comparative_claim(sent)
            if peer_res:
                vc_peer = VerifiedClaim(
                    claim=FinancialClaim(
                        raw_text=sent,
                        company=peer_res.company_a,
                        metric=f"peer_vs_{peer_res.company_b.lower()}",
                        metric_label=f"{peer_res.metric.replace('_', ' ').title()} vs {peer_res.company_b}",
                        period="Latest",
                        numeric_value=peer_res.val_a,
                        raw_number_str=f"{peer_res.val_a} vs {peer_res.val_b}",
                        currency=None,
                        scale="%",
                        normalized_value=peer_res.val_a
                    ),
                    status=peer_res.status,
                    claimed_value_str=f"{peer_res.relation} than {peer_res.company_b}",
                    evidence_value_str=f"{peer_res.company_a}: {peer_res.val_a} | {peer_res.company_b}: {peer_res.val_b}",
                    discrepancy_pct=None,
                    evidence_source=f"Industry Peer Benchmark ({peer_res.company_a} & {peer_res.company_b})",
                    evidence_quote=peer_res.explanation,
                    explanation=peer_res.explanation
                )
                verified_claims.append(vc_peer)
                if peer_res.status == ClaimStatus.SUPPORTED:
                    supported_cnt += 1
                elif peer_res.status == ClaimStatus.CONTRADICTED:
                    contradicted_cnt += 1

        # Step 4: Calculate Hallucination Probability and Risk Tier
        total_claims = len(verified_claims)
        if total_claims == 0:
            hallucination_prob = 0.05
            quality_score = 0.95
            risk = FinancialRiskLevel.LOW
        elif contradicted_cnt > 0:
            # When contradicted, high probability (e.g. 94% for TCS 65k vs 61k)
            max_discrepancy = max((abs(c.discrepancy_pct or 0.0) for c in verified_claims), default=5.0)
            base_prob = self.config.contradiction_weight
            discrepancy_boost = min(0.12, (max_discrepancy / 100.0) * 0.5 + 0.05)
            hallucination_prob = min(0.99, base_prob + discrepancy_boost + (0.04 * (contradicted_cnt - 1)))
            quality_score = max(0.05, 1.0 - hallucination_prob)
            risk = FinancialRiskLevel.CRITICAL if hallucination_prob >= 0.85 else FinancialRiskLevel.HIGH
        elif unknown_cnt > 0:
            unknown_ratio = unknown_cnt / total_claims
            hallucination_prob = min(0.70, self.config.unknown_weight * unknown_ratio + 0.10)
            quality_score = 1.0 - hallucination_prob
            risk = FinancialRiskLevel.MEDIUM if unknown_ratio < 0.5 else FinancialRiskLevel.HIGH
        else:
            # All claims supported
            hallucination_prob = 0.02
            quality_score = 0.98
            risk = FinancialRiskLevel.LOW

        is_hallucinating = risk in (FinancialRiskLevel.HIGH, FinancialRiskLevel.CRITICAL)
        latency = (time.perf_counter() - t0) * 1000.0

        return FinancialVerificationReport(
            question=question,
            answer=answer,
            company=detected_company,
            verified_claims=verified_claims,
            hallucination_probability=hallucination_prob,
            quality_score=quality_score,
            risk_level=risk,
            is_hallucinating=is_hallucinating,
            supported_count=supported_cnt,
            contradicted_count=contradicted_cnt,
            unknown_count=unknown_cnt,
            latency_ms=latency,
        )

    def _verify_claim(
        self,
        claim: FinancialClaim,
        company: Optional[str],
        filing_chunks: List[str]
    ) -> VerifiedClaim:
        """Verify an individual claim against filings or market data."""
        if not company:
            return VerifiedClaim(
                claim=claim,
                status=ClaimStatus.UNKNOWN,
                claimed_value_str=claim.raw_number_str,
                evidence_value_str=None,
                discrepancy_pct=None,
                evidence_source=None,
                evidence_quote=None,
                explanation="Entity not identified for verification.",
            )

        # 1. Check audited filing records
        audited_record = self.report_store.lookup_metric(company, claim.metric, claim.period)

        if audited_record:
            if claim.is_percentage:
                diff_pts = abs(claim.numeric_value - audited_record.reported_value)
                pct_error = (diff_pts / audited_record.reported_value) * 100 if audited_record.reported_value != 0 else 0
                signed_diff = claim.numeric_value - audited_record.reported_value
                if diff_pts <= self.config.margin_tolerance_pts:
                    return VerifiedClaim(
                        claim=claim,
                        status=ClaimStatus.SUPPORTED,
                        claimed_value_str=claim.raw_number_str,
                        evidence_value_str=audited_record.reported_str,
                        discrepancy_pct=signed_diff,
                        evidence_source=f"{audited_record.source_document}, {audited_record.page_number}",
                        evidence_quote=audited_record.exact_filing_quote,
                        explanation=f"Matches audited margin {audited_record.reported_str} within ±{self.config.margin_tolerance_pts}pt tolerance."
                    )
                else:
                    return VerifiedClaim(
                        claim=claim,
                        status=ClaimStatus.CONTRADICTED,
                        claimed_value_str=claim.raw_number_str,
                        evidence_value_str=audited_record.reported_str,
                        discrepancy_pct=signed_diff,
                        evidence_source=f"{audited_record.source_document}, {audited_record.page_number}",
                        evidence_quote=audited_record.exact_filing_quote,
                        explanation=f"Contradicts audited filing: stated {claim.raw_number_str} vs actual {audited_record.reported_str} ({signed_diff:+.1f}pt discrepancy)."
                    )
            else:
                # Absolute currency value comparison
                claimed_norm = claim.normalized_value
                audited_norm = audited_record.normalized_value
                pct_discrepancy = ((claimed_norm - audited_norm) / audited_norm) * 100.0

                if abs(pct_discrepancy) <= (self.config.relative_tolerance * 100.0):
                    return VerifiedClaim(
                        claim=claim,
                        status=ClaimStatus.SUPPORTED,
                        claimed_value_str=claim.raw_number_str,
                        evidence_value_str=audited_record.reported_str,
                        discrepancy_pct=pct_discrepancy,
                        evidence_source=f"{audited_record.source_document}, {audited_record.page_number}",
                        evidence_quote=audited_record.exact_filing_quote,
                        explanation=f"Supported by regulatory filing: {audited_record.reported_str} (within {self.config.relative_tolerance:.0%} tolerance)."
                    )
                else:
                    return VerifiedClaim(
                        claim=claim,
                        status=ClaimStatus.CONTRADICTED,
                        claimed_value_str=claim.raw_number_str,
                        evidence_value_str=audited_record.reported_str,
                        discrepancy_pct=pct_discrepancy,
                        evidence_source=f"{audited_record.source_document}, {audited_record.page_number}",
                        evidence_quote=audited_record.exact_filing_quote,
                        explanation=f"Direct numerical contradiction: claimed {claim.raw_number_str} vs audited {audited_record.reported_str} ({pct_discrepancy:+.2f}% variance)."
                    )

        # 2. Check market data (e.g. for P/E ratio, market cap, stock price)
        if claim.metric in ("pe_ratio", "market_cap", "price"):
            quote = self.market_data.get_quote(company)
            if quote:
                if claim.metric == "pe_ratio":
                    actual_pe = quote.pe_ratio
                    diff = claim.numeric_value - actual_pe
                    if abs(diff) <= 2.0:
                        return VerifiedClaim(
                            claim=claim,
                            status=ClaimStatus.SUPPORTED,
                            claimed_value_str=claim.raw_number_str,
                            evidence_value_str=f"{actual_pe:.1f}x",
                            discrepancy_pct=diff,
                            evidence_source=f"Market Quote ({quote.ticker})",
                            evidence_quote=f"Trailing P/E ratio is {actual_pe:.1f}x",
                            explanation=f"P/E ratio verified against live market data: {actual_pe:.1f}x."
                        )
                    else:
                        return VerifiedClaim(
                            claim=claim,
                            status=ClaimStatus.CONTRADICTED,
                            claimed_value_str=claim.raw_number_str,
                            evidence_value_str=f"{actual_pe:.1f}x",
                            discrepancy_pct=diff,
                            evidence_source=f"Market Quote ({quote.ticker})",
                            evidence_quote=f"Trailing P/E ratio is {actual_pe:.1f}x",
                            explanation=f"P/E ratio discrepancy: claimed {claim.raw_number_str} vs actual market multiple {actual_pe:.1f}x."
                        )

        # 3. Fallback: Search textual filing chunks
        if filing_chunks:
            claim_metric_words = claim.metric.split('_')
            for chunk in filing_chunks:
                chunk_lower = chunk.lower()
                if any(w in chunk_lower for w in claim_metric_words):
                    # Check if claim's raw number string appears in chunk
                    if str(int(claim.numeric_value)) in chunk or claim.raw_number_str in chunk:
                        return VerifiedClaim(
                            claim=claim,
                            status=ClaimStatus.SUPPORTED,
                            claimed_value_str=claim.raw_number_str,
                            evidence_value_str=claim.raw_number_str,
                            discrepancy_pct=0.0,
                            evidence_source="Retrieved Filing Documents",
                            evidence_quote=chunk,
                            explanation="Verified through textual alignment in retrieved regulatory context.",
                        )

        # 4. If not found
        return VerifiedClaim(
            claim=claim,
            status=ClaimStatus.UNKNOWN,
            claimed_value_str=claim.raw_number_str,
            evidence_value_str=None,
            discrepancy_pct=None,
            evidence_source=None,
            evidence_quote=None,
            explanation=f"Cannot verify '{claim.metric_label}' from available financial statements or market feeds.",
        )
