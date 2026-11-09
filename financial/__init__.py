"""
financial/__init__.py
=====================
Financial Hallucination Detector and AI Financial Research Agent.
"""

from .config import (
    ClaimStatus,
    FinancialRiskLevel,
    FinancialDetectorConfig,
    ResearchPillar,
)
from .claim_extractor import (
    FinancialClaim,
    FinancialClaimExtractor,
)
from .sample_data import (
    AuditedMetricRecord,
    AUDITED_FINANCIAL_RECORDS,
    SAMPLE_FILING_TEXT_CHUNKS,
)
from .data_sources import (
    FinancialReportStore,
    MarketDataProvider,
    MarketDataQuote,
)
from .detector import (
    VerifiedClaim,
    FinancialVerificationReport,
    FinancialHallucinationDetector,
)
from .healer import (
    FinancialHealingResult,
    FinancialHealer,
)
from .sentiment import (
    FinSentimentAnalyzer,
    SentimentResult,
)
from .research_agent import (
    FinancialResearchReport,
    FinancialResearchAgent,
)
from .forensics import (
    ForensicAnalyzer,
    ForensicHealthReport,
)
from .peer_benchmark import (
    PeerBenchmarkEngine,
    PeerMetricProfile,
    ComparativeClaimResult,
)
from .rag_indexer import (
    FinancialDocumentIndexer,
    ExtractedFilingMetric,
)
from .temporal_verifier import (
    TemporalGrowthVerifier,
    TemporalVerificationResult,
)

__all__ = [
    "ClaimStatus",
    "FinancialRiskLevel",
    "FinancialDetectorConfig",
    "ResearchPillar",
    "FinancialClaim",
    "FinancialClaimExtractor",
    "AuditedMetricRecord",
    "AUDITED_FINANCIAL_RECORDS",
    "SAMPLE_FILING_TEXT_CHUNKS",
    "FinancialReportStore",
    "MarketDataProvider",
    "MarketDataQuote",
    "VerifiedClaim",
    "FinancialVerificationReport",
    "FinancialHallucinationDetector",
    "FinancialHealingResult",
    "FinancialHealer",
    "FinSentimentAnalyzer",
    "SentimentResult",
    "FinancialResearchReport",
    "FinancialResearchAgent",
    "ForensicAnalyzer",
    "ForensicHealthReport",
    "PeerBenchmarkEngine",
    "PeerMetricProfile",
    "ComparativeClaimResult",
    "FinancialDocumentIndexer",
    "ExtractedFilingMetric",
    "TemporalGrowthVerifier",
    "TemporalVerificationResult",
]
