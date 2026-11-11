"""
test_financial.py
=================
Comprehensive pytest test suite for the Financial Hallucination Detector,
Claim Extractor, Self-Healer, Sentiment Analyzer, and AI Research Agent.

Run:
    python -m pytest test_financial.py -v
"""

from __future__ import annotations
import pytest

from financial import (
    ClaimStatus,
    FinancialRiskLevel,
    FinancialDetectorConfig,
    FinancialClaimExtractor,
    FinancialHallucinationDetector,
    FinancialHealer,
    FinSentimentAnalyzer,
    FinancialResearchAgent,
    FinancialReportStore,
    MarketDataProvider,
    AuditedMetricRecord,
)


@pytest.fixture
def claim_extractor():
    return FinancialClaimExtractor()


@pytest.fixture
def financial_detector():
    cfg = FinancialDetectorConfig(
        relative_tolerance=0.02,
        margin_tolerance_pts=0.5,
        enable_yfinance_live=False,
    )
    return FinancialHallucinationDetector(config=cfg)


@pytest.fixture
def financial_healer():
    return FinancialHealer()


@pytest.fixture
def research_agent(financial_detector):
    return FinancialResearchAgent(detector=financial_detector)


# ─────────────────────────────────────────────────────────────
# 1. CLAIM EXTRACTOR TESTS
# ─────────────────────────────────────────────────────────────

class TestFinancialClaimExtractor:

    def test_extract_inr_crore_claim(self, claim_extractor):
        text = "TCS reported ₹65,000 crore in revenue for Q4 FY24."
        claims = claim_extractor.extract_claims(text)
        assert len(claims) == 1
        claim = claims[0]
        assert claim.company == "TCS"
        assert claim.metric == "revenue"
        assert claim.numeric_value == 65000.0
        assert claim.currency == "INR"
        assert claim.scale == "crore"
        assert claim.normalized_value == 65000.0 * 1e7
        assert "Q4" in claim.period

    def test_extract_usd_billion_claim(self, claim_extractor):
        text = "Apple posted quarterly revenue of $94.93 billion."
        claims = claim_extractor.extract_claims(text)
        assert len(claims) == 1
        claim = claims[0]
        assert claim.company == "APPLE"
        assert claim.metric == "revenue"
        assert claim.numeric_value == 94.93
        assert claim.currency == "USD"
        assert claim.scale == "billion"
        assert claim.normalized_value == 94.93 * 1e9

    def test_extract_percentage_margin_claim(self, claim_extractor):
        text = "TCS operating margin expanded to 26.0% during the quarter."
        claims = claim_extractor.extract_claims(text)
        assert len(claims) == 1
        claim = claims[0]
        assert claim.metric == "operating_margin"
        assert claim.numeric_value == 26.0
        assert claim.is_percentage is True

    def test_ignores_quarter_and_fiscal_year_numbers(self, claim_extractor):
        text = "TCS reported Q4 FY24 results."
        claims = claim_extractor.extract_claims(text)
        assert len(claims) == 0  # Q4 and FY24 are periods, not financial figures

    def test_company_detection(self, claim_extractor):
        assert claim_extractor._detect_company("Tata Consultancy Services shares rose") == "TCS"
        assert claim_extractor._detect_company("Apple Inc quarterly filing") == "APPLE"
        assert claim_extractor._detect_company("Reliance Industries reported profits") == "RELIANCE"
        assert claim_extractor._detect_company("Infosys guidance update") == "INFOSYS"


# ─────────────────────────────────────────────────────────────
# 2. FINANCIAL HALLUCINATION DETECTOR TESTS
# ─────────────────────────────────────────────────────────────

class TestFinancialHallucinationDetector:

    def test_scenario_1_tcs_revenue_contradiction(self, financial_detector):
        """TCS Q4 revenue: 65,000 Cr stated vs 61,237 Cr actual -> Contradicted ~93%"""
        question = "What was TCS's revenue in Q4?"
        answer = "TCS reported ₹65,000 crore in revenue for Q4."
        report = financial_detector.inspect(question, answer, company="TCS")

        assert report.is_hallucinating is True
        assert report.risk_level == FinancialRiskLevel.CRITICAL
        assert report.contradicted_count == 1
        assert report.hallucination_probability >= 0.90
        claim_audit = report.verified_claims[0]
        assert claim_audit.status == ClaimStatus.CONTRADICTED
        assert "61,237" in claim_audit.evidence_value_str
        assert claim_audit.discrepancy_pct > 5.0

    def test_supported_claim_within_tolerance(self, financial_detector):
        """₹61,200 Cr stated vs ₹61,237 Cr actual (within 2% rounding) -> Supported"""
        question = "What was TCS's revenue in Q4?"
        answer = "TCS reported ₹61,200 crore in revenue for Q4."
        report = financial_detector.inspect(question, answer, company="TCS")

        assert report.is_hallucinating is False
        assert report.risk_level == FinancialRiskLevel.LOW
        assert report.supported_count == 1
        assert report.contradicted_count == 0
        assert report.hallucination_probability < 0.10

    def test_exact_supported_claim(self, financial_detector):
        """Exact ₹61,237 crore -> Supported with high quality score"""
        question = "What was TCS's revenue in Q4?"
        answer = "TCS posted Q4 revenue from operations of ₹61,237 crore."
        report = financial_detector.inspect(question, answer, company="TCS")

        assert report.supported_count == 1
        assert report.verified_claims[0].status == ClaimStatus.SUPPORTED
        assert report.quality_score >= 0.95

    def test_margin_contradiction(self, financial_detector):
        """Operating margin stated 29.5% vs audited 26.0% -> Contradicted"""
        question = "What was TCS's operating margin?"
        answer = "TCS achieved an operating margin of 29.5% in Q4."
        report = financial_detector.inspect(question, answer, company="TCS")

        assert report.contradicted_count == 1
        assert report.verified_claims[0].status == ClaimStatus.CONTRADICTED
        assert report.verified_claims[0].evidence_value_str == "26.0%"

    def test_apple_global_usd_contradiction(self, financial_detector):
        """Apple revenue stated $105.5 billion vs audited $94.93 billion -> Contradicted"""
        question = "What was Apple's Q4 revenue?"
        answer = "Apple delivered quarterly revenue of $105.5 billion for Q4."
        report = financial_detector.inspect(question, answer, company="APPLE")

        assert report.contradicted_count == 1
        assert report.verified_claims[0].status == ClaimStatus.CONTRADICTED
        assert "$94.93 billion" in report.verified_claims[0].evidence_value_str

    def test_reliance_revenue_grounded(self, financial_detector):
        """Reliance revenue ₹240,715 crore -> Supported"""
        question = "What was Reliance's Q4 revenue?"
        answer = "Reliance reported gross revenue of ₹240,715 crore for Q4."
        report = financial_detector.inspect(question, answer, company="RELIANCE")

        assert report.supported_count == 1
        assert report.verified_claims[0].status == ClaimStatus.SUPPORTED

    def test_unknown_metric_flagged(self, financial_detector):
        """An unsupported claim not found in filings -> Unknown"""
        question = "What was TCS's R&D expenditure?"
        answer = "TCS spent ₹18,500 crore on artificial intelligence research."
        report = financial_detector.inspect(question, answer, company="TCS")

        assert report.unknown_count >= 1
        assert report.verified_claims[0].status == ClaimStatus.UNKNOWN


# ─────────────────────────────────────────────────────────────
# 3. FINANCIAL SELF-HEALER TESTS
# ─────────────────────────────────────────────────────────────

class TestFinancialHealer:

    def test_healer_patches_contradicted_number(self, financial_detector, financial_healer):
        question = "What was TCS's revenue in Q4?"
        answer = "TCS reported ₹65,000 crore in revenue for Q4 FY24."
        report = financial_detector.inspect(question, answer, company="TCS")

        healing = financial_healer.heal(report)
        assert healing.healing_applied is True
        assert "₹61,237 crore" in healing.healed_answer
        assert "₹65,000 crore" not in healing.healed_answer
        assert len(healing.patches_made) == 1
        assert healing.final_risk == FinancialRiskLevel.LOW
        assert "Verified Sources & Citations" in healing.healed_answer

    def test_clean_answer_passes_without_healing(self, financial_detector, financial_healer):
        question = "What was TCS's revenue in Q4?"
        answer = "TCS reported ₹61,237 crore in revenue for Q4."
        report = financial_detector.inspect(question, answer, company="TCS")

        healing = financial_healer.heal(report)
        assert healing.healing_applied is False
        assert healing.healed_answer == answer

    def test_multi_claim_healing(self, financial_detector, financial_healer):
        question = "What were TCS's Q4 margin and dividend?"
        answer = "TCS reported an operating margin of 29.5% and a final dividend of ₹35 per share."
        report = financial_detector.inspect(question, answer, company="TCS")

        healing = financial_healer.heal(report)
        assert healing.healing_applied is True
        assert len(healing.patches_made) == 2
        assert "26.0%" in healing.healed_answer
        assert "₹28 per share" in healing.healed_answer


# ─────────────────────────────────────────────────────────────
# 4. FINANCIAL SENTIMENT ANALYZER TESTS
# ─────────────────────────────────────────────────────────────

class TestFinSentimentAnalyzer:

    def test_positive_earnings_sentiment(self):
        analyzer = FinSentimentAnalyzer()
        text = "TCS achieved record deal wins, robust operating leverage, and healthy dividend growth."
        result = analyzer.analyze(text)
        assert result.label == "POSITIVE"
        assert result.score > 0.0
        assert "growth" in result.positive_terms or "robust" in result.positive_terms

    def test_negative_headwinds_sentiment(self):
        analyzer = FinSentimentAnalyzer()
        text = "The company faces severe macroeconomic headwinds, margin pressure, and customer churn."
        result = analyzer.analyze(text)
        assert result.label == "NEGATIVE"
        assert result.score < 0.0
        assert "headwinds" in result.negative_terms


# ─────────────────────────────────────────────────────────────
# 5. AI FINANCIAL RESEARCH AGENT TESTS
# ─────────────────────────────────────────────────────────────

class TestFinancialResearchAgent:

    def test_analyze_tcs_returns_structured_report(self, research_agent):
        report = research_agent.analyze("Analyze TCS before I invest")

        assert report.company == "TCS"
        assert 0 <= report.financial_health_score <= 100
        assert 0 <= report.growth_score <= 100
        assert 0 <= report.profitability_score <= 100
        assert 0 <= report.risk_score <= 100
        assert report.overall_assessment in ("BULLISH", "MODERATE POSITIVE", "NEUTRAL", "CAUTIOUS")
        assert report.confidence_pct > 50
        assert len(report.key_findings) > 0
        assert len(report.recent_events) > 0
        assert len(report.evidence_citations) > 0

    def test_research_agent_self_verification_passes(self, research_agent):
        """Ensure the research agent's self-audit verifies all claims with 0 hallucinations."""
        report = research_agent.analyze("Analyze TCS before I invest")
        assert report.is_fully_grounded is True
        assert report.verification_audit is not None
        assert report.verification_audit.contradicted_count == 0
        assert report.verification_audit.hallucination_probability < 0.05
        assert report.forensic_report is not None
        assert report.forensic_report.piotroski_f_score >= 7


# ─────────────────────────────────────────────────────────────
# 6. FORENSIC HEALTH ANALYZER TESTS
# ─────────────────────────────────────────────────────────────

class TestForensicAnalyzer:

    def test_tcs_forensic_metrics(self):
        from financial import ForensicAnalyzer
        analyzer = ForensicAnalyzer()
        res = analyzer.analyze("TCS")
        assert res.piotroski_f_score >= 8
        assert res.altman_z_score > 2.6
        assert "SAFE ZONE" in res.z_score_zone
        assert res.cash_conversion_ratio > 1.0

    def test_apple_forensic_metrics(self):
        from financial import ForensicAnalyzer
        analyzer = ForensicAnalyzer()
        res = analyzer.analyze("APPLE")
        assert res.piotroski_f_score >= 7
        assert res.altman_z_score > 2.6


# ─────────────────────────────────────────────────────────────
# 7. PEER BENCHMARK & COMPARATIVE CLAIM TESTS
# ─────────────────────────────────────────────────────────────

class TestPeerBenchmarkEngine:

    def test_peer_basket_retrieval(self):
        from financial import PeerBenchmarkEngine
        engine = PeerBenchmarkEngine()
        peers = engine.get_peer_comparison("TCS")
        peer_names = [p.company for p in peers]
        assert "TCS" in peer_names
        assert "INFOSYS" in peer_names

    def test_comparative_contradiction(self):
        from financial import PeerBenchmarkEngine, ClaimStatus
        engine = PeerBenchmarkEngine()
        # TCS margin is 26%, Infosys is 20.1% -> claiming TCS is lower is CONTRADICTED
        res = engine.audit_comparative_claim("TCS has a lower operating margin than Infosys.")
        assert res is not None
        assert res.status == ClaimStatus.CONTRADICTED
        assert "Contradiction" in res.explanation

    def test_comparative_supported(self):
        from financial import PeerBenchmarkEngine, ClaimStatus
        engine = PeerBenchmarkEngine()
        res = engine.audit_comparative_claim("TCS has a higher operating margin than Infosys.")
        assert res is not None
        assert res.status == ClaimStatus.SUPPORTED


# ─────────────────────────────────────────────────────────────
# 8. TEMPORAL & BASIS GROWTH VERIFIER TESTS
# ─────────────────────────────────────────────────────────────

class TestTemporalGrowthVerifier:

    def test_yoy_vs_qoq_basis_flip(self):
        from financial import TemporalGrowthVerifier
        verifier = TemporalGrowthVerifier()
        res = verifier.verify_temporal_statement(
            "TCS reported that its revenue grew 3.5% QoQ sequentially.",
            company="TCS"
        )
        assert res.detected_issue is True
        assert res.issue_type == "BASIS_FLIP"
        assert "YoY" in res.explanation

    def test_clean_temporal_statement(self):
        from financial import TemporalGrowthVerifier
        verifier = TemporalGrowthVerifier()
        res = verifier.verify_temporal_statement(
            "TCS reported that its revenue grew 3.5% YoY year over year.",
            company="TCS"
        )
        assert res.detected_issue is False
        assert res.issue_type == "CLEAN"


# ─────────────────────────────────────────────────────────────
# 9. DOCUMENT INDEXER TESTS
# ─────────────────────────────────────────────────────────────

class TestFinancialDocumentIndexer:

    def test_index_text_statement(self):
        from financial import FinancialDocumentIndexer, FinancialReportStore
        store = FinancialReportStore()
        indexer = FinancialDocumentIndexer(report_store=store)

        text = (
            "Statement of Operations:\n"
            "Revenue from operations was 45000 cr.\n"
            "Net profit for the period was 9000 cr.\n"
        )
        records = indexer.index_text_document("CUSTOM_CO", "Q4 Results", text, period="Q4")
        assert len(records) >= 2
        rev = store.lookup_metric("CUSTOM_CO", "revenue", "Q4")
        assert rev is not None
        assert rev.reported_value == 45000.0


# ─────────────────────────────────────────────────────────────
# 10. MULTI-COMPANY COVERAGE TESTS
# ─────────────────────────────────────────────────────────────

class TestMultiCompanyCoverage:

    def test_nvidia_hallucination_detected(self, financial_detector):
        # NVIDIA audited Q4 revenue is $22.10B, claim is $28.00B
        report = financial_detector.inspect(
            question="What was Nvidia Q4 revenue?",
            answer="Nvidia reported Q4 revenue of $28.00 billion with strong AI compute sales.",
            company="NVIDIA"
        )
        assert report.is_hallucinating is True
        assert any(vc.status == ClaimStatus.CONTRADICTED for vc in report.verified_claims)

    def test_google_cloud_revenue_supported(self, financial_detector):
        report = financial_detector.inspect(
            question="What was Google revenue?",
            answer="Alphabet announced consolidated revenues of $86.31 billion for the quarter.",
            company="GOOGLE"
        )
        assert any(vc.status == ClaimStatus.SUPPORTED for vc in report.verified_claims)

    def test_research_agent_multi_companies(self, research_agent):
        for comp in ["NVIDIA", "GOOGLE", "TESLA", "HDFC", "TATAMOTORS", "AMAZON", "META", "WIPRO"]:
            rep = research_agent.analyze(f"Analyze {comp} before I invest")
            assert rep.company == comp
            assert rep.company_name != ""
            assert rep.financial_health_score > 0
            assert rep.overall_assessment in ["STRONG BUY / BULLISH", "MODERATE POSITIVE", "NEUTRAL", "CAUTIOUS / UNDERWEIGHT"]
            assert rep.forensic_report is not None
            assert len(rep.market_metrics) > 0


# ─────────────────────────────────────────────────────────────
# 11. FINANCIAL RAG ENGINE TESTS
# ─────────────────────────────────────────────────────────────

class TestFinancialRAGEngine:

    def test_rag_engine_initialization(self):
        from financial import FinancialRAGEngine
        rag = FinancialRAGEngine()
        assert len(rag.chunks) >= 20
        categories = rag.get_all_categories()
        assert "Valuation" in categories
        assert "Financial Health" in categories
        assert "Investment Strategy" in categories

    def test_rag_retrieval_pe_ratio(self):
        from financial import FinancialRAGEngine
        rag = FinancialRAGEngine()
        retrieved = rag.retrieve("What is P/E ratio?", top_k=3)
        assert len(retrieved) > 0
        top_chunk, score = retrieved[0]
        assert "P/E" in top_chunk.title or "Earnings" in top_chunk.title
        assert score > 0.05

    def test_rag_retrieval_piotroski(self):
        from financial import FinancialRAGEngine
        rag = FinancialRAGEngine()
        retrieved = rag.retrieve("Explain Piotroski F-score and its 9 criteria", top_k=3)
        assert len(retrieved) > 0
        titles = [c.title for c, _ in retrieved]
        assert any("Piotroski" in t for t in titles)

    def test_rag_answer_grounding(self):
        from financial import FinancialRAGEngine
        rag = FinancialRAGEngine()
        res = rag.answer_query("How does SIP and rupee cost averaging work?")
        assert res["grounded"] is True
        assert res["confidence"] > 0.7
        assert len(res["sources"]) > 0
        assert "SIP" in res["answer"] or "Averaging" in res["answer"] or "Systematic" in res["answer"]

    def test_rag_custom_knowledge_addition(self):
        from financial import FinancialRAGEngine, KnowledgeChunk
        rag = FinancialRAGEngine()
        initial_count = len(rag.chunks)
        new_chunk = KnowledgeChunk(
            chunk_id="test_pharma_01",
            category="Sector Analysis",
            title="Indian Pharmaceutical Sector Dynamics",
            content="Indian pharma companies supply over 20% of global generic drugs with USFDA compliance.",
            keywords=["pharma", "generic", "usfda", "drugs"]
        )
        rag.add_knowledge(new_chunk)
        assert len(rag.chunks) == initial_count + 1
        res = rag.retrieve("Tell me about Indian pharma and generics", top_k=3)
        assert any("Pharmaceutical" in c.title for c, _ in res)



