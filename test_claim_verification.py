"""Tests for local financial claim extraction, retrieval, and verification."""

from __future__ import annotations

import json
import os

import pytest

from financial import (
    DebertaNLIClassifier,
    FinancialClaimVerifier,
    FinancialEvidenceRetriever,
    FinancialReportStore,
)
from financial.claim_verification import FinancialClaimExtractor


class FixedNLI:
    model_name = "test-nli"

    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict(self, premise, hypothesis):
        if "increased" in premise.lower() and "decreased" in hypothesis.lower():
            return {"entailment": 0.01, "contradiction": 0.97, "neutral": 0.02}
        if "100 million" in premise.lower() and "150 million" in hypothesis.lower():
            return {"entailment": 0.01, "contradiction": 0.97, "neutral": 0.02}
        return self.probabilities


class TestFinancialClaimExtractor:
    def test_splits_compound_claims_and_preserves_claim_text(self):
        text = "TCS revenue was ₹65,000 crore and net profit was ₹12,434 crore."
        claims = FinancialClaimExtractor().extract(text)

        assert len(claims) == 2
        assert claims[0].original_text == "TCS revenue was ₹65,000 crore"
        assert claims[0].metrics == ["revenue"]
        assert claims[1].metrics == ["net_profit"]

    @pytest.mark.parametrize(
        ("text", "currency", "scale", "normalized"),
        [
            ("Revenue was $100 million.", "USD", "million", 100_000_000),
            ("Revenue was €2.5 billion.", "EUR", "billion", 2_500_000_000),
            ("Revenue was £750 thousand.", "GBP", "thousand", 750_000),
            ("Revenue was ₹3 lakh.", "INR", "lakh", 300_000),
            ("Revenue was ₹2 crore.", "INR", "crore", 20_000_000),
            ("Revenue was ₹1 lakh crore.", "INR", "lakh crore", 1_000_000_000_000),
            ("Revenue was Rs. 2 crore.", "INR", "crore", 20_000_000),
        ],
    )
    def test_extracts_supported_currencies_and_scales(self, text, currency, scale, normalized):
        claim = FinancialClaimExtractor().extract(text)[0]
        value = claim.values[0]

        assert value.currency == currency
        assert value.scale == scale
        assert value.normalized_value == normalized

    def test_extracts_negative_values_and_percentages(self):
        extractor = FinancialClaimExtractor()
        negative = extractor.extract("Net income was -$2.5 million.")[0].values[0]
        percentage = extractor.extract("Revenue increased by 15%.")[0].values[0]

        assert negative.value == -2.5
        assert negative.normalized_value == -2_500_000
        assert percentage.is_percentage
        assert percentage.value == 15

    def test_rupee_abbreviation_does_not_split_the_claim(self):
        claim = FinancialClaimExtractor().extract("Revenue was Rs. 2 crore.")[0]
        assert claim.original_text == "Revenue was Rs. 2 crore."
        assert claim.values[0].currency == "INR"
        assert claim.values[0].normalized_value == 20_000_000


class TestLocalFinancialEvidence:
    def test_tfidf_retrieval_returns_attributed_relevant_passage(self):
        retriever = FinancialEvidenceRetriever()
        retriever.add_text("Operating margin was 26% in Q4.", "margins.pdf, page 2")
        retriever.add_text("Revenue was $100 million in Q4.", "revenue.pdf, page 1")

        results = retriever.retrieve("Revenue was $150 million in Q4.", top_k=1)

        assert len(results) == 1
        assert "Revenue was $100 million" in results[0].text
        assert results[0].source == "revenue.pdf, page 1"
        assert results[0].relevance > 0

    def test_financebench_loader_indexes_only_evidence_with_doc_source(self, tmp_path):
        data_file = tmp_path / "financebench.jsonl"
        data_file.write_text(
            json.dumps({
                "question": "What was revenue?",
                "answer": "$100 million",
                "doc_name": "annual-report.pdf",
                "evidence": [{"evidence_text": "Revenue was $100 million."}],
            }) + "\n",
            encoding="utf-8",
        )
        retriever = FinancialEvidenceRetriever()

        assert retriever.load_financebench(data_file) == 1
        result = retriever.retrieve("Revenue was $150 million.", top_k=1)[0]
        assert result.source == "annual-report.pdf"
        assert result.text == "Revenue was $100 million."

    def test_no_matching_passage_is_not_fabricated(self):
        retriever = FinancialEvidenceRetriever(min_relevance=0.1)
        retriever.add_text("The company reported revenue of $100 million.", "filing.pdf")

        assert retriever.retrieve("A different company reported its board meeting.", top_k=1) == []


class TestNLIAndClaimPairing:
    def test_deberta_label_mapping_uses_model_labels_not_position_assumptions(self):
        labels = DebertaNLIClassifier.resolve_label_ids(
            {0: "contradiction", 1: "entailment", 2: "neutral"}
        )
        assert labels == {"contradiction": 0, "entailment": 1, "neutral": 2}

    def test_rejects_unmapped_nli_labels(self):
        with pytest.raises(RuntimeError, match="must expose"):
            DebertaNLIClassifier.resolve_label_ids(
                {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
            )

    def test_tcs_numeric_contradiction_reports_evidence_and_discrepancy(self):
        verifier = FinancialClaimVerifier(
            report_store=FinancialReportStore(),
            nli_classifier=FixedNLI(
                {"entailment": 0.01, "contradiction": 0.97, "neutral": 0.02}
            ),
        )
        report = verifier.verify(
            "What was TCS Q4 revenue?",
            "TCS reported ₹65,000 crore in revenue for Q4.",
            company="TCS",
        )

        result = report.claims[0]
        assert result.status == "Hallucinated"
        assert "61,237" in result.evidence[0].text
        assert result.entailment_probability == 0.01
        assert result.contradiction_probability == 0.97
        assert result.neutral_probability == 0.02
        assert result.numerical_comparisons[0].status == "inconsistent"
        assert result.numerical_comparisons[0].percentage_discrepancy > 5

    def test_directional_contradiction_overrides_neutral_nli(self):
        retriever = FinancialEvidenceRetriever()
        retriever.add_text("Revenue increased by 15%.", "earnings.pdf, page 1")
        verifier = FinancialClaimVerifier(
            retriever=retriever,
            nli_classifier=FixedNLI(
                {"entailment": 0.03, "contradiction": 0.04, "neutral": 0.93}
            ),
        )
        result = verifier.verify(
            "How did revenue change?",
            "Revenue decreased by 15%.",
        ).claims[0]

        assert result.status == "Hallucinated"
        assert "direction conflicts" in result.explanation

    def test_neutral_claim_is_potential_hallucination_not_hallucinated(self):
        retriever = FinancialEvidenceRetriever()
        retriever.add_text("Revenue was $100 million.", "filing.pdf")
        verifier = FinancialClaimVerifier(
            retriever=retriever,
            nli_classifier=FixedNLI(
                {"entailment": 0.02, "contradiction": 0.03, "neutral": 0.95}
            ),
        )
        result = verifier.verify(
            "What was revenue?",
            "Revenue remains stable.",
        ).claims[0]

        assert result.status == "Potential Hallucination"
        assert result.contradiction_probability == 0.03

    def test_missing_evidence_is_insufficient_not_false(self):
        verifier = FinancialClaimVerifier(
            retriever=FinancialEvidenceRetriever(min_relevance=0.2),
            nli_classifier=FixedNLI(
                {"entailment": 0.33, "contradiction": 0.33, "neutral": 0.34}
            ),
        )
        result = verifier.verify(
            "What was revenue?",
            "Revenue was $500 million.",
        ).claims[0]

        assert result.status == "Insufficient Evidence"
        assert result.evidence == []
        assert result.contradiction_probability is None

    def test_company_substitution_is_reported_as_hallucinated(self):
        retriever = FinancialEvidenceRetriever()
        retriever.add_text("Microsoft revenue was $100 million.", "microsoft.pdf")
        verifier = FinancialClaimVerifier(
            retriever=retriever,
            nli_classifier=FixedNLI(
                {"entailment": 0.02, "contradiction": 0.03, "neutral": 0.95}
            ),
        )

        result = verifier.verify(
            "What was Apple's revenue?",
            "Apple revenue was $100 million.",
        ).claims[0]

        assert result.entity_mismatch
        assert result.status == "Hallucinated"

    def test_currency_mismatch_is_not_marked_faithful_without_fx_conversion(self):
        retriever = FinancialEvidenceRetriever()
        retriever.add_text("Revenue was €100 million.", "filing.pdf")
        verifier = FinancialClaimVerifier(
            retriever=retriever,
            nli_classifier=FixedNLI(
                {"entailment": 0.95, "contradiction": 0.02, "neutral": 0.03}
            ),
        )

        result = verifier.verify(
            "What was revenue?",
            "Revenue was $100 million.",
        ).claims[0]

        assert result.status == "Potential Hallucination"
        assert result.numerical_comparisons[0].status == "incomparable"

    def test_entity_extractor_finds_ticker_institution_and_executive(self):
        from financial.claim_verification import FinancialEntityExtractor

        entities = FinancialEntityExtractor().extract(
            "AAPL was discussed with Tim Cook at the Reserve Bank of India."
        )
        observed = {(entity.text, entity.label) for entity in entities}

        assert ("AAPL", "TICKER") in observed
        assert ("Reserve Bank of India", "FINANCIAL_INSTITUTION") in observed
        assert any(text == "Tim Cook" and label == "PERSON" for text, label in observed)

    @pytest.mark.skipif(
        os.environ.get("RUN_FINANCIAL_NLI_TESTS") != "1",
        reason="Set RUN_FINANCIAL_NLI_TESTS=1 to download/run the real DeBERTa checkpoint.",
    )
    def test_real_deberta_nli_checks_entailment_contradiction_and_neutral(self):
        classifier = DebertaNLIClassifier()
        entailment = classifier.predict(
            "Revenue increased by 15%.",
            "Revenue grew by 15%.",
        )
        contradiction = classifier.predict(
            "Revenue increased by 15%.",
            "Revenue decreased by 15%.",
        )
        neutral = classifier.predict(
            "Revenue increased by 15%.",
            "The company is headquartered in London.",
        )

        assert entailment["entailment"] > entailment["contradiction"]
        assert contradiction["contradiction"] > contradiction["entailment"]
        assert neutral["neutral"] > neutral["contradiction"]
        for result in (entailment, contradiction, neutral):
            assert set(result) == {"entailment", "contradiction", "neutral"}
            assert sum(result.values()) == pytest.approx(1.0, abs=1e-5)
