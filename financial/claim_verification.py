"""Claim-level financial evidence, NLI, entity, and numerical verification."""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .claim_extractor import FinancialClaimExtractor as BaseFinancialClaimExtractor
from .config import CURRENCY_SCALES
from .data_sources import FinancialReportStore
from .evidence_retrieval import EvidencePassage, FinancialEvidenceRetriever

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FinancialEntity:
    text: str
    label: str
    canonical: Optional[str] = None


@dataclass(frozen=True)
class FinancialValue:
    raw: str
    value: float
    metric: Optional[str]
    currency: Optional[str]
    scale: Optional[str]
    normalized_value: float
    is_percentage: bool


@dataclass
class ExtractedFinancialClaim:
    original_text: str
    company: Optional[str]
    period: Optional[str]
    metrics: list[str]
    entities: list[FinancialEntity]
    values: list[FinancialValue]


@dataclass(frozen=True)
class NumericalComparison:
    status: str
    claim_value: str
    evidence_value: str
    absolute_discrepancy: Optional[float] = None
    percentage_discrepancy: Optional[float] = None
    explanation: str = ""


@dataclass
class ClaimVerification:
    claim: ExtractedFinancialClaim
    evidence: list[EvidencePassage]
    evidence_values: list[FinancialValue]
    entailment_probability: Optional[float]
    contradiction_probability: Optional[float]
    neutral_probability: Optional[float]
    evidence_entities: list[FinancialEntity]
    numerical_comparisons: list[NumericalComparison]
    entity_mismatch: bool
    status: str
    explanation: str


@dataclass
class FinancialClaimVerificationReport:
    question: str
    answer: str
    model_name: str
    claims: list[ClaimVerification] = field(default_factory=list)


class FinancialEntityExtractor:
    """spaCy NER supplemented with explicit company, ticker, and institution rules."""

    INSTITUTIONS = (
        "Federal Reserve", "Reserve Bank of India", "RBI", "SEC",
        "Securities and Exchange Commission", "NSE", "BSE", "NYSE",
        "NASDAQ", "European Central Bank", "Bank of England",
        "World Bank", "International Monetary Fund", "IMF",
    )
    TICKER_CANONICAL = {
        **{ticker: company for company, ticker in {
            "TCS": "TCS.NS", "INFOSYS": "INFY.NS", "RELIANCE": "RELIANCE.NS",
            "HDFC": "HDFCBANK.NS", "WIPRO": "WIPRO.NS",
            "TATAMOTORS": "TATAMOTORS.NS", "ICICIBANK": "ICICIBANK.NS",
            "BHARTIARTL": "BHARTIARTL.NS", "APPLE": "AAPL",
            "MICROSOFT": "MSFT", "GOOGLE": "GOOGL", "NVIDIA": "NVDA",
            "AMAZON": "AMZN", "META": "META", "TESLA": "TSLA",
        }.items()},
    }

    def __init__(self) -> None:
        self._nlp = None
        self._spacy_loaded = False
        self._claim_extractor = BaseFinancialClaimExtractor()

    def extract(self, text: str) -> list[FinancialEntity]:
        if not text:
            return []
        found: dict[tuple[str, str], FinancialEntity] = {}
        lowered = text.lower()

        for company, aliases in self._claim_extractor.KNOWN_COMPANIES.items():
            for alias in aliases:
                for match in re.finditer(rf"(?<!\w){re.escape(alias)}(?!\w)", lowered):
                    entity = FinancialEntity(text[match.start():match.end()], "COMPANY", company)
                    found[(entity.text.lower(), entity.label)] = entity
        for ticker, canonical in self.TICKER_CANONICAL.items():
            for match in re.finditer(rf"(?<![A-Za-z0-9]){re.escape(ticker)}(?![A-Za-z0-9])", text, re.I):
                entity = FinancialEntity(match.group(0), "TICKER", canonical)
                found[(entity.text.lower(), entity.label)] = entity

        for institution in self.INSTITUTIONS:
            match = re.search(rf"(?<!\w){re.escape(institution)}(?!\w)", text, re.I)
            if match:
                entity = FinancialEntity(match.group(0), "FINANCIAL_INSTITUTION")
                found[(entity.text.lower(), entity.label)] = entity

        nlp = self._get_nlp()
        if nlp is not None:
            doc = nlp(text)
            for span in doc.ents:
                if span.label_ in {"PERSON", "ORG"}:
                    entity = FinancialEntity(span.text, span.label_)
                    found.setdefault((entity.text.lower(), entity.label), entity)
        return list(found.values())

    def _get_nlp(self):
        if not self._spacy_loaded:
            self._spacy_loaded = True
            try:
                import spacy

                self._nlp = spacy.load("en_core_web_sm")
            except (ImportError, OSError):
                logger.warning(
                    "spaCy's en_core_web_sm model is unavailable; financial entity rules remain active, "
                    "but executive/person NER will be limited. Install it with "
                    "'python -m spacy download en_core_web_sm'."
                )
                self._nlp = None
        return self._nlp


class FinancialClaimExtractor:
    """Split answer text into preserved financial claims with structured facts."""

    NUMBER_PATTERN = re.compile(
        r"(?P<open>\()?\s*(?P<sign>-)?\s*"
        r"(?P<prefix_currency>₹|\$|€|£|USD\b|INR\b|EUR\b|GBP\b|Rs\.?(?=\s|[0-9]))?\s*"
        r"(?P<number>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*"
        r"(?P<scale>lakh\s+crores?|trillions?|billions?|millions?|thousands?|crores?|crore|"
        r"lakh(?:s)?|lac(?:s)?|cr|bn|mn|[kmbt])?\s*"
        r"(?P<suffix_currency>USD\b|INR\b|EUR\b|GBP\b)?\s*"
        r"(?P<percent>%|percent)?\s*(?P<close>\))?",
        re.IGNORECASE,
    )
    CURRENCY_PREFIX = {
        "₹": "INR", "rs": "INR", "rs.": "INR", "$": "USD",
        "€": "EUR", "£": "GBP", "usd": "USD", "inr": "INR",
        "eur": "EUR", "gbp": "GBP",
    }
    METRIC_ALIASES = {
        "revenue": ("revenue", "revenues", "sales", "turnover"),
        "net_profit": ("net profit", "net income", "profit after tax", "pat"),
        "gross_profit": ("gross profit",),
        "operating_profit": ("operating profit", "operating income", "ebit"),
        "ebitda": ("ebitda",),
        "eps": ("eps", "earnings per share"),
        "assets": ("assets", "total assets"),
        "liabilities": ("liabilities", "total liabilities"),
        "cash_flow": ("cash flow", "operating cash flow", "free cash flow"),
        "operating_margin": ("operating margin", "ebit margin"),
        "net_margin": ("net margin", "net profit margin"),
        "gross_margin": ("gross margin",),
        "debt_to_equity": ("debt-to-equity", "debt to equity"),
        "pe_ratio": ("p/e ratio", "pe ratio", "price-to-earnings"),
        "growth": ("growth", "grew", "increased", "decreased", "declined"),
    }

    def __init__(self, entity_extractor: Optional[FinancialEntityExtractor] = None):
        self.entity_extractor = entity_extractor or FinancialEntityExtractor()
        self._base_extractor = BaseFinancialClaimExtractor()

    def extract(self, text: str, default_company: Optional[str] = None) -> list[ExtractedFinancialClaim]:
        claims: list[ExtractedFinancialClaim] = []
        for sentence in re.split(
            r"(?<=[!?])\s+|(?<=\.)\s+(?=[A-Z₹$€£])|\n+",
            text or "",
        ):
            sentence = sentence.strip()
            if not sentence:
                continue
            for clause in self._split_compound(sentence):
                entities = self.entity_extractor.extract(clause)
                company = next(
                    (entity.canonical for entity in entities if entity.label == "COMPANY"),
                    default_company,
                )
                spans = self._metric_spans(clause)
                metrics = list(dict.fromkeys(metric for metric, _, _ in spans))
                period = self._base_extractor._detect_period(clause)
                values = self._extract_values(clause, spans)
                if metrics or values or entities:
                    claims.append(ExtractedFinancialClaim(
                        original_text=clause,
                        company=company,
                        period=period,
                        metrics=metrics,
                        entities=entities,
                        values=values,
                    ))
        return claims

    def _split_compound(self, sentence: str) -> list[str]:
        pieces = re.split(r"\s+(?:and|but|whereas|while|however)\s+|;\s*", sentence, flags=re.I)
        if len(pieces) <= 1:
            return [sentence]
        result: list[str] = []
        for piece in pieces:
            metrics = self._metric_spans(piece)
            has_financial_fact = bool(metrics) and (
                bool(self.NUMBER_PATTERN.search(piece))
                or bool(re.search(r"\b(?:increased|decreased|grew|declined|rose|fell|was|were|reported|held)\b", piece, re.I))
            )
            result.append(piece.strip() if has_financial_fact else sentence)
        if all(piece != sentence for piece in result):
            return result
        return [sentence]

    def _metric_spans(self, text: str) -> list[tuple[str, int, int]]:
        spans: list[tuple[str, int, int]] = []
        for metric, aliases in self.METRIC_ALIASES.items():
            for alias in aliases:
                for match in re.finditer(rf"(?<!\w){re.escape(alias)}(?!\w)", text, re.I):
                    spans.append((metric, match.start(), match.end()))
        if not spans:
            spans.extend(
                (item["metric"], item["start"], item["end"])
                for item in self._base_extractor._detect_metric_spans(text)
            )
        return spans

    def _extract_values(
        self,
        text: str,
        metric_spans: list[tuple[str, int, int]],
    ) -> list[FinancialValue]:
        values: list[FinancialValue] = []
        for match in self.NUMBER_PATTERN.finditer(text):
            raw_number = match.group("number")
            prefix = text[max(0, match.start() - 5):match.start()].lower()
            if re.search(r"(?:q|fy)\s*$", prefix):
                continue
            if re.fullmatch(r"20\d{2}", raw_number) and not (
                match.group("scale") or match.group("prefix_currency") or match.group("percent")
            ):
                continue
            try:
                number = float(raw_number.replace(",", ""))
            except ValueError:
                continue
            negative = bool(match.group("sign")) or (
                bool(match.group("open") and match.group("close"))
            )
            if negative:
                number = -number
            scale = (match.group("scale") or "").lower() or None
            percent = bool(match.group("percent"))
            if scale == "%":
                percent = True
                scale = None
            currency_token = match.group("prefix_currency") or match.group("suffix_currency")
            currency = self.CURRENCY_PREFIX.get(currency_token.lower(), currency_token.upper()) if currency_token else None
            metric = min(
                metric_spans,
                key=lambda span: min(abs(match.start() - span[2]), abs(match.end() - span[1])),
                default=(None, 0, 0),
            )[0]
            if percent and metric == "revenue":
                metric = "growth"
            if not currency and scale in ("crore", "crores", "cr", "lakh", "lakhs", "lac", "lacs"):
                currency = "INR"
            if not currency and scale in ("billion", "billions", "million", "millions"):
                currency = None
            multiplier = CURRENCY_SCALES.get(scale or "", 1.0)
            normalized = number if percent else number * multiplier
            if not math.isfinite(normalized):
                continue
            values.append(FinancialValue(
                raw=match.group(0).strip(),
                value=number,
                metric=metric,
                currency=currency,
                scale=scale,
                normalized_value=normalized,
                is_percentage=percent,
            ))
        return values


class DebertaNLIClassifier:
    """Lazy CPU inference with class labels validated from model configuration."""

    MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
    ALIASES = {
        "entailment": {"entailment", "entails", "entailed"},
        "contradiction": {"contradiction", "contradictory", "contradicts"},
        "neutral": {"neutral", "undetermined"},
    }

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._label_ids: dict[str, int] = {}

    def predict(self, premise: str, hypothesis: str) -> dict[str, float]:
        self._load()
        import torch

        encoded = self._tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.inference_mode():
            logits = self._model(**encoded).logits[0]
            probabilities = torch.softmax(logits, dim=-1).tolist()
        result = {
            label: float(probabilities[index])
            for label, index in self._label_ids.items()
        }
        if any(not math.isfinite(value) for value in result.values()):
            raise RuntimeError("The NLI model returned a non-finite probability.")
        return result

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            os.environ.setdefault("USE_TF", "0")
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install torch, transformers, and sentencepiece to run DeBERTa NLI."
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to("cpu")
        self._model.eval()
        config = self._model.config
        label2id = {str(key): int(value) for key, value in (config.label2id or {}).items()}
        id2label = {int(key): str(value) for key, value in (config.id2label or {}).items()}
        self._label_ids = self.resolve_label_ids(id2label, label2id)

    @classmethod
    def resolve_label_ids(
        cls, id2label: dict[int, str], label2id: Optional[dict[str, int]] = None
    ) -> dict[str, int]:
        """Resolve NLI classes from model metadata, never from assumed class order."""
        discovered: dict[str, int] = {}
        for index, label in id2label.items():
            normalized = str(label).lower().replace("label_", "").replace("-", "_").strip()
            for canonical, aliases in cls.ALIASES.items():
                if normalized in aliases:
                    if canonical in discovered and discovered[canonical] != int(index):
                        raise RuntimeError(f"Ambiguous {canonical} label in NLI model configuration.")
                    discovered[canonical] = int(index)
        for label, index in (label2id or {}).items():
            normalized = str(label).lower().replace("label_", "").replace("-", "_").strip()
            for canonical, aliases in cls.ALIASES.items():
                if normalized in aliases:
                    if canonical in discovered and discovered[canonical] != int(index):
                        raise RuntimeError(f"Conflicting {canonical} label IDs in NLI model configuration.")
                    discovered[canonical] = int(index)
        required = {"entailment", "contradiction", "neutral"}
        if set(discovered) != required:
            raise RuntimeError(
                "NLI checkpoint must expose entailment, contradiction, and neutral labels; "
                f"found id2label={id2label!r}, label2id={label2id!r}."
            )
        return discovered


class FinancialClaimVerifier:
    """Pairs extracted claims with local evidence and applies NLI and rules."""

    def __init__(
        self,
        report_store: Optional[FinancialReportStore] = None,
        retriever: Optional[FinancialEvidenceRetriever] = None,
        nli_classifier: Optional[Any] = None,
        nli_threshold: float = 0.70,
        numerical_tolerance: float = 0.02,
    ):
        self.entity_extractor = FinancialEntityExtractor()
        self.claim_extractor = FinancialClaimExtractor(self.entity_extractor)
        self.report_store = report_store
        self.retriever = retriever or FinancialEvidenceRetriever(report_store=report_store)
        self.nli_classifier = nli_classifier or DebertaNLIClassifier()
        self.nli_threshold = nli_threshold
        self.numerical_tolerance = numerical_tolerance

    def verify(
        self,
        question: str,
        answer: str,
        company: Optional[str] = None,
        extra_evidence: Optional[list[EvidencePassage]] = None,
    ) -> FinancialClaimVerificationReport:
        extracted_claims = self.claim_extractor.extract(answer, default_company=company)
        results: list[ClaimVerification] = []
        for claim in extracted_claims:
            retrieval_query = f"{question}\n{claim.original_text}".strip()
            evidence = self.retriever.retrieve(
                retrieval_query, company=claim.company or company, top_k=3
            )
            if extra_evidence:
                evidence.extend(extra_evidence)
                evidence = self._rank_extra_evidence(retrieval_query, evidence)
            evidence = evidence[:3]
            if not evidence:
                results.append(self._empty_result(claim))
                continue

            pair_results = [
                (passage, self._predict(passage.text, claim.original_text))
                for passage in evidence
            ]
            selected_evidence, nli = max(
                pair_results,
                key=lambda item: (
                    max(item[1]["entailment"], item[1]["contradiction"]),
                    item[0].relevance,
                ),
            )
            selected_evidence = EvidencePassage(
                selected_evidence.text, selected_evidence.source, selected_evidence.relevance
            )
            claim_entities = claim.entities
            evidence_entities = self.entity_extractor.extract(selected_evidence.text)
            entity_mismatch = self._has_entity_mismatch(claim_entities, evidence_entities)
            evidence_claims = self.claim_extractor.extract(
                selected_evidence.text, default_company=company
            )
            evidence_values = [value for item in evidence_claims for value in item.values]
            numerical = self._compare_values(claim.values, evidence_claims)
            status, explanation = self._classify(
                selected_evidence,
                nli,
                numerical,
                entity_mismatch,
                len(claim.values),
                claim.original_text,
                selected_evidence.text,
            )
            results.append(ClaimVerification(
                claim=claim,
                evidence=[selected_evidence],
                evidence_values=evidence_values,
                entailment_probability=nli["entailment"],
                contradiction_probability=nli["contradiction"],
                neutral_probability=nli["neutral"],
                evidence_entities=evidence_entities,
                numerical_comparisons=numerical,
                entity_mismatch=entity_mismatch,
                status=status,
                explanation=explanation,
            ))
        return FinancialClaimVerificationReport(
            question=question,
            answer=answer,
            model_name=getattr(self.nli_classifier, "model_name", "Injected NLI classifier"),
            claims=results,
        )

    def _predict(self, premise: str, hypothesis: str) -> dict[str, float]:
        probabilities = self.nli_classifier.predict(premise, hypothesis)
        required = {"entailment", "contradiction", "neutral"}
        if set(probabilities) != required:
            raise RuntimeError(
                "NLI classifier must return entailment, contradiction, and neutral probabilities."
            )
        total = sum(probabilities.values())
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in probabilities.values()):
            raise RuntimeError("NLI classifier returned an invalid probability.")
        if not math.isclose(total, 1.0, rel_tol=1e-3, abs_tol=1e-3):
            raise RuntimeError(f"NLI probabilities must sum to 1; received {total:.6f}.")
        return probabilities

    def _rank_extra_evidence(
        self, query: str, evidence: list[EvidencePassage]
    ) -> list[EvidencePassage]:
        ranked = self.retriever.rank_passages(query, evidence)
        return ranked

    def _empty_result(self, claim: ExtractedFinancialClaim) -> ClaimVerification:
        return ClaimVerification(
            claim=claim,
            evidence=[],
            evidence_values=[],
            entailment_probability=None,
            contradiction_probability=None,
            neutral_probability=None,
            evidence_entities=[],
            numerical_comparisons=[],
            entity_mismatch=False,
            status="Insufficient Evidence",
            explanation="No relevant local financial evidence was retrieved; this is unsupported, not proven false.",
        )

    def _compare_values(
        self,
        claim_values: list[FinancialValue],
        evidence_claims: list[ExtractedFinancialClaim],
    ) -> list[NumericalComparison]:
        evidence_values = [value for item in evidence_claims for value in item.values]
        comparisons: list[NumericalComparison] = []
        for claim_value in claim_values:
            candidates = [
                value for value in evidence_values
                if self._same_metric(claim_value.metric, value.metric)
            ]
            if not candidates:
                continue
            candidate = min(
                candidates,
                key=lambda value: abs(value.normalized_value - claim_value.normalized_value),
            )
            if claim_value.is_percentage != candidate.is_percentage:
                continue
            if claim_value.currency and candidate.currency and claim_value.currency != candidate.currency:
                comparisons.append(NumericalComparison(
                    "incomparable", claim_value.raw, candidate.raw,
                    explanation=f"Currency mismatch ({claim_value.currency} vs {candidate.currency}); no FX conversion was applied.",
                ))
                continue
            if bool(claim_value.currency) != bool(candidate.currency):
                comparisons.append(NumericalComparison(
                    "incomparable", claim_value.raw, candidate.raw,
                    explanation="Currency is specified for only one value; no currency conversion or assumption was applied.",
                ))
                continue
            if claim_value.scale and candidate.scale and (
                claim_value.currency != candidate.currency
            ):
                comparisons.append(NumericalComparison(
                    "incomparable", claim_value.raw, candidate.raw,
                    explanation="Currency could not be established for both values.",
                ))
                continue
            difference = claim_value.normalized_value - candidate.normalized_value
            denominator = abs(candidate.normalized_value)
            pct_difference = (difference / denominator * 100.0) if denominator else (
                0.0 if difference == 0 else None
            )
            consistent = (
                abs(difference) <= self.numerical_tolerance * max(denominator, 1.0)
                if not claim_value.is_percentage
                else abs(difference) <= 0.1
            )
            comparisons.append(NumericalComparison(
                "consistent" if consistent else "inconsistent",
                claim_value.raw,
                candidate.raw,
                absolute_discrepancy=abs(difference),
                percentage_discrepancy=pct_difference,
                explanation=(
                    "Values match within numerical tolerance."
                    if consistent else
                    f"Claim differs from evidence by {abs(difference):,.6g}"
                    + (f" ({abs(pct_difference):.2f}%)." if pct_difference is not None else ".")
                ),
            ))
        return comparisons

    @staticmethod
    def _same_metric(claim_metric: Optional[str], evidence_metric: Optional[str]) -> bool:
        if not claim_metric or not evidence_metric:
            return claim_metric is None and evidence_metric is None
        if claim_metric == evidence_metric:
            return True
        groups = (
            {"revenue", "growth"},
            {"net_profit", "gross_profit", "operating_profit", "ebitda"},
            {"operating_margin", "net_margin", "gross_margin"},
        )
        return any(claim_metric in group and evidence_metric in group for group in groups)

    @staticmethod
    def _has_entity_mismatch(
        claim_entities: list[FinancialEntity],
        evidence_entities: list[FinancialEntity],
    ) -> bool:
        claim_companies = {
            entity.canonical for entity in claim_entities
            if entity.label in {"COMPANY", "TICKER"} and entity.canonical
        }
        evidence_companies = {
            entity.canonical for entity in evidence_entities
            if entity.label in {"COMPANY", "TICKER"} and entity.canonical
        }
        return bool(claim_companies and evidence_companies and claim_companies.isdisjoint(evidence_companies))

    def _classify(
        self,
        passage: EvidencePassage,
        probabilities: dict[str, float],
        comparisons: list[NumericalComparison],
        entity_mismatch: bool,
        claim_value_count: int,
        claim_text: str,
        evidence_text: str,
    ) -> tuple[str, str]:
        if passage.relevance < self.retriever.min_relevance:
            return "Insufficient Evidence", "Retrieved passages do not meet the minimum relevance threshold."
        numerical_conflict = any(item.status == "inconsistent" for item in comparisons)
        direction_conflict = self._direction_conflict(claim_text, evidence_text)
        if numerical_conflict or entity_mismatch or direction_conflict:
            reasons = []
            if numerical_conflict:
                reasons.append("financial values conflict")
            if entity_mismatch:
                reasons.append("claim and evidence identify different companies")
            if direction_conflict:
                reasons.append("financial direction conflicts")
            return "Hallucinated", "Relevant evidence " + " and ".join(reasons) + "."
        if probabilities["contradiction"] >= self.nli_threshold:
            return "Hallucinated", "DeBERTa NLI classifies the claim as contradicted by relevant evidence."
        if claim_value_count and (
            len(comparisons) < claim_value_count
            or any(item.status == "incomparable" for item in comparisons)
        ):
            return (
                "Potential Hallucination",
                "Evidence is relevant, but the financial value could not be numerically verified.",
            )
        if (
            probabilities["entailment"] >= self.nli_threshold
            and probabilities["contradiction"] < 1.0 - self.nli_threshold
        ):
            return "Faithful", "DeBERTa NLI finds the claim entailed by relevant evidence."
        if probabilities["neutral"] >= self.nli_threshold:
            return "Potential Hallucination", "Evidence is relevant but does not establish or contradict the claim."
        return "Potential Hallucination", "Evidence is relevant, but the model did not find strong entailment or contradiction."

    @staticmethod
    def _direction_conflict(claim: str, evidence: str) -> bool:
        increase = r"\b(?:increased?|grew|growth|rose|risen|up|expanded|higher)\b"
        decrease = r"\b(?:decreased?|declined?|fell|down|contracted|lower|reduced?)\b"
        claim_up = bool(re.search(increase, claim, re.I))
        claim_down = bool(re.search(decrease, claim, re.I))
        evidence_up = bool(re.search(increase, evidence, re.I))
        evidence_down = bool(re.search(decrease, evidence, re.I))
        return (claim_up and evidence_down) or (claim_down and evidence_up)
