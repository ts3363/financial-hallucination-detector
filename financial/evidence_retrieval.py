"""Local evidence indexing and TF-IDF retrieval for financial claims."""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import FINANCIAL_METRICS_MAP
from .data_sources import FinancialReportStore


@dataclass(frozen=True)
class EvidencePassage:
    text: str
    source: str
    relevance: float = 0.0


class FinancialEvidenceRetriever:
    """Indexes local financial text and returns source-attributed passages."""

    SUPPORTED_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".txt", ".pdf"}
    _EVIDENCE_KEYS = {
        "evidence", "evidence_text", "passage", "passages", "snippet",
        "context", "text", "page_content",
    }
    _STOP_WORDS = list(ENGLISH_STOP_WORDS | {"company", "reported", "report", "financial"})

    def __init__(
        self,
        report_store: Optional[FinancialReportStore] = None,
        min_relevance: float = 0.05,
    ):
        self.report_store = report_store
        self.min_relevance = min_relevance
        self._passages: list[EvidencePassage] = []
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            stop_words=self._STOP_WORDS,
            token_pattern=r"(?u)\b[\w$€£₹.-]+\b",
            sublinear_tf=True,
        )
        self._matrix = None

    @property
    def passage_count(self) -> int:
        self._load_report_store()
        return len(self._passages)

    def add_text(self, text: str, source: str) -> int:
        """Add source text as sentence-aware passages; returns passages added."""
        before = len(self._passages)
        for paragraph in re.split(r"\n\s*\n", text or ""):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9₹$€£])", paragraph)
            chunk = ""
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if chunk and len(chunk) + len(sentence) > 700:
                    self._passages.append(EvidencePassage(chunk, source))
                    chunk = ""
                chunk = f"{chunk} {sentence}".strip()
            if chunk:
                self._passages.append(EvidencePassage(chunk, source))
        self._matrix = None
        return len(self._passages) - before

    def add_pdf_bytes(self, content: bytes, source: str) -> int:
        """Extract text from an uploaded PDF without persisting it to disk."""
        import pypdf

        reader = pypdf.PdfReader(io.BytesIO(content))
        added = 0
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            added += self.add_text(page_text, f"{source}, page {page_number}")
        return added

    def load_path(self, path: str | Path) -> int:
        """Index a local PDF, text, CSV, JSON, or JSONL corpus file/directory."""
        target = Path(path).expanduser()
        if not target.exists():
            raise FileNotFoundError(f"Evidence path does not exist: {target}")
        if target.is_dir():
            total = 0
            for file_path in sorted(target.rglob("*")):
                if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_SUFFIXES:
                    total += self._load_file(file_path)
            return total
        if target.suffix.lower() not in self.SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported evidence file type: {target.suffix}")
        return self._load_file(target)

    def load_financebench(self, path: str | Path) -> int:
        """Load evidence fields from a local FinanceBench JSON/JSONL/CSV export."""
        target = Path(path).expanduser()
        if not target.is_file() or target.suffix.lower() not in {".json", ".jsonl", ".csv"}:
            raise ValueError("FinanceBench input must be a local .json, .jsonl, or .csv file.")
        return self._load_file(target)

    def retrieve(
        self,
        query: str,
        company: Optional[str] = None,
        top_k: int = 3,
    ) -> list[EvidencePassage]:
        """Return the most relevant local passages with their source labels."""
        self._load_report_store(company)
        if not self._passages or not query.strip():
            return []
        if self._matrix is None:
            self._matrix = self._vectorizer.fit_transform(p.text for p in self._passages)
        query_vector = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self._matrix).ravel()
        ranked = sorted(
            (
                (
                    self._adjust_metric_relevance(
                        query, f"{passage.text} {passage.source}", float(scores[index])
                    ),
                    int(index),
                )
                for index, passage in enumerate(self._passages)
            ),
            reverse=True,
        )
        results: list[EvidencePassage] = []
        seen: set[tuple[str, str]] = set()
        for score, index in ranked:
            passage = self._passages[index]
            if score < self.min_relevance:
                break
            key = (passage.source, passage.text)
            if key in seen:
                continue
            seen.add(key)
            results.append(EvidencePassage(passage.text, passage.source, score))
            if len(results) >= top_k:
                break
        return results

    def rank_passages(
        self, query: str, passages: list[EvidencePassage]
    ) -> list[EvidencePassage]:
        """Assign TF-IDF relevance to evidence passages supplied by a caller."""
        self._load_report_store()
        if not passages:
            return []
        if self._passages:
            if self._matrix is None:
                self._matrix = self._vectorizer.fit_transform(p.text for p in self._passages)
            query_vector = self._vectorizer.transform([query])
            candidate_vectors = self._vectorizer.transform([item.text for item in passages])
            scores = cosine_similarity(query_vector, candidate_vectors).ravel()
        else:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                stop_words=self._STOP_WORDS,
                token_pattern=r"(?u)\b[\w$€£₹.-]+\b",
                sublinear_tf=True,
            )
            vectors = vectorizer.fit_transform([query] + [item.text for item in passages])
            scores = cosine_similarity(vectors[0:1], vectors[1:]).ravel()
        ranked = [
            EvidencePassage(
                item.text,
                item.source,
                self._adjust_metric_relevance(
                    query, f"{item.text} {item.source}", float(scores[index])
                ),
            )
            for index, item in enumerate(passages)
        ]
        return sorted(ranked, key=lambda item: item.relevance, reverse=True)

    @staticmethod
    def _metric_names(text: str) -> set[str]:
        lowered = text.lower()
        return {
            metric
            for metric, aliases in FINANCIAL_METRICS_MAP.items()
            if any(re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", lowered) for alias in aliases)
        }

    @classmethod
    def _adjust_metric_relevance(cls, query: str, passage: str, score: float) -> float:
        query_metrics = cls._metric_names(query)
        if not query_metrics:
            return score
        passage_metrics = cls._metric_names(passage)
        if query_metrics.intersection(passage_metrics):
            score += 0.20
        elif passage_metrics:
            score = max(0.0, score - 0.08)
        query_periods = set(re.findall(r"\b(?:Q[1-4]|FY\s?\d{2,4})\b", query, re.I))
        passage_periods = set(re.findall(r"\b(?:Q[1-4]|FY\s?\d{2,4})\b", passage, re.I))
        query_quarters = {period.upper() for period in query_periods if period.upper().startswith("Q")}
        passage_quarters = {period.upper() for period in passage_periods if period.upper().startswith("Q")}
        if query_quarters:
            is_annual = bool(re.search(r"\b(?:full year|annual|fiscal year)\b", passage, re.I))
            is_quarterly = bool(passage_quarters or re.search(r"\bquarter(?:ly)?\b", passage, re.I))
            if query_quarters.intersection(passage_quarters) or (is_quarterly and not is_annual):
                score += 0.12
            elif is_annual:
                score -= 0.16
        query_years = {period.upper().replace(" ", "") for period in query_periods if period.upper().startswith("FY")}
        passage_years = {period.upper().replace(" ", "") for period in passage_periods if period.upper().startswith("FY")}
        if query_years and passage_years:
            score += 0.04 if query_years.intersection(passage_years) else -0.04
        return min(1.0, max(0.0, score))

    def _load_report_store(self, company: Optional[str] = None) -> None:
        if self.report_store is None:
            return
        companies = [company] if company else list(self.report_store.companies())
        known_sources = {(item.source, item.text) for item in self._passages}
        changed = False
        for company_name in companies:
            for record in self.report_store.get_evidence_records(company_name):
                passage = EvidencePassage(
                    record.exact_filing_quote,
                    f"{record.source_document}, {record.page_number}",
                )
                key = (passage.source, passage.text)
                if passage.text and key not in known_sources:
                    self._passages.append(passage)
                    known_sources.add(key)
                    changed = True
            for chunk in self.report_store.get_filing_chunks(company_name):
                if not chunk.strip():
                    continue
                source, text = self._split_source_prefix(chunk, company_name)
                key = (source, text)
                if key not in known_sources:
                    self._passages.append(EvidencePassage(text, source))
                    known_sources.add(key)
                    changed = True
        if changed:
            self._matrix = None

    def _load_file(self, path: Path) -> int:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            import pypdf

            reader = pypdf.PdfReader(str(path))
            count = 0
            for page_number, page in enumerate(reader.pages, start=1):
                count += self.add_text(
                    page.extract_text() or "",
                    f"{path.name}, page {page_number}",
                )
            return count
        if suffix in {".txt", ".md"}:
            return self.add_text(path.read_text(encoding="utf-8"), str(path))
        rows = self._read_structured(path)
        before = len(self._passages)
        for row in rows:
            evidence = self._extract_evidence(row)
            source = self._source_name(row, path.name)
            for passage in evidence:
                self.add_text(passage, source)
        return len(self._passages) - before

    @staticmethod
    def _read_structured(path: Path) -> list[dict[str, Any]]:
        if path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as stream:
                return list(csv.DictReader(stream))
        if path.suffix.lower() == ".jsonl":
            rows = []
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if line.strip():
                        value = json.loads(line)
                        rows.append(value if isinstance(value, dict) else {"evidence": value, "row": line_number})
            return rows
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            return [row if isinstance(row, dict) else {"evidence": row} for row in value]
        if isinstance(value, dict):
            for key in ("data", "records", "examples"):
                if isinstance(value.get(key), list):
                    return [row if isinstance(row, dict) else {"evidence": row} for row in value[key]]
            return [value]
        return []

    def _extract_evidence(self, row: dict[str, Any]) -> list[str]:
        found: list[str] = []

        def visit(value: Any, key: Optional[str] = None) -> None:
            if isinstance(value, str):
                if key and key.lower() == "evidence":
                    try:
                        decoded = json.loads(value)
                    except json.JSONDecodeError:
                        found.append(value.strip())
                        return
                    if decoded != value:
                        visit(decoded, key)
                        return
                if key and key.lower() in self._EVIDENCE_KEYS and value.strip():
                    found.append(value.strip())
            elif isinstance(value, dict):
                for child_key, child in value.items():
                    visit(child, child_key)
            elif isinstance(value, list):
                for child in value:
                    visit(child, key)

        for key, value in row.items():
            if key.lower() in self._EVIDENCE_KEYS:
                visit(value, key)
        return list(dict.fromkeys(found))

    @staticmethod
    def _source_name(row: dict[str, Any], fallback: str) -> str:
        for key in ("doc_name", "document", "source", "filename", "file_name"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return fallback

    @staticmethod
    def _split_source_prefix(text: str, company: str) -> tuple[str, str]:
        match = re.match(r"^([^:]{3,100}):\s+(.+)$", text.strip())
        if match:
            return match.group(1), match.group(2)
        return f"{company} local filing passage", text.strip()
