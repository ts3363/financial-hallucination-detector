"""
financial/rag_indexer.py
========================
Financial Document & PDF Ingestion Engine.
Extracts financial tables, statement lines, and metric ground truths from
corporate PDFs (Annual Reports, 10-K, 10-Q, quarterly press releases)
and indexes them directly into FinancialReportStore.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

from .sample_data import AuditedMetricRecord
from .data_sources import FinancialReportStore
from .config import CURRENCY_SCALES, CURRENCY_SYMBOLS


@dataclass
class ExtractedFilingMetric:
    metric: str
    reported_value: float
    reported_str: str
    currency: str
    scale: str
    page_number: int
    raw_line: str


class FinancialDocumentIndexer:
    """
    Ingests and parses financial regulatory documents and PDFs.
    Uses pypdf to extract pages, statement lines, and key metrics.
    """

    STATEMENT_PATTERNS = [
        (re.compile(r'revenue\s+from\s+operations', re.IGNORECASE), "revenue"),
        (re.compile(r'total\s+revenue', re.IGNORECASE), "revenue"),
        (re.compile(r'profit\s+for\s+the\s+(?:period|year)', re.IGNORECASE), "net_profit"),
        (re.compile(r'net\s+profit', re.IGNORECASE), "net_profit"),
        (re.compile(r'operating\s+profit', re.IGNORECASE), "operating_profit"),
        (re.compile(r'operating\s+margin', re.IGNORECASE), "operating_margin"),
        (re.compile(r'basic\s+eps|diluted\s+eps', re.IGNORECASE), "eps"),
        (re.compile(r'dividend\s+per\s+share', re.IGNORECASE), "dividend"),
    ]

    NUM_EXTRACTOR = re.compile(
        r'([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)\s*(%|cr|crore|crores|billion|million)?\b',
        re.IGNORECASE
    )

    def __init__(self, report_store: Optional[FinancialReportStore] = None):
        self.report_store = report_store or FinancialReportStore()

    def index_text_document(
        self,
        company: str,
        document_title: str,
        text_content: str,
        period: str = "Q4"
    ) -> List[AuditedMetricRecord]:
        """
        Parse raw text of an earnings release or report, extract metrics,
        and index them into the report store.
        """
        lines = text_content.splitlines()
        extracted: List[AuditedMetricRecord] = []

        currency = "INR" if company.upper() in ("TCS", "INFOSYS", "RELIANCE", "WIPRO") else "USD"
        scale = "crore" if currency == "INR" else "billion"
        multiplier = CURRENCY_SCALES.get(scale, 1.0)

        for line_idx, line in enumerate(lines):
            line_str = line.strip()
            if not line_str or len(line_str) < 5:
                continue

            for pattern, metric_key in self.STATEMENT_PATTERNS:
                if pattern.search(line_str):
                    matches = list(self.NUM_EXTRACTOR.finditer(line_str))
                    if matches:
                        match = matches[-1]  # Take right-most financial figure in statement row
                        raw_num = match.group(1).replace(',', '')
                        unit = match.group(2) or scale
                        try:
                            val = float(raw_num)
                        except ValueError:
                            continue

                        is_pct = "%" in unit or "margin" in metric_key
                        norm_val = val if is_pct else (val * multiplier)
                        rep_str = f"{val}%" if is_pct else (f"₹{val:,.0f} {scale}" if currency == "INR" else f"${val:.2f} {scale}")

                        rec = AuditedMetricRecord(
                            company=company.upper(),
                            metric=metric_key,
                            period=period,
                            reported_value=val,
                            reported_str=rep_str,
                            currency=currency,
                            scale="%" if is_pct else scale,
                            normalized_value=norm_val,
                            source_document=document_title,
                            page_number=f"Line {line_idx + 1}",
                            exact_filing_quote=line_str
                        )
                        self.report_store.add_record(rec)
                        extracted.append(rec)
                        break

        # Also store the raw passages in report store chunks
        self.report_store.add_text_chunks(company, [f"{document_title}: {line.strip()}" for line in lines if len(line.strip()) > 30])
        return extracted

    def index_pdf_file(
        self,
        company: str,
        pdf_path: str,
        period: str = "Q4"
    ) -> List[AuditedMetricRecord]:
        """
        Extract text from a PDF file using pypdf and index extracted financial metrics.
        """
        try:
            import pypdf
        except ImportError:
            raise RuntimeError("pypdf is required for PDF ingestion. Run: pip install pypdf")

        reader = pypdf.PdfReader(pdf_path)
        all_text = []
        for i, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            if txt:
                all_text.append(f"--- Page {i+1} ---\n{txt}")

        full_doc = "\n".join(all_text)
        doc_name = pdf_path.split("/")[-1].split("\\")[-1]
        return self.index_text_document(company, doc_name, full_doc, period=period)
