"""
financial/claim_extractor.py
============================
Domain-specific financial claim extraction module.
Extracts companies, financial metrics, fiscal periods, currencies,
scale multipliers, and numerical quantities from natural language answers.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

from .config import (
    CURRENCY_SCALES,
    CURRENCY_SYMBOLS,
    FINANCIAL_METRICS_MAP,
)


@dataclass
class FinancialClaim:
    """A structured representation of a quantitative financial assertion."""
    raw_text: str                       # The original sentence / clause containing the claim
    company: Optional[str]              # Company or ticker mentioned (e.g. "TCS", "Apple")
    metric: str                         # Standardized metric name (e.g. "revenue", "operating_margin")
    metric_label: str                   # Human-readable metric (e.g. "Revenue", "Operating Margin")
    period: Optional[str]               # e.g. "Q4 FY24", "FY24", "Q4"
    numeric_value: float                # Extracted number (e.g. 65000.0, 26.0)
    raw_number_str: str                 # Exact string in text (e.g. "₹65,000 crore", "26%")
    currency: Optional[str]             # "INR", "USD", etc.
    scale: Optional[str]                # "crore", "billion", "percentage", etc.
    normalized_value: float             # Standardized base scale value for comparison
    is_percentage: bool = False         # True if metric is ratio/percentage


class FinancialClaimExtractor:
    """
    Extracts structured financial claims from LLM responses.
    
    Handles both Indian financial notation (e.g., ₹65,000 crore, Rs 12,434 cr)
    and Global notation (e.g., $94.93 billion, 46.2% margin).
    """

    KNOWN_COMPANIES: Dict[str, List[str]] = {
        "TCS": ["tcs", "tata consultancy services", "tata consultancy", "tcs.ns"],
        "INFOSYS": ["infosys", "infy", "infy.ns"],
        "RELIANCE": ["reliance", "ril", "reliance industries", "reliance.ns"],
        "APPLE": ["apple", "aapl", "apple inc"],
        "MICROSOFT": ["microsoft", "msft"],
        "GOOGLE": ["google", "alphabet", "googl", "goog"],
        "NVIDIA": ["nvidia", "nvda"],
        "AMAZON": ["amazon", "amzn"],
        "META": ["meta", "facebook", "fb"],
        "TESLA": ["tesla", "tsla"],
        "HDFC": ["hdfc", "hdfc bank", "hdfcbank.ns", "hdfcbank"],
        "WIPRO": ["wipro", "wit", "wipro.ns"],
        "TATAMOTORS": ["tata motors", "tatamotors", "tatamotors.ns"],
        "ICICIBANK": ["icici", "icici bank", "icicibank.ns"],
        "BHARTIARTL": ["bharti airtel", "airtel", "bhartiartl.ns"],
    }

    # Regex for fiscal periods like "Q4 FY24", "Q4", "FY 2024", "FY24", "fourth quarter"
    PERIOD_PATTERN = re.compile(
        r'\b(?:(q[1-4])\s*(?:of\s*)?)?(?:fy\s*([0-9]{2,4})|20[1-3][0-9])\b'
        r'|\b(q[1-4])\b'
        r'|\b(fourth|third|second|first)\s+quarter\b'
        r'|\b(full\s*year|annual|fy[0-9]{2,4})\b',
        re.IGNORECASE
    )

    # Regex to capture financial numbers with currency and scale:
    # e.g. "₹65,000 crore", "Rs. 12,434 cr", "$94.93 billion", "26.0%", "₹28 per share"
    FIN_NUM_PATTERN = re.compile(
        r'(?:(₹|\$|€|£|rs\.?|inr|usd)\s*)?'
        r'([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)'
        r'\s*(crores?|cr|lakhs?|lac|billions?|bn|b|millions?|m|mn|trillions?|t|%|per\s*share)?\b',
        re.IGNORECASE
    )

    def extract_claims(self, text: str, default_company: Optional[str] = None) -> List[FinancialClaim]:
        """
        Analyze text and return all extracted quantitative financial claims.
        """
        claims: List[FinancialClaim] = []
        if not text:
            return claims

        # Split into sentences
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text) if s.strip()]

        # Global company detection if not specified per sentence
        global_company = default_company or self._detect_company(text)

        for sent in sentences:
            sent_company = self._detect_company(sent) or global_company
            sent_period = self._detect_period(sent)
            detected_metrics = self._detect_metric_spans(sent)

            if not detected_metrics:
                continue

            # Find all numbers in the sentence
            num_matches = list(self.FIN_NUM_PATTERN.finditer(sent))
            if not num_matches:
                continue

            # Filter valid financial numbers (exclude quarter numbers like "4" in Q4 or year "24" in FY24)
            valid_numbers = []
            for match in num_matches:
                start_idx = match.start()
                raw_num = match.group(2)
                curr_str = match.group(1)
                scale_str = match.group(3)

                # Check preceding characters to see if this is part of Q4, FY24, FY2024, etc.
                prefix = sent[max(0, start_idx - 3):start_idx].lower()
                if re.search(r'[qf][y]?$', prefix):
                    continue  # Ignore "4" in "Q4" or "24" in "FY24"
                
                # If neither currency nor scale is present:
                if not curr_str and not scale_str:
                    # Ignore single digit numbers (e.g. 1, 2, 3, 4)
                    if len(raw_num) == 1 and not ('.' in raw_num):
                        continue
                    # Ignore 4-digit years like 2024, 2023, 2025
                    if re.match(r'^20[1-3][0-9]$', raw_num):
                        continue

                valid_numbers.append(match)

            if not valid_numbers:
                continue

            # Assign each valid number to the closest relevant metric in the sentence
            for match in valid_numbers:
                num_start = match.start()
                num_end = match.end()
                curr_str = match.group(1)
                raw_num_str = match.group(2)
                scale_str = match.group(3)

                # Clean numeric value
                clean_num_str = raw_num_str.replace(',', '')
                try:
                    numeric_val = float(clean_num_str)
                except ValueError:
                    continue

                scale = scale_str.lower().strip() if scale_str else None
                is_percentage = (scale == "%")

                # Semantic distance function between metric span and number
                def _calc_dist(m: Dict[str, Any]) -> float:
                    if m["end"] <= num_start:
                        d = float(num_start - m["end"])
                    elif m["start"] >= num_end:
                        d = float(m["start"] - num_end) * 1.5
                    else:
                        d = 0.0
                    
                    # If number has % or metric is margin, bias appropriately
                    if is_percentage and "margin" in m["metric"]:
                        d *= 0.4
                    elif not is_percentage and "margin" in m["metric"]:
                        d *= 3.0
                    return d

                # Find the best matching metric
                best_metric = min(detected_metrics, key=_calc_dist)
                metric_key = best_metric["metric"]

                # Determine currency
                currency = None
                if curr_str:
                    cleaned_curr = curr_str.lower().strip()
                    currency = CURRENCY_SYMBOLS.get(cleaned_curr, "INR" if "₹" in cleaned_curr or "rs" in cleaned_curr else "USD")
                elif sent_company in ("TCS", "INFOSYS", "RELIANCE", "HDFC", "WIPRO"):
                    currency = "INR"
                elif sent_company in ("APPLE", "MICROSOFT", "GOOGLE", "NVIDIA", "TESLA"):
                    currency = "USD"

                if scale == "%" or "margin" in metric_key:
                    is_percentage = True
                    if metric_key in ("revenue", "net_profit", "ebitda", "sales"):
                        metric_key = f"{metric_key}_growth"

                # Calculate normalized value (base scale)
                multiplier = 1.0
                if scale and scale in CURRENCY_SCALES:
                    multiplier = CURRENCY_SCALES[scale]

                normalized_val = numeric_val if is_percentage else (numeric_val * multiplier)

                full_num_repr = match.group(0).strip()

                claim = FinancialClaim(
                    raw_text=sent,
                    company=sent_company,
                    metric=metric_key,
                    metric_label=metric_key.replace('_', ' ').title(),
                    period=sent_period,
                    numeric_value=numeric_val,
                    raw_number_str=full_num_repr,
                    currency=currency,
                    scale=scale or ("%" if is_percentage else "unit"),
                    normalized_value=normalized_val,
                    is_percentage=is_percentage,
                )
                claims.append(claim)

        return claims

    def _detect_company(self, text: str) -> Optional[str]:
        """Identify which financial entity is referenced."""
        text_lower = text.lower()
        for company_id, aliases in self.KNOWN_COMPANIES.items():
            for alias in aliases:
                pattern = rf'\b{re.escape(alias)}\b'
                if re.search(pattern, text_lower):
                    return company_id

        # Fallback pattern for research prompts like "Analyze XYZ before I invest" or "Research XYZ"
        m = re.search(r'\b(?:analyze|research|audit|investigate|evaluate|check|assess)\s+([a-zA-Z0-9_\.\-]+)', text, re.IGNORECASE)
        if m:
            candidate = m.group(1).upper()
            if candidate.lower() not in {"the", "this", "that", "my", "a", "an", "our", "company", "stock"}:
                return candidate
        return None

    def _detect_period(self, text: str) -> Optional[str]:
        """Extract fiscal quarter / year."""
        m = self.PERIOD_PATTERN.search(text)
        if m:
            raw = m.group(0).upper().strip()
            if "FOURTH QUARTER" in raw:
                return "Q4"
            elif "THIRD QUARTER" in raw:
                return "Q3"
            elif "SECOND QUARTER" in raw:
                return "Q2"
            elif "FIRST QUARTER" in raw:
                return "Q1"
            return raw
        return None

    def _detect_metric_spans(self, text: str) -> List[Dict[str, Any]]:
        """Identify metric mentions and their character start index."""
        text_lower = text.lower()
        found: List[Dict[str, Any]] = []
        for metric_key, aliases in FINANCIAL_METRICS_MAP.items():
            for alias in aliases:
                pattern = rf'\b{re.escape(alias)}\b'
                for m in re.finditer(pattern, text_lower):
                    found.append({
                        "metric": metric_key,
                        "alias": alias,
                        "start": m.start(),
                        "end": m.end(),
                    })
                    break
        return found
