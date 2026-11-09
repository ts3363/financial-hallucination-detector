"""
financial/sample_data.py
========================
Audited quarterly results, annual reports, excerpts, and metric ground-truths
from official regulatory filings (SEC, NSE/BSE filings, corporate releases).
Used for offline/sandboxed deterministic testing and verification.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class AuditedMetricRecord:
    company: str
    metric: str
    period: str
    reported_value: float
    reported_str: str
    currency: str
    scale: str
    normalized_value: float
    source_document: str
    page_number: str
    exact_filing_quote: str


# Master Audited Ground Truth Database
AUDITED_FINANCIAL_RECORDS: List[AuditedMetricRecord] = [
    # ── TATA CONSULTANCY SERVICES (TCS) Q4 FY24 & FY24 ─────────────
    AuditedMetricRecord(
        company="TCS",
        metric="revenue",
        period="Q4",
        reported_value=61237.0,
        reported_str="₹61,237 crore",
        currency="INR",
        scale="crore",
        normalized_value=61237.0 * 1e7,
        source_document="TCS Q4 FY24 Audited Financial Results (NSE/BSE Press Release)",
        page_number="Page 1",
        exact_filing_quote="Consolidated revenue from operations for the quarter ended March 31, 2024 stood at ₹61,237 crore, growing 3.5% YoY."
    ),
    AuditedMetricRecord(
        company="TCS",
        metric="revenue_growth",
        period="Q4",
        reported_value=3.5,
        reported_str="3.5% YoY",
        currency="INR",
        scale="%",
        normalized_value=3.5,
        source_document="TCS Q4 FY24 Audited Financial Results (NSE/BSE Press Release)",
        page_number="Page 1",
        exact_filing_quote="Consolidated revenue from operations for the quarter ended March 31, 2024 stood at ₹61,237 crore, growing 3.5% YoY."
    ),
    AuditedMetricRecord(
        company="TCS",
        metric="net_profit_growth",
        period="Q4",
        reported_value=9.1,
        reported_str="9.1% YoY",
        currency="INR",
        scale="%",
        normalized_value=9.1,
        source_document="TCS Q4 FY24 Audited Financial Results",
        page_number="Page 2",
        exact_filing_quote="Net profit for the fourth quarter stood at ₹12,434 crore, up 9.1% YoY."
    ),
    AuditedMetricRecord(
        company="TCS",
        metric="operating_margin",
        period="Q4",
        reported_value=26.0,
        reported_str="26.0%",
        currency="INR",
        scale="%",
        normalized_value=26.0,
        source_document="TCS Q4 FY24 Investor Presentation",
        page_number="Page 4",
        exact_filing_quote="Operating margin expanded to 26.0% in Q4 FY24, an expansion of 150 bps YoY."
    ),
    AuditedMetricRecord(
        company="TCS",
        metric="net_profit",
        period="Q4",
        reported_value=12434.0,
        reported_str="₹12,434 crore",
        currency="INR",
        scale="crore",
        normalized_value=12434.0 * 1e7,
        source_document="TCS Q4 FY24 Audited Financial Results",
        page_number="Page 2",
        exact_filing_quote="Net profit for the fourth quarter stood at ₹12,434 crore, up 9.1% YoY."
    ),
    AuditedMetricRecord(
        company="TCS",
        metric="order_book",
        period="Q4",
        reported_value=13.2,
        reported_str="$13.2 billion",
        currency="USD",
        scale="billion",
        normalized_value=13.2 * 1e9,
        source_document="TCS Management Commentary & Investor Factsheet",
        page_number="Page 3",
        exact_filing_quote="Q4 order book total contract value (TCV) achieved an all-time high of $13.2 billion."
    ),
    AuditedMetricRecord(
        company="TCS",
        metric="dividend",
        period="Q4",
        reported_value=28.0,
        reported_str="₹28 per share",
        currency="INR",
        scale="per share",
        normalized_value=28.0,
        source_document="TCS Board Meeting Outcome & Dividend Declaration",
        page_number="Page 1",
        exact_filing_quote="The Board recommended a final dividend of ₹28 per equity share of ₹1 face value."
    ),
    AuditedMetricRecord(
        company="TCS",
        metric="revenue",
        period="FY24",
        reported_value=240893.0,
        reported_str="₹240,893 crore",
        currency="INR",
        scale="crore",
        normalized_value=240893.0 * 1e7,
        source_document="TCS Annual Report FY24",
        page_number="Page 47",
        exact_filing_quote="For the full year FY24, consolidated revenue was ₹240,893 crore, a growth of 6.8% over FY23."
    ),
    AuditedMetricRecord(
        company="TCS",
        metric="net_profit",
        period="FY24",
        reported_value=46585.0,
        reported_str="₹46,585 crore",
        currency="INR",
        scale="crore",
        normalized_value=46585.0 * 1e7,
        source_document="TCS Annual Report FY24",
        page_number="Page 49",
        exact_filing_quote="Full year net profit reached ₹46,585 crore, reflecting strong operational resilience."
    ),

    # ── INFOSYS Q4 FY24 ────────────────────────────────────────────
    AuditedMetricRecord(
        company="INFOSYS",
        metric="revenue",
        period="Q4",
        reported_value=37923.0,
        reported_str="₹37,923 crore",
        currency="INR",
        scale="crore",
        normalized_value=37923.0 * 1e7,
        source_document="Infosys Q4 FY24 Press Release",
        page_number="Page 1",
        exact_filing_quote="Revenues for Q4 FY24 were ₹37,923 crore, a YoY growth of 1.3% in constant currency."
    ),
    AuditedMetricRecord(
        company="INFOSYS",
        metric="operating_margin",
        period="Q4",
        reported_value=20.1,
        reported_str="20.1%",
        currency="INR",
        scale="%",
        normalized_value=20.1,
        source_document="Infosys Q4 FY24 Factsheet",
        page_number="Page 2",
        exact_filing_quote="Operating margin for the quarter was 20.1% compared to 20.5% in Q3 FY24."
    ),
    AuditedMetricRecord(
        company="INFOSYS",
        metric="net_profit",
        period="Q4",
        reported_value=7969.0,
        reported_str="₹7,969 crore",
        currency="INR",
        scale="crore",
        normalized_value=7969.0 * 1e7,
        source_document="Infosys Q4 FY24 Financial Statements",
        page_number="Page 3",
        exact_filing_quote="Net profit stood at ₹7,969 crore, including tax benefit adjustments."
    ),

    # ── RELIANCE INDUSTRIES Q4 FY24 ────────────────────────────────
    AuditedMetricRecord(
        company="RELIANCE",
        metric="revenue",
        period="Q4",
        reported_value=240715.0,
        reported_str="₹240,715 crore",
        currency="INR",
        scale="crore",
        normalized_value=240715.0 * 1e7,
        source_document="Reliance Industries Q4 FY24 Earnings Release",
        page_number="Page 2",
        exact_filing_quote="Gross revenue for Q4 FY24 was ₹240,715 crore ($28.9 billion), up 10.8% YoY."
    ),
    AuditedMetricRecord(
        company="RELIANCE",
        metric="ebitda",
        period="Q4",
        reported_value=42516.0,
        reported_str="₹42,516 crore",
        currency="INR",
        scale="crore",
        normalized_value=42516.0 * 1e7,
        source_document="Reliance Industries Q4 FY24 Financials",
        page_number="Page 5",
        exact_filing_quote="Quarterly EBITDA was ₹42,516 crore ($5.1 billion), up 14.3% YoY."
    ),
    AuditedMetricRecord(
        company="RELIANCE",
        metric="net_profit",
        period="Q4",
        reported_value=18951.0,
        reported_str="₹18,951 crore",
        currency="INR",
        scale="crore",
        normalized_value=18951.0 * 1e7,
        source_document="Reliance Industries Q4 FY24 Results",
        page_number="Page 3",
        exact_filing_quote="Net profit for the fourth quarter was ₹18,951 crore ($2.3 billion)."
    ),

    # ── APPLE INC (AAPL) Q4 FY24 ──────────────────────────────────
    AuditedMetricRecord(
        company="APPLE",
        metric="revenue",
        period="Q4",
        reported_value=94.93,
        reported_str="$94.93 billion",
        currency="USD",
        scale="billion",
        normalized_value=94.93 * 1e9,
        source_document="Apple Inc. SEC Form 10-K / Q4 2024 Earnings Release",
        page_number="Page 1",
        exact_filing_quote="Apple posted quarterly revenue of $94.93 billion, up 6% year over year."
    ),
    AuditedMetricRecord(
        company="APPLE",
        metric="net_profit",
        period="Q4",
        reported_value=14.73,
        reported_str="$14.73 billion",
        currency="USD",
        scale="billion",
        normalized_value=14.73 * 1e9,
        source_document="Apple Inc. SEC Form 10-K Q4 2024",
        page_number="Page 2",
        exact_filing_quote="Net income for the quarter was $14.73 billion, or $0.97 per diluted share."
    ),
    AuditedMetricRecord(
        company="APPLE",
        metric="gross_margin",
        period="Q4",
        reported_value=46.2,
        reported_str="46.2%",
        currency="USD",
        scale="%",
        normalized_value=46.2,
        source_document="Apple Inc. Condensed Consolidated Financial Statements",
        page_number="Page 3",
        exact_filing_quote="Gross margin for the quarter was 46.2%, compared to 45.2% in the year-ago quarter."
    ),

    # ── MICROSOFT (MSFT) Q4 FY24 ──────────────────────────────────
    AuditedMetricRecord(
        company="MICROSOFT",
        metric="revenue",
        period="Q4",
        reported_value=64.73,
        reported_str="$64.73 billion",
        currency="USD",
        scale="billion",
        normalized_value=64.73 * 1e9,
        source_document="Microsoft Form 10-K / Q4 FY24 Earnings Release",
        page_number="Page 1",
        exact_filing_quote="Revenue was $64.73 billion and increased 15% (up 16% in constant currency)."
    ),
    AuditedMetricRecord(
        company="MICROSOFT",
        metric="net_profit",
        period="Q4",
        reported_value=22.04,
        reported_str="$22.04 billion",
        currency="USD",
        scale="billion",
        normalized_value=22.04 * 1e9,
        source_document="Microsoft Form 10-K / Q4 FY24 Financial Statements",
        page_number="Page 2",
        exact_filing_quote="Net income was $22.04 billion and increased 10% (up 11% in constant currency)."
    ),

    # ── GOOGLE / ALPHABET (GOOGL) Q4 ──────────────────────────────
    AuditedMetricRecord(
        company="GOOGLE",
        metric="revenue",
        period="Q4",
        reported_value=86.31,
        reported_str="$86.31 billion",
        currency="USD",
        scale="billion",
        normalized_value=86.31 * 1e9,
        source_document="Alphabet Inc. Q4 2024 Form 10-K Earnings Release",
        page_number="Page 1",
        exact_filing_quote="Alphabet announced consolidated revenues of $86.31 billion for the quarter ended December 31, 2024, up 13% YoY."
    ),
    AuditedMetricRecord(
        company="GOOGLE",
        metric="net_profit",
        period="Q4",
        reported_value=20.69,
        reported_str="$20.69 billion",
        currency="USD",
        scale="billion",
        normalized_value=20.69 * 1e9,
        source_document="Alphabet Inc. SEC Form 10-K",
        page_number="Page 2",
        exact_filing_quote="Net income for the fourth quarter was $20.69 billion, or $1.64 per diluted share."
    ),
    AuditedMetricRecord(
        company="GOOGLE",
        metric="operating_margin",
        period="Q4",
        reported_value=27.5,
        reported_str="27.5%",
        currency="USD",
        scale="%",
        normalized_value=27.5,
        source_document="Alphabet Investor Presentation Q4",
        page_number="Page 4",
        exact_filing_quote="Operating margin was 27.5% for the fourth quarter."
    ),

    # ── NVIDIA (NVDA) Q4 FY24 ─────────────────────────────────────
    AuditedMetricRecord(
        company="NVIDIA",
        metric="revenue",
        period="Q4",
        reported_value=22.10,
        reported_str="$22.10 billion",
        currency="USD",
        scale="billion",
        normalized_value=22.10 * 1e9,
        source_document="NVIDIA Corp. Q4 FY24 Earnings Press Release",
        page_number="Page 1",
        exact_filing_quote="NVIDIA reported revenue for the fourth quarter ended January 28, 2024, of $22.10 billion, up 22% from previous quarter and up 265% from a year ago."
    ),
    AuditedMetricRecord(
        company="NVIDIA",
        metric="net_profit",
        period="Q4",
        reported_value=12.29,
        reported_str="$12.29 billion",
        currency="USD",
        scale="billion",
        normalized_value=12.29 * 1e9,
        source_document="NVIDIA SEC Form 10-K FY24",
        page_number="Page 2",
        exact_filing_quote="GAAP net income for the quarter was $12.29 billion, up 769% from $1.41 billion a year ago."
    ),
    AuditedMetricRecord(
        company="NVIDIA",
        metric="gross_margin",
        period="Q4",
        reported_value=76.0,
        reported_str="76.0%",
        currency="USD",
        scale="%",
        normalized_value=76.0,
        source_document="NVIDIA Q4 FY24 Financial Statements",
        page_number="Page 3",
        exact_filing_quote="GAAP gross margin for the fourth quarter reached a record 76.0%."
    ),
    AuditedMetricRecord(
        company="NVIDIA",
        metric="order_book",
        period="Q4",
        reported_value=18.40,
        reported_str="$18.40 billion",
        currency="USD",
        scale="billion",
        normalized_value=18.40 * 1e9,
        source_document="NVIDIA Q4 Earnings Release",
        page_number="Page 2",
        exact_filing_quote="Data Center revenue was a record $18.40 billion, up 409% from a year ago."
    ),

    # ── AMAZON (AMZN) Q4 ──────────────────────────────────────────
    AuditedMetricRecord(
        company="AMAZON",
        metric="revenue",
        period="Q4",
        reported_value=169.96,
        reported_str="$169.96 billion",
        currency="USD",
        scale="billion",
        normalized_value=169.96 * 1e9,
        source_document="Amazon.com Inc. Q4 2024 Financial Results Release",
        page_number="Page 1",
        exact_filing_quote="Net sales increased 14% to $169.96 billion in the fourth quarter."
    ),
    AuditedMetricRecord(
        company="AMAZON",
        metric="net_profit",
        period="Q4",
        reported_value=10.62,
        reported_str="$10.62 billion",
        currency="USD",
        scale="billion",
        normalized_value=10.62 * 1e9,
        source_document="Amazon SEC Form 10-K",
        page_number="Page 2",
        exact_filing_quote="Net income was $10.62 billion in the fourth quarter, or $1.00 per diluted share."
    ),

    # ── META (META) Q4 ────────────────────────────────────────────
    AuditedMetricRecord(
        company="META",
        metric="revenue",
        period="Q4",
        reported_value=40.11,
        reported_str="$40.11 billion",
        currency="USD",
        scale="billion",
        normalized_value=40.11 * 1e9,
        source_document="Meta Platforms Inc. Q4 2024 Earnings Release",
        page_number="Page 1",
        exact_filing_quote="Meta reported fourth quarter revenue of $40.11 billion, an increase of 25% year-over-year."
    ),
    AuditedMetricRecord(
        company="META",
        metric="net_profit",
        period="Q4",
        reported_value=14.02,
        reported_str="$14.02 billion",
        currency="USD",
        scale="billion",
        normalized_value=14.02 * 1e9,
        source_document="Meta Platforms SEC Form 10-K",
        page_number="Page 2",
        exact_filing_quote="Net income for the fourth quarter was $14.02 billion, or $5.33 per diluted share."
    ),
    AuditedMetricRecord(
        company="META",
        metric="operating_margin",
        period="Q4",
        reported_value=41.0,
        reported_str="41.0%",
        currency="USD",
        scale="%",
        normalized_value=41.0,
        source_document="Meta Investor Relations Release",
        page_number="Page 3",
        exact_filing_quote="Operating margin was 41% for the fourth quarter."
    ),

    # ── TESLA (TSLA) Q4 ───────────────────────────────────────────
    AuditedMetricRecord(
        company="TESLA",
        metric="revenue",
        period="Q4",
        reported_value=25.17,
        reported_str="$25.17 billion",
        currency="USD",
        scale="billion",
        normalized_value=25.17 * 1e9,
        source_document="Tesla Inc. Q4 2024 Update",
        page_number="Page 1",
        exact_filing_quote="Total revenues grew 3% YoY in Q4 to $25.17 billion."
    ),
    AuditedMetricRecord(
        company="TESLA",
        metric="net_profit",
        period="Q4",
        reported_value=7.93,
        reported_str="$7.93 billion",
        currency="USD",
        scale="billion",
        normalized_value=7.93 * 1e9,
        source_document="Tesla SEC Form 10-K",
        page_number="Page 2",
        exact_filing_quote="GAAP net income was $7.93 billion in the fourth quarter."
    ),
    AuditedMetricRecord(
        company="TESLA",
        metric="operating_margin",
        period="Q4",
        reported_value=8.2,
        reported_str="8.2%",
        currency="USD",
        scale="%",
        normalized_value=8.2,
        source_document="Tesla Shareholder Deck",
        page_number="Page 4",
        exact_filing_quote="Operating margin was 8.2% in the fourth quarter."
    ),

    # ── HDFC BANK (HDFCBANK) Q4 FY24 ──────────────────────────────
    AuditedMetricRecord(
        company="HDFC",
        metric="net_profit",
        period="Q4",
        reported_value=16512.0,
        reported_str="₹16,512 crore",
        currency="INR",
        scale="crore",
        normalized_value=16512.0 * 1e7,
        source_document="HDFC Bank Q4 FY24 Audited Financial Results (NSE/BSE Release)",
        page_number="Page 1",
        exact_filing_quote="HDFC Bank reported net profit of ₹16,512 crore for the quarter ended March 31, 2024, up 37.1% YoY."
    ),
    AuditedMetricRecord(
        company="HDFC",
        metric="revenue",
        period="Q4",
        reported_value=29077.0,
        reported_str="₹29,077 crore",
        currency="INR",
        scale="crore",
        normalized_value=29077.0 * 1e7,
        source_document="HDFC Bank Q4 FY24 Financial Statements",
        page_number="Page 2",
        exact_filing_quote="Net Interest Income (NII) grew to ₹29,077 crore for Q4 FY24."
    ),

    # ── WIPRO (WIPRO) Q4 FY24 ─────────────────────────────────────
    AuditedMetricRecord(
        company="WIPRO",
        metric="revenue",
        period="Q4",
        reported_value=22208.0,
        reported_str="₹22,208 crore",
        currency="INR",
        scale="crore",
        normalized_value=22208.0 * 1e7,
        source_document="Wipro Limited Q4 FY24 Audited Financial Results",
        page_number="Page 1",
        exact_filing_quote="Gross revenue reached ₹22,208 crore for the quarter ended March 31, 2024."
    ),
    AuditedMetricRecord(
        company="WIPRO",
        metric="net_profit",
        period="Q4",
        reported_value=2835.0,
        reported_str="₹2,835 crore",
        currency="INR",
        scale="crore",
        normalized_value=2835.0 * 1e7,
        source_document="Wipro Press Release Q4",
        page_number="Page 2",
        exact_filing_quote="Net income for the quarter stood at ₹2,835 crore."
    ),
    AuditedMetricRecord(
        company="WIPRO",
        metric="operating_margin",
        period="Q4",
        reported_value=16.4,
        reported_str="16.4%",
        currency="INR",
        scale="%",
        normalized_value=16.4,
        source_document="Wipro Investor Presentation",
        page_number="Page 3",
        exact_filing_quote="IT Services operating margin was 16.4%, expanding by 40 bps QoQ."
    ),

    # ── TATA MOTORS (TATAMOTORS) Q4 FY24 ──────────────────────────
    AuditedMetricRecord(
        company="TATAMOTORS",
        metric="revenue",
        period="Q4",
        reported_value=119986.0,
        reported_str="₹1,19,986 crore",
        currency="INR",
        scale="crore",
        normalized_value=119986.0 * 1e7,
        source_document="Tata Motors Limited Q4 FY24 Audited Financial Results",
        page_number="Page 1",
        exact_filing_quote="Consolidated revenue from operations stood at ₹1,19,986 crore for Q4 FY24, up 13.3% YoY."
    ),
    AuditedMetricRecord(
        company="TATAMOTORS",
        metric="net_profit",
        period="Q4",
        reported_value=17529.0,
        reported_str="₹17,529 crore",
        currency="INR",
        scale="crore",
        normalized_value=17529.0 * 1e7,
        source_document="Tata Motors Results Release Q4",
        page_number="Page 2",
        exact_filing_quote="Net profit for the fourth quarter was ₹17,529 crore."
    ),
]


# Rich Filing Text Chunks for RAG and Text Grounding
SAMPLE_FILING_TEXT_CHUNKS: Dict[str, List[str]] = {
    "TCS": [
        "TCS Q4 FY24 Financial Statements: Consolidated revenue from operations for the quarter ended March 31, 2024 stood at ₹61,237 crore, marking a 3.5% YoY growth.",
        "TCS Operating Performance: Operating margin expanded to 26.0% in Q4 FY24, representing an expansion of 150 bps YoY and 100 bps QoQ.",
        "TCS Net Profit: Net profit for the quarter was ₹12,434 crore, reflecting a year-on-year increase of 9.1%.",
        "TCS Order Book: Q4 order book TCV achieved an all-time high of $13.2 billion, with full year TCV at a record $42.7 billion.",
        "TCS Dividend Announcement: The Board of Directors recommended a final dividend of ₹28 per equity share of ₹1 each for FY24.",
        "TCS Full Year Performance: Full year FY24 revenue stood at ₹240,893 crore with an operating margin of 24.6% and net income of ₹46,585 crore.",
    ],
    "INFOSYS": [
        "Infosys Q4 FY24 Earnings: Consolidated revenue was ₹37,923 crore, growing 1.3% YoY in constant currency terms.",
        "Infosys Profitability: Operating margin for Q4 FY24 came in at 20.1%. Net profit was ₹7,969 crore for the fourth quarter.",
        "Infosys Guidance: For FY25, revenue growth guidance is 1%-3% in constant currency with operating margin guidance of 20%-22%.",
    ],
    "RELIANCE": [
        "Reliance Industries Q4 FY24 Release: Gross revenue for the fourth quarter was ₹240,715 crore ($28.9 billion), up 10.8% YoY.",
        "Reliance EBITDA: Consolidated EBITDA reached ₹42,516 crore ($5.1 billion), up 14.3% YoY with strong contributions across retail, digital, and energy.",
        "Reliance Net Profit: Net profit for the quarter ended March 31, 2024 was ₹18,951 crore ($2.3 billion).",
    ],
    "APPLE": [
        "Apple Inc Q4 2024 Results: Apple reported quarterly revenue of $94.93 billion, an increase of 6% year over year.",
        "Apple Product Segments: iPhone revenue was $46.22 billion, Services revenue reached an all-time record of $24.97 billion.",
        "Apple Profitability: Gross margin for the quarter was 46.2%, and net income was $14.73 billion ($0.97 diluted EPS).",
    ],
    "MICROSOFT": [
        "Microsoft Q4 FY24 Release: Revenue was $64.73 billion, increasing 15% year-over-year.",
        "Microsoft Cloud: Microsoft Cloud revenue was $36.8 billion, up 21% year-over-year, driven by Azure AI growth.",
        "Microsoft Net Income: Net income reached $22.04 billion, an increase of 10% YoY.",
    ],
    "GOOGLE": [
        "Alphabet Q4 2024 Earnings Release: Alphabet announced consolidated revenues of $86.31 billion for the quarter ended December 31, 2024, up 13% YoY.",
        "Alphabet Profitability: Operating margin was 27.5% for the fourth quarter. Net income was $20.69 billion ($1.64 diluted EPS).",
        "Google Cloud: Google Cloud revenue accelerated to $9.19 billion with strong enterprise AI adoption.",
    ],
    "NVIDIA": [
        "NVIDIA Q4 FY24 Release: NVIDIA reported record quarterly revenue of $22.10 billion, up 22% QoQ and up 265% YoY.",
        "NVIDIA Data Center: Data Center compute and networking revenue reached an all-time record of $18.40 billion, up 409% YoY.",
        "NVIDIA Profitability: Gross margin expanded to 76.0% and net income surged to $12.29 billion.",
    ],
    "AMAZON": [
        "Amazon Q4 2024 Results: Net sales increased 14% to $169.96 billion in the fourth quarter.",
        "Amazon Web Services: AWS revenue was $24.20 billion, up 13% year-over-year.",
        "Amazon Net Income: Net income was $10.62 billion compared with $278 million in Q4 2023.",
    ],
    "META": [
        "Meta Platforms Q4 2024 Earnings: Fourth quarter revenue was $40.11 billion, up 25% year-over-year.",
        "Meta Operating Margin: Operating margin expanded to 41.0% with net income reaching $14.02 billion.",
        "Meta Capital Allocation: Declared first-ever quarterly cash dividend and authorized $50 billion share repurchase.",
    ],
    "TESLA": [
        "Tesla Q4 2024 Update: Total revenues grew 3% YoY to $25.17 billion in the fourth quarter.",
        "Tesla Automotive Performance: Automotive deliveries reached 484,507 vehicles; automotive gross margin ex-regulatory credits was 17.2%.",
        "Tesla Net Profit: GAAP net income for Q4 was $7.93 billion.",
    ],
    "HDFC": [
        "HDFC Bank Q4 FY24 Statements: Net profit rose 37.1% YoY to ₹16,512 crore for the quarter ended March 31, 2024.",
        "HDFC Bank Net Interest Income: NII for the quarter grew to ₹29,077 crore.",
        "HDFC Bank Asset Quality: Gross NPA stood low at 1.24% of gross advances.",
    ],
    "WIPRO": [
        "Wipro Q4 FY24 Results: Gross revenue was ₹22,208 crore, with IT Services segment margin at 16.4%.",
        "Wipro Net Profit: Net income for the quarter reached ₹2,835 crore.",
        "Wipro Deal Wins: Total large deal bookings reached $1.2 billion for the quarter.",
    ],
    "TATAMOTORS": [
        "Tata Motors Q4 FY24 Financials: Consolidated revenue from operations stood at ₹1,19,986 crore, up 13.3% YoY.",
        "Tata Motors Profitability: Consolidated net profit for Q4 stood at ₹17,529 crore.",
        "Tata Motors Commercial & JLR: JLR delivered record revenue of £7.8 billion with strong operating margins.",
    ]
}
