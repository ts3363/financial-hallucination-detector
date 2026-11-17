"""Opt-in integration test for current Yahoo Finance market data.

Run with RUN_LIVE_FINANCIAL_DATA_TESTS=1 to make a real network request.
"""

from __future__ import annotations

import os

import pytest

from financial import (
    ClaimStatus,
    FinancialDetectorConfig,
    FinancialHallucinationDetector,
    MarketDataProvider,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_FINANCIAL_DATA_TESTS") != "1",
    reason="Set RUN_LIVE_FINANCIAL_DATA_TESTS=1 to enable live market-data tests.",
)


def test_live_quote_is_used_as_financial_evidence():
    market_data = MarketDataProvider(enable_live=True)
    quote = market_data.get_quote("APPLE")

    assert quote.is_current
    assert quote.source.startswith("Yahoo Finance")
    assert quote.current_price > 0
    assert quote.pe_ratio > 0

    detector = FinancialHallucinationDetector(
        config=FinancialDetectorConfig(enable_yfinance_live=True),
        market_data=market_data,
    )
    report = detector.inspect(
        question="What is Apple's current P/E ratio?",
        answer=f"Apple's P/E ratio is {quote.pe_ratio:.1f}.",
        company="APPLE",
    )

    claim = next(vc for vc in report.verified_claims if vc.claim.metric == "pe_ratio")
    assert claim.status == ClaimStatus.SUPPORTED
    assert "Yahoo Finance" in claim.evidence_source
