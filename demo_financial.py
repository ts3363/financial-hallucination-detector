"""
demo_financial.py
=================
Demonstration of the Financial Hallucination Detector,
Comparative Peer Engine, Forensics, and AI Financial Research Agent.

Run:
    python demo_financial.py
    python demo_financial.py --scenario 1
    python demo_financial.py --scenario 5
"""

from __future__ import annotations
import sys
import argparse

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from financial import (
    FinancialDetectorConfig,
    FinancialHallucinationDetector,
    FinancialHealer,
    FinancialResearchAgent,
    FinancialReportStore,
    MarketDataProvider,
    ForensicAnalyzer,
    PeerBenchmarkEngine,
    ClaimStatus,
)


def _sep(title: str) -> None:
    width = 66
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def _sub_sep(title: str) -> None:
    width = 66
    print(f"\n{'-' * width}")
    print(f"  {title}")
    print(f"{'-' * width}")


def scenario_1_tcs_revenue():
    """Scenario 1: Confident Revenue Confabulation"""
    _sep("SCENARIO 1: Confident Lie — TCS Q4 Revenue Confabulation")
    print("""
  Problem:
    User: "What was TCS's revenue in Q4?"
    LLM:  "TCS reported ₹65,000 crore in Q4 FY24."
    
  Audit Expected:
    Contradicted — Audited revenue was ₹61,237 crore (+6.14% variance).
    Hallucination Probability: ~94%.
    Self-healer fixes value in-place and attaches filing footnote.
    """)

    question = "What was TCS's revenue in Q4 FY24?"
    answer = (
        "TCS reported ₹65,000 crore in revenue for Q4 FY24, showing accelerated "
        "growth across its cloud and enterprise software verticals."
    )

    detector = FinancialHallucinationDetector()
    healer = FinancialHealer()

    report = detector.inspect(question, answer, company="TCS")
    _sub_sep("DETECTION & AUDIT REPORT")
    print(report.summary())

    _sub_sep("SELF-HEALING LAYER")
    healing_res = healer.heal(report)
    print(healing_res.summary())


def scenario_2_margin_dividend():
    """Scenario 2: Multi-metric Distortion (Margins and Dividends)"""
    _sep("SCENARIO 2: Multi-Metric Distortion — Operating Margin & Dividend")
    print("""
  The model hallucinated both the operating margin (29.5% vs 26.0%)
  and the final dividend (₹35 vs ₹28 per share).
    """)

    question = "What were TCS's operating margins and final dividend in Q4?"
    answer = (
        "For Q4, TCS achieved an operating margin of 29.5% and the board "
        "announced a final dividend of ₹35 per share."
    )

    detector = FinancialHallucinationDetector()
    healer = FinancialHealer()

    report = detector.inspect(question, answer, company="TCS")
    _sub_sep("DETECTION & AUDIT REPORT")
    print(report.summary())

    _sub_sep("SELF-HEALING LAYER")
    healing_res = healer.heal(report)
    print(healing_res.summary())


def scenario_3_apple_global():
    """Scenario 3: Global Tech Company SEC Filing Verification"""
    _sep("SCENARIO 3: Global Tech Filing Audit — Apple Inc. (AAPL)")
    print("""
  Verifying Apple Q4 FY24 revenue and gross margin against SEC Form 10-K filings.
  LLM asserts $105.5 billion revenue and 42.0% gross margin.
  Audited Form 10-K: $94.93 billion and 46.2% gross margin.
    """)

    question = "How did Apple perform in Q4 FY24?"
    answer = (
        "Apple delivered quarterly revenue of $105.5 billion and reported a "
        "gross margin of 42.0% for the fourth quarter."
    )

    detector = FinancialHallucinationDetector()
    healer = FinancialHealer()

    report = detector.inspect(question, answer, company="APPLE")
    _sub_sep("DETECTION & AUDIT REPORT")
    print(report.summary())

    _sub_sep("SELF-HEALING LAYER")
    healing_res = healer.heal(report)
    print(healing_res.summary())


def scenario_4_financial_research_agent():
    """Scenario 4: AI Financial Research Agent (Mini Bloomberg)"""
    _sep("SCENARIO 4: AI Financial Research Agent (Mini Bloomberg)")
    print("""
  User Input: "Analyze TCS before I invest."
  System autonomously gathers:
    Annual reports -> Quarterly results -> Financial ratios -> Forensics -> Peer Benchmarks
  Then synthesizes a Bloomberg-style research report and self-verifies all claims.
    """)

    agent = FinancialResearchAgent()
    report = agent.analyze("Analyze TCS before I invest")
    print(report.formatted_report())


def scenario_5_temporal_basis_flip():
    """Scenario 5: YoY vs QoQ Basis Flip and Period Slip"""
    _sep("SCENARIO 5: Temporal & Basis Conflict (YoY vs QoQ Growth Flip)")
    print("""
  Problem:
    LLM confuses YoY and QoQ growth basis:
    "TCS revenue grew 3.5% QoQ sequentially."
    Audited filing confirms 3.5% was Year-over-Year (YoY); actual QoQ was 1.1%.
    """)

    question = "How fast did TCS revenue grow?"
    answer = "TCS reported that its revenue grew 3.5% QoQ sequentially in the fourth quarter."

    detector = FinancialHallucinationDetector()
    report = detector.inspect(question, answer, company="TCS")
    _sub_sep("TEMPORAL BASIS AUDIT REPORT")
    print(report.summary())


def scenario_6_peer_comparative_contradiction():
    """Scenario 6: Comparative Peer Contradiction"""
    _sep("SCENARIO 6: Industry Peer Benchmark & Comparative Contradiction")
    print("""
  Problem:
    LLM asserts a false competitive ranking:
    "TCS has a lower operating margin than Infosys."
    Audited Truth: TCS operating margin is 26.0%, which is HIGHER than Infosys (20.1%).
    """)

    question = "How do TCS margins compare to Infosys?"
    answer = "TCS has a lower operating margin than Infosys in the recent financial year."

    detector = FinancialHallucinationDetector()
    report = detector.inspect(question, answer, company="TCS")
    _sub_sep("PEER COMPARISON AUDIT REPORT")
    print(report.summary())


def scenario_7_forensic_health():
    """Scenario 7: Institutional Quantitative Forensics (Piotroski & Altman Z-Score)"""
    _sep("SCENARIO 7: Forensic Solvency Health & Bankruptcy Risk Models")
    print("""
  Institutional Quantitative Screening:
    - Piotroski F-Score (0-9): Profitability, capital structure, operating efficiency.
    - Altman Z-Score: Default & credit distress risk index.
    - Cash Conversion & Quality of Earnings.
    """)

    analyzer = ForensicAnalyzer()
    forensic_tcs = analyzer.analyze("TCS")
    print(forensic_tcs.summary())


SCENARIOS = {
    "1": scenario_1_tcs_revenue,
    "2": scenario_2_margin_dividend,
    "3": scenario_3_apple_global,
    "4": scenario_4_financial_research_agent,
    "5": scenario_5_temporal_basis_flip,
    "6": scenario_6_peer_comparative_contradiction,
    "7": scenario_7_forensic_health,
}


def main():
    parser = argparse.ArgumentParser(description="Financial Hallucination Detector Demo")
    parser.add_argument("--scenario", choices=["1", "2", "3", "4", "5", "6", "7", "all"], default="all",
                        help="Run a specific scenario (1-7) or 'all'")
    args = parser.parse_args()

    print("\n" + "=" * 66)
    print("  FINANCIAL HALLUCINATION DETECTOR & AI RESEARCH AGENT DEMO")
    print("=" * 66)

    if args.scenario == "all":
        for s in ("1", "2", "3", "4", "5", "6", "7"):
            SCENARIOS[s]()
    else:
        SCENARIOS[args.scenario]()


if __name__ == "__main__":
    main()
