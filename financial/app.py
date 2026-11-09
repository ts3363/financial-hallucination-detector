"""
financial/app.py
================
Streamlit Web Application for Financial Hallucination Detector,
AI Financial Research Agent, Peer Benchmarking, Forensics, and Document Ingestion.
"""

from __future__ import annotations
import sys
import os

# Ensure repository root is on sys.path
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import streamlit as st
import pandas as pd
from typing import Dict, Any, List

from financial.config import FinancialDetectorConfig, ClaimStatus, FinancialRiskLevel
from financial.detector import FinancialHallucinationDetector
from financial.healer import FinancialHealer
from financial.research_agent import FinancialResearchAgent
from financial.data_sources import FinancialReportStore, MarketDataProvider
from financial.forensics import ForensicAnalyzer
from financial.peer_benchmark import PeerBenchmarkEngine
from financial.rag_indexer import FinancialDocumentIndexer

# Page configuration
st.set_page_config(
    page_title="Financial Hallucination Detector & AI Analyst",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Sidebar Configuration & Presets ──────────────────────────
st.sidebar.title("⚙️ Financial AI Settings")

relative_tolerance = st.sidebar.slider(
    "Numeric Tolerance (%)",
    min_value=0.5,
    max_value=10.0,
    value=2.0,
    step=0.5,
    help="Allowable variance before flagging a number as a contradiction."
)

margin_tolerance = st.sidebar.slider(
    "Margin Tolerance (pts)",
    min_value=0.1,
    max_value=3.0,
    value=0.5,
    step=0.1,
    help="Allowable percentage point variance for margins and ratios."
)

enable_live_market = st.sidebar.checkbox(
    "Enable Live Market Data (yfinance)",
    value=False,
    help="Query live market quotes if online; falls back to cached fundamentals."
)

st.sidebar.markdown("---")
st.sidebar.subheader("📋 Preset Scenarios")
preset = st.sidebar.selectbox(
    "Load Scenario:",
    [
        "1. TCS Q4 Revenue Confabulation (₹65,000 Cr vs ₹61,237 Cr)",
        "2. TCS Margin & Dividend Misstatement",
        "3. Apple Q4 Gross Margin & Revenue Drift",
        "4. Nvidia Q4 Data Center Revenue Hallucination ($25B vs $18.4B)",
        "5. Google Q4 Cloud Revenue & Profit Drift",
        "6. Tesla Automotive Margin Fabrication (25% vs 17.2%)",
        "7. HDFC Bank Net Profit Overstatement (₹22,000 Cr vs ₹16,512 Cr)",
        "8. Temporal Flip: YoY vs QoQ Basis Mix-up",
        "9. Peer Ranking Contradiction (TCS vs Infosys)",
        "10. Clean Grounded Answer (100% Verified)",
    ]
)

# Preset texts
PRESETS = {
    "1. TCS Q4 Revenue Confabulation (₹65,000 Cr vs ₹61,237 Cr)": {
        "company": "TCS",
        "question": "What was TCS's revenue in Q4 FY24?",
        "answer": "TCS reported ₹65,000 crore in revenue for Q4 FY24, showing accelerated growth across its cloud and enterprise software verticals."
    },
    "2. TCS Margin & Dividend Misstatement": {
        "company": "TCS",
        "question": "What were TCS's operating margins and final dividend in Q4?",
        "answer": "For Q4, TCS achieved an operating margin of 29.5% and the board announced a final dividend of ₹35 per share."
    },
    "3. Apple Q4 Gross Margin & Revenue Drift": {
        "company": "APPLE",
        "question": "How did Apple perform in Q4 FY24?",
        "answer": "Apple delivered quarterly revenue of $105.5 billion and reported a gross margin of 42.0% for the quarter."
    },
    "4. Nvidia Q4 Data Center Revenue Hallucination ($25B vs $18.4B)": {
        "company": "NVIDIA",
        "question": "What was Nvidia's revenue and gross margin in Q4 FY24?",
        "answer": "Nvidia delivered record Q4 revenue of $25.0 billion and reported a gross margin of 68.0% for the fourth quarter."
    },
    "5. Google Q4 Cloud Revenue & Profit Drift": {
        "company": "GOOGLE",
        "question": "What were Google's Q4 operating margins and revenue?",
        "answer": "Alphabet reported Q4 revenue of $92.5 billion with an operating margin of 34.0%."
    },
    "6. Tesla Automotive Margin Fabrication (25% vs 17.2%)": {
        "company": "TESLA",
        "question": "What was Tesla's performance in Q4?",
        "answer": "Tesla posted Q4 revenue of $28.5 billion and achieved an operating margin of 15.5%."
    },
    "7. HDFC Bank Net Profit Overstatement (₹22,000 Cr vs ₹16,512 Cr)": {
        "company": "HDFC",
        "question": "What was HDFC Bank's net profit in Q4 FY24?",
        "answer": "HDFC Bank reported quarterly net profit of ₹22,000 crore following the parent merger."
    },
    "8. Temporal Flip: YoY vs QoQ Basis Mix-up": {
        "company": "TCS",
        "question": "How fast did TCS revenue grow in Q4?",
        "answer": "TCS reported that its revenue grew 3.5% QoQ sequentially in the fourth quarter."
    },
    "9. Peer Ranking Contradiction (TCS vs Infosys)": {
        "company": "TCS",
        "question": "How do TCS margins compare to Infosys?",
        "answer": "TCS has a lower operating margin than Infosys in the recent financial year."
    },
    "10. Clean Grounded Answer (100% Verified)": {
        "company": "TCS",
        "question": "Provide verified Q4 results for TCS.",
        "answer": "According to the company filings, TCS posted Q4 FY24 revenue from operations of ₹61,237 crore with an operating margin of 26.0% and net profit of ₹12,434 crore."
    }
}

ALL_COMPANIES = [
    "TCS", "INFOSYS", "RELIANCE", "HDFC", "WIPRO", "TATAMOTORS",
    "APPLE", "MICROSOFT", "GOOGLE", "NVIDIA", "AMAZON", "META", "TESLA"
]

selected_preset = PRESETS[preset]

# Initialize engines
@st.cache_resource
def get_engines(rel_tol: float, m_tol: float, live: bool):
    cfg = FinancialDetectorConfig(
        relative_tolerance=rel_tol / 100.0,
        margin_tolerance_pts=m_tol,
        enable_yfinance_live=live
    )
    store = FinancialReportStore()
    market = MarketDataProvider(enable_live=live)
    detector = FinancialHallucinationDetector(config=cfg, report_store=store, market_data=market)
    healer = FinancialHealer()
    agent = FinancialResearchAgent(report_store=store, market_data=market, detector=detector)
    forensics = ForensicAnalyzer()
    peer_engine = PeerBenchmarkEngine()
    indexer = FinancialDocumentIndexer(report_store=store)
    return detector, healer, agent, forensics, peer_engine, indexer, store

detector, healer, agent, forensics, peer_engine, indexer, store = get_engines(
    relative_tolerance, margin_tolerance, enable_live_market
)

# ── Main Header ───────────────────────────────────────────────
st.title("💰 Financial Hallucination Detector & AI Research Agent")
st.markdown(
    "Catch financial fabrications, verify claims against regulatory filings & market feeds, "
    "audit temporal basis slips & peer comparisons, and generate institutional research reports."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Hallucination Detector & Healer",
    "📊 AI Research Agent",
    "⚖️ Peer Benchmarking",
    "🔬 Quantitative Forensics",
    "📑 Document & PDF Ingestion"
])

# ── TAB 1: FINANCIAL HALLUCINATION DETECTOR ───────────────────
with tab1:
    st.subheader("Detect & Fix LLM Financial Hallucinations")
    st.caption("Verifies quantitative claims, temporal basis (YoY/QoQ), and peer comparisons against audited statements.")

    col_in1, col_in2 = st.columns([1, 3])
    with col_in1:
        default_idx = ALL_COMPANIES.index(selected_preset["company"]) if selected_preset["company"] in ALL_COMPANIES else 0
        company_input = st.selectbox(
            "Target Company",
            ALL_COMPANIES,
            index=default_idx
        )
    with col_in2:
        question_input = st.text_input("User Question", value=selected_preset["question"])

    answer_input = st.text_area(
        "LLM Generated Answer to Audit",
        value=selected_preset["answer"],
        height=110
    )

    if st.button("🚀 Audit Financial Claims", type="primary", use_container_width=True):
        with st.spinner("Extracting claims & cross-verifying with regulatory filings..."):
            report = detector.inspect(
                question=question_input,
                answer=answer_input,
                company=company_input
            )
            healing_result = healer.heal(report)

        st.markdown("---")
        
        # Metric KPI cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            prob = report.hallucination_probability
            delta_color = "inverse" if prob > 0.5 else "normal"
            kpi1.metric("Hallucination Probability", f"{prob:.1%}", delta="- High Risk" if prob > 0.5 else "Safe", delta_color=delta_color)
        with kpi2:
            kpi2.metric("Audit Risk Tier", report.risk_level.value)
        with kpi3:
            kpi3.metric("Quality & Grounding", f"{report.quality_score:.2f} / 1.0")
        with kpi4:
            kpi4.metric("Verification Latency", f"{report.latency_ms:.1f} ms")

        # Claim Breakdown Table
        st.markdown("### 📋 Claim Verification Table")
        if report.verified_claims:
            rows = []
            for vc in report.verified_claims:
                status_icon = "🟢 SUPPORTED" if vc.status == ClaimStatus.SUPPORTED else ("🔴 CONTRADICTED" if vc.status == ClaimStatus.CONTRADICTED else "🟡 UNKNOWN")
                rows.append({
                    "Status": status_icon,
                    "Metric": vc.claim.metric_label,
                    "Period": vc.claim.period or "Latest",
                    "Claimed Value": vc.claimed_value_str,
                    "Audited Truth": vc.evidence_value_str or "Unverifiable",
                    "Variance": f"{vc.discrepancy_pct:+.2f}%" if vc.discrepancy_pct is not None else "N/A",
                    "Filing Source": vc.evidence_source or "None",
                    "Explanation": vc.explanation
                })
            df_claims = pd.DataFrame(rows)
            st.dataframe(df_claims, use_container_width=True)
        else:
            st.info("No specific quantitative metrics detected in the answer.")

        # Self-Healing Section
        st.markdown("### 🩹 Self-Healing: In-Place Correction")
        col_heal1, col_heal2 = st.columns(2)
        with col_heal1:
            st.markdown("**Original LLM Answer:**")
            st.error(report.answer)
        with col_heal2:
            st.markdown("**Audited & Healed Answer:**")
            if healing_result.healing_applied:
                st.success(healing_result.healed_answer)
                st.caption(f"Applied {len(healing_result.patches_made)} factual patches.")
            else:
                st.success(healing_result.original_answer)
                st.caption("No factual contradictions found. Original answer accepted.")

        with st.expander("🔎 View Full Audited Filing Passages & Quotes"):
            for vc in report.verified_claims:
                if vc.evidence_quote:
                    st.markdown(f"**{vc.claim.metric_label} Evidence:**")
                    st.info(f"\"{vc.evidence_quote}\"\n\n*Source: {vc.evidence_source}*")


# ── TAB 2: AI FINANCIAL RESEARCH AGENT ────────────────────────
with tab2:
    st.subheader("AI Financial Research Agent (Mini Bloomberg)")
    st.caption("Autonomous financial analysis backed by audited filings, market multiples, and quantitative forensics.")

    st.markdown("#### ⚡ Quick Company Selector")
    st.caption("Select any major enterprise or click a quick chip below to generate an institutional report:")

    # Quick selection buttons
    c_sub1, c_sub2 = st.columns(2)
    with c_sub1:
        st.markdown("**🇮🇳 Indian Bluechips:**")
        ib1, ib2, ib3, ib4, ib5, ib6 = st.columns(6)
        if ib1.button("TCS", use_container_width=True):
            st.session_state["target_research_company"] = "TCS"
            st.session_state["research_ran"] = True
        if ib2.button("Infosys", use_container_width=True):
            st.session_state["target_research_company"] = "INFOSYS"
            st.session_state["research_ran"] = True
        if ib3.button("Reliance", use_container_width=True):
            st.session_state["target_research_company"] = "RELIANCE"
            st.session_state["research_ran"] = True
        if ib4.button("HDFC Bank", use_container_width=True):
            st.session_state["target_research_company"] = "HDFC"
            st.session_state["research_ran"] = True
        if ib5.button("Wipro", use_container_width=True):
            st.session_state["target_research_company"] = "WIPRO"
            st.session_state["research_ran"] = True
        if ib6.button("Tata Motors", use_container_width=True):
            st.session_state["target_research_company"] = "TATAMOTORS"
            st.session_state["research_ran"] = True

    with c_sub2:
        st.markdown("**🌐 Global Market & Tech Leaders:**")
        gb1, gb2, gb3, gb4, gb5, gb6, gb7 = st.columns(7)
        if gb1.button("Apple", use_container_width=True):
            st.session_state["target_research_company"] = "APPLE"
            st.session_state["research_ran"] = True
        if gb2.button("Microsoft", use_container_width=True):
            st.session_state["target_research_company"] = "MICROSOFT"
            st.session_state["research_ran"] = True
        if gb3.button("Google", use_container_width=True):
            st.session_state["target_research_company"] = "GOOGLE"
            st.session_state["research_ran"] = True
        if gb4.button("Nvidia", use_container_width=True):
            st.session_state["target_research_company"] = "NVIDIA"
            st.session_state["research_ran"] = True
        if gb5.button("Amazon", use_container_width=True):
            st.session_state["target_research_company"] = "AMAZON"
            st.session_state["research_ran"] = True
        if gb6.button("Meta", use_container_width=True):
            st.session_state["target_research_company"] = "META"
            st.session_state["research_ran"] = True
        if gb7.button("Tesla", use_container_width=True):
            st.session_state["target_research_company"] = "TESLA"
            st.session_state["research_ran"] = True

    selected_quick_comp = st.session_state.get("target_research_company", "TCS")
    default_prompt_val = f"Analyze {selected_quick_comp} before I invest"

    col_ag1, col_ag2 = st.columns([3, 1])
    with col_ag1:
        research_query = st.text_input(
            "Ask Any Financial Query (Company valuation, comparisons, metrics, or forensic concepts):",
            value=default_prompt_val,
            key="research_prompt_text",
            help="You can ask literally any financial question, e.g. 'Compare TCS and Infosys margins', 'Is Nvidia overvalued?', 'What was Apple revenue?', 'Explain Altman Z-score'."
        )
    with col_ag2:
        st.write("")
        st.write("")
        run_research = st.button("📈 Run Financial Query / Analysis", type="primary", use_container_width=True)

    with st.expander("💡 Click to view example financial queries you can ask:"):
        st.markdown("""
        - 🏢 **Company Investment Research:** `Analyze TCS before I invest`, `Is Nvidia a good buy right now?`, `Analyze Tesla valuation and risks`
        - ⚖️ **Cross-Company Peer Comparisons:** `Compare TCS and Infosys operating margins`, `Is Apple more profitable than Microsoft?`
        - 📋 **Direct Audited Metric Lookups:** `What was Google's revenue in Q4?`, `What was Apple's gross margin?`, `What was Reliance net profit?`
        - 🔬 **Financial Concepts & Forensics:** `Explain Piotroski F-score and its 9 criteria`, `What is Altman Z-Score safe zone?`, `What is cash conversion ratio?`
        """)

    if run_research or st.session_state.get("research_ran"):
        st.session_state["research_ran"] = True
        with st.spinner("Gathering filings, analyzing ratios, and verifying claims..."):
            res_report = agent.analyze(research_query)

        st.markdown("---")
        st.markdown(f"## 🏢 {res_report.company} — AI Financial Analysis")
        st.caption(f"{res_report.company_name} | Ticker: `{res_report.ticker}` | Grounding Status: **{'✅ 100% Grounded' if res_report.is_fully_grounded else '⚠️ Discrepancy Found'}**")

        if res_report.direct_answer:
            st.markdown("### 💬 Direct Analyst Response")
            st.info(res_report.direct_answer)

        st.markdown("### 🎯 Pillar Scorecards")
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.metric("Financial Health", f"{res_report.financial_health_score}/100")
            st.progress(res_report.financial_health_score / 100.0)
        with p2:
            st.metric("Growth Score", f"{res_report.growth_score}/100")
            st.progress(res_report.growth_score / 100.0)
        with p3:
            st.metric("Profitability Score", f"{res_report.profitability_score}/100")
            st.progress(res_report.profitability_score / 100.0)
        with p4:
            st.metric("Risk (Lower is Safer)", f"{res_report.risk_score}/100")
            st.progress(res_report.risk_score / 100.0)

        st.markdown(f"#### 🏆 Overall Rating: **{res_report.overall_assessment}** (Confidence: {res_report.confidence_pct}%)")

        if res_report.forensic_report:
            st.markdown("### 🔬 Quantitative Forensics & Solvency Health")
            f1, f2, f3 = st.columns(3)
            with f1:
                st.metric("Piotroski F-Score", f"{res_report.forensic_report.piotroski_f_score}/9", res_report.forensic_report.f_score_rating)
            with f2:
                st.metric("Altman Z-Score", f"{res_report.forensic_report.altman_z_score:.2f}", res_report.forensic_report.z_score_zone)
            with f3:
                st.metric("Cash Conversion Ratio", f"{res_report.forensic_report.cash_conversion_ratio:.2f}x", res_report.forensic_report.earnings_quality)

        st.markdown("### 📊 Market Multiples & Ratios")
        m_cols = st.columns(4)
        metrics_list = list(res_report.market_metrics.items())
        for i, (k, v) in enumerate(metrics_list):
            m_cols[i % 4].metric(k, v)

        st.markdown("---")
        c_find, c_ev = st.columns(2)
        with c_find:
            st.markdown("### 💡 Key Findings")
            for item in res_report.key_findings:
                if "✓" in item:
                    st.success(item)
                else:
                    st.warning(item)

        with c_ev:
            st.markdown("### 📰 Recent Events & Earnings Commentary")
            for ev in res_report.recent_events:
                st.markdown(f"• {ev}")

        st.markdown("### 📑 Verified Evidence & Citations")
        for cite in res_report.evidence_citations:
            st.markdown(f"- 📎 `[{cite}]`")


# ── TAB 3: PEER BENCHMARKING ──────────────────────────────────
with tab3:
    st.subheader("⚖️ Industry Peer Benchmarking & Relative Claims")
    st.caption("Compares companies within industry sectors and audits relative performance claims.")

    peer_comp_target = st.selectbox("Select Target Company for Peer Review", ALL_COMPANIES)
    peers_list = peer_engine.get_peer_comparison(peer_comp_target)

    p_rows = []
    for p in peers_list:
        p_rows.append({
            "Company": p.company,
            "Ticker": p.ticker,
            "Sector": p.sector,
            "Operating Margin (%)": f"{p.operating_margin:.1f}%",
            "Net Margin (%)": f"{p.net_margin:.1f}%",
            "P/E Multiple": f"{p.pe_ratio:.1f}x",
            "Return on Equity (%)": f"{p.roe:.1f}%",
            "Dividend Yield (%)": f"{p.dividend_yield:.2f}%",
            "Debt/Equity": f"{p.debt_to_equity:.2f}",
            "Market Cap": p.market_cap_str
        })
    st.dataframe(pd.DataFrame(p_rows), use_container_width=True)

    st.markdown("### 🔎 Audit Comparative Statements")
    comp_claim = st.text_input(
        "Enter comparative claim to verify",
        value=f"{peer_comp_target} has a lower operating margin than Infosys" if peer_comp_target == "TCS" else "Apple has a higher P/E multiple than Microsoft"
    )
    if st.button("Check Comparative Claim"):
        comp_res = peer_engine.audit_comparative_claim(comp_claim)
        if comp_res:
            if comp_res.status == ClaimStatus.SUPPORTED:
                st.success(f"✅ SUPPORTED: {comp_res.explanation}")
            else:
                st.error(f"❌ CONTRADICTED: {comp_res.explanation}")
        else:
            st.warning("Could not identify at least two recognized peer companies or a recognized metric in the statement.")


# ── TAB 4: QUANTITATIVE FORENSICS ─────────────────────────────
with tab4:
    st.subheader("🔬 Institutional Forensic Health & Insolvency Risk")
    st.caption("Piotroski 9-Point F-Score and Altman Z-Score credit distress models.")

    forensic_target = st.selectbox("Company to Audit", ALL_COMPANIES, key="forensic_comp")
    forensic_data = forensics.analyze(forensic_target)

    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        st.metric("Piotroski F-Score", f"{forensic_data.piotroski_f_score}/9")
        st.write(f"**Rating:** {forensic_data.f_score_rating}")
    with fcol2:
        st.metric("Altman Z-Score", f"{forensic_data.altman_z_score:.2f}")
        st.write(f"**Zone:** {forensic_data.z_score_zone}")
    with fcol3:
        st.metric("Cash Conversion Ratio", f"{forensic_data.cash_conversion_ratio:.2f}x")
        st.write(f"**Quality:** {forensic_data.earnings_quality}")

    st.markdown("#### 📋 Piotroski Criteria Met:")
    for c in forensic_data.f_score_breakdown:
        st.markdown(f"- ✅ {c}")


# ── TAB 5: DOCUMENT & PDF INGESTION ───────────────────────────
with tab5:
    st.subheader("📑 Document & PDF Regulatory Ingestion Engine")
    st.caption("Upload annual reports or quarterly filings to expand the verified knowledge base.")

    ingest_company = st.text_input("Entity / Ticker Symbol", value="TCS")
    ingest_quarter = st.selectbox("Fiscal Period", ["Q1", "Q2", "Q3", "Q4", "FY24", "FY25"])
    
    doc_text = st.text_area(
        "Paste Financial Press Release / Statement Excerpt",
        value="Revenue from operations for the quarter stood at 61237 cr. Net profit was 12434 cr. Operating margin expanded to 26.0%.",
        height=120
    )

    if st.button("📥 Parse & Index Statement Rows"):
        new_records = indexer.index_text_document(
            company=ingest_company,
            document_title=f"{ingest_company} {ingest_quarter} Uploaded Release",
            text_content=doc_text,
            period=ingest_quarter
        )
        if new_records:
            st.success(f"Successfully extracted and indexed {len(new_records)} audited metric ground-truths into the store!")
            rows = [{"Metric": r.metric, "Value": r.reported_str, "Period": r.period, "Filing Quote": r.exact_filing_quote} for r in new_records]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("Text indexed into context chunks. No exact tabular statement line patterns matched.")
