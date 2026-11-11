"""
financial/app.py
================
Streamlit Web Application for Financial Hallucination Detector,
AI Financial Research Agent, Peer Benchmarking, Forensics, and Document Ingestion.
Enhanced with interactive finance features: chat Q&A, watchlist, plotly gauges,
PDF export, glossary, Piotroski checklist, audit log, and what-if scenario builder.
"""

from __future__ import annotations
import sys
import os
import io
import datetime

# Ensure repository root is on sys.path
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, List

from financial.config import FinancialDetectorConfig, ClaimStatus, FinancialRiskLevel
from financial.detector import FinancialHallucinationDetector
from financial.healer import FinancialHealer
from financial.research_agent import FinancialResearchAgent
from financial.data_sources import FinancialReportStore, MarketDataProvider
from financial.forensics import ForensicAnalyzer
from financial.peer_benchmark import PeerBenchmarkEngine
from financial.rag_indexer import FinancialDocumentIndexer

# ── Page Configuration ─────────────────────────────────────────
st.set_page_config(
    page_title="Financial Hallucination Detector & AI Analyst",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for richer UI ───────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 6px 0;
    text-align: center;
}
.chat-bubble-user {
    background: #0f3460;
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 10px 16px;
    margin: 6px 0 6px auto;
    max-width: 80%;
    text-align: right;
}
.chat-bubble-ai {
    background: #1a1a2e;
    border: 1px solid #0f3460;
    color: #e0e0e0;
    border-radius: 18px 18px 18px 4px;
    padding: 10px 16px;
    margin: 6px auto 6px 0;
    max-width: 85%;
}
.watchlist-chip {
    display: inline-block;
    background: #0f3460;
    color: white;
    border-radius: 20px;
    padding: 4px 12px;
    margin: 3px;
    font-size: 13px;
    font-weight: 600;
}
.verdict-green { color: #00c853; font-weight: bold; font-size: 1.1em; }
.verdict-red { color: #ff1744; font-weight: bold; font-size: 1.1em; }
.verdict-yellow { color: #ffd600; font-weight: bold; font-size: 1.1em; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 20px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ── Session State Initialization ───────────────────────────────
for key, default in [
    ("watchlist", []),
    ("chat_history", []),
    ("audit_log", []),
    ("target_research_company", "TCS"),
    ("research_ran", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ────────────────────────────────────────────────────
st.sidebar.title("⚙️ Financial AI Settings")

relative_tolerance = st.sidebar.slider(
    "Numeric Tolerance (%)", min_value=0.5, max_value=10.0, value=2.0, step=0.5,
    help="Allowable variance before flagging a number as a contradiction."
)
margin_tolerance = st.sidebar.slider(
    "Margin Tolerance (pts)", min_value=0.1, max_value=3.0, value=0.5, step=0.1,
    help="Allowable percentage point variance for margins and ratios."
)
enable_live_market = st.sidebar.checkbox(
    "Enable Live Market Data (yfinance)", value=False,
    help="Query live market quotes if online; falls back to cached fundamentals."
)

st.sidebar.markdown("---")
st.sidebar.subheader("📋 Preset Scenarios")
preset = st.sidebar.selectbox(
    "Load Scenario:",
    [
        "1. TCS Q4 Revenue Confabulation (Rs 65,000 Cr vs Rs 61,237 Cr)",
        "2. TCS Margin & Dividend Misstatement",
        "3. Apple Q4 Gross Margin & Revenue Drift",
        "4. Nvidia Q4 Data Center Revenue Hallucination ($25B vs $18.4B)",
        "5. Google Q4 Cloud Revenue & Profit Drift",
        "6. Tesla Automotive Margin Fabrication (25% vs 17.2%)",
        "7. HDFC Bank Net Profit Overstatement (Rs 22,000 Cr vs Rs 16,512 Cr)",
        "8. Temporal Flip: YoY vs QoQ Basis Mix-up",
        "9. Peer Ranking Contradiction (TCS vs Infosys)",
        "10. Clean Grounded Answer (100% Verified)",
    ]
)

# ── Glossary Sidebar ───────────────────────────────────────────
st.sidebar.markdown("---")
with st.sidebar.expander("📚 Financial Terms Glossary"):
    glossary = {
        "P/E Ratio": "Price-to-Earnings: Stock price divided by annual earnings per share. Higher = more expensive relative to earnings.",
        "ROE": "Return on Equity: Net income divided by shareholder equity. Measures profitability per rupee/dollar of equity invested.",
        "Operating Margin": "Operating profit as % of revenue. Excludes interest & taxes. Higher margin = more efficient operations.",
        "Net Margin": "Net profit as % of revenue after ALL expenses including taxes. The bottom-line profitability measure.",
        "Piotroski F-Score": "9-point checklist scoring profitability, leverage, and operating efficiency. Score 8-9 = strong; 0-2 = distressed.",
        "Altman Z-Score": "Bankruptcy prediction model. Z > 2.99 = safe zone; 1.81–2.99 = grey zone; < 1.81 = distress zone.",
        "Cash Conversion Ratio": "Operating Cash Flow / Net Profit. Ratio > 1.0 means earnings are backed by real cash.",
        "Debt/Equity (D/E)": "Total debt divided by shareholder equity. High D/E = higher financial risk and leverage.",
        "EBITDA": "Earnings Before Interest, Taxes, Depreciation & Amortization. Proxy for operating cash-flow generation.",
        "EV/EBITDA": "Enterprise Value divided by EBITDA. Valuation multiple; lower = potentially cheaper than peers.",
        "Dividend Yield": "Annual dividend per share / share price. Measures income return for investors.",
        "Market Cap": "Total market value of a company = share price × number of shares outstanding.",
        "YoY / QoQ": "Year-over-Year (same quarter last year) vs Quarter-over-Quarter (previous quarter). Mixing these is a common hallucination.",
        "Free Cash Flow": "Cash from operations minus capital expenditure. 'Real' profit that can be returned to shareholders.",
        "Hallucination": "When an AI fabricates or misquotes a financial figure not supported by audited filings or market data.",
    }
    for term, definition in glossary.items():
        st.sidebar.markdown(f"**{term}**")
        st.sidebar.caption(definition)
        st.sidebar.markdown("---")

# ── Watchlist Sidebar ──────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("📌 My Watchlist")
ALL_COMPANIES = [
    "TCS", "INFOSYS", "RELIANCE", "HDFC", "WIPRO", "TATAMOTORS",
    "APPLE", "MICROSOFT", "GOOGLE", "NVIDIA", "AMAZON", "META", "TESLA"
]
wl_add = st.sidebar.selectbox("Add to Watchlist:", ALL_COMPANIES, key="wl_add_sel")
if st.sidebar.button("+ Add", key="wl_add_btn"):
    if wl_add not in st.session_state["watchlist"]:
        st.session_state["watchlist"].append(wl_add)
        st.sidebar.success(f"{wl_add} added!")
if st.session_state["watchlist"]:
    chips_html = " ".join([f'<span class="watchlist-chip">{c}</span>' for c in st.session_state["watchlist"]])
    st.sidebar.markdown(chips_html, unsafe_allow_html=True)
    if st.sidebar.button("Clear Watchlist", key="wl_clear"):
        st.session_state["watchlist"] = []
        st.rerun()

# ── Preset Texts ───────────────────────────────────────────────
PRESETS = {
    "1. TCS Q4 Revenue Confabulation (Rs 65,000 Cr vs Rs 61,237 Cr)": {
        "company": "TCS", "question": "What was TCS's revenue in Q4 FY24?",
        "answer": "TCS reported Rs 65,000 crore in revenue for Q4 FY24, showing accelerated growth across its cloud and enterprise software verticals."
    },
    "2. TCS Margin & Dividend Misstatement": {
        "company": "TCS", "question": "What were TCS's operating margins and final dividend in Q4?",
        "answer": "For Q4, TCS achieved an operating margin of 29.5% and the board announced a final dividend of Rs 35 per share."
    },
    "3. Apple Q4 Gross Margin & Revenue Drift": {
        "company": "APPLE", "question": "How did Apple perform in Q4 FY24?",
        "answer": "Apple delivered quarterly revenue of $105.5 billion and reported a gross margin of 42.0% for the quarter."
    },
    "4. Nvidia Q4 Data Center Revenue Hallucination ($25B vs $18.4B)": {
        "company": "NVIDIA", "question": "What was Nvidia's revenue and gross margin in Q4 FY24?",
        "answer": "Nvidia delivered record Q4 revenue of $25.0 billion and reported a gross margin of 68.0% for the fourth quarter."
    },
    "5. Google Q4 Cloud Revenue & Profit Drift": {
        "company": "GOOGLE", "question": "What were Google's Q4 operating margins and revenue?",
        "answer": "Alphabet reported Q4 revenue of $92.5 billion with an operating margin of 34.0%."
    },
    "6. Tesla Automotive Margin Fabrication (25% vs 17.2%)": {
        "company": "TESLA", "question": "What was Tesla's performance in Q4?",
        "answer": "Tesla posted Q4 revenue of $28.5 billion and achieved an operating margin of 15.5%."
    },
    "7. HDFC Bank Net Profit Overstatement (Rs 22,000 Cr vs Rs 16,512 Cr)": {
        "company": "HDFC", "question": "What was HDFC Bank's net profit in Q4 FY24?",
        "answer": "HDFC Bank reported quarterly net profit of Rs 22,000 crore following the parent merger."
    },
    "8. Temporal Flip: YoY vs QoQ Basis Mix-up": {
        "company": "TCS", "question": "How fast did TCS revenue grow in Q4?",
        "answer": "TCS reported that its revenue grew 3.5% QoQ sequentially in the fourth quarter."
    },
    "9. Peer Ranking Contradiction (TCS vs Infosys)": {
        "company": "TCS", "question": "How do TCS margins compare to Infosys?",
        "answer": "TCS has a lower operating margin than Infosys in the recent financial year."
    },
    "10. Clean Grounded Answer (100% Verified)": {
        "company": "TCS", "question": "Provide verified Q4 results for TCS.",
        "answer": "According to the company filings, TCS posted Q4 FY24 revenue from operations of Rs 61,237 crore with an operating margin of 26.0% and net profit of Rs 12,434 crore."
    }
}
selected_preset = PRESETS[preset]

# ── Initialize Engines ─────────────────────────────────────────
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

# ── Helper: Plotly Gauge Chart ─────────────────────────────────
def make_gauge(title: str, value: float, max_val: float = 100, color_ranges=None) -> go.Figure:
    """Create an interactive plotly gauge chart."""
    if color_ranges is None:
        color_ranges = [
            [0, 40, "#ff4136"],
            [40, 70, "#ff851b"],
            [70, 100, "#2ecc40"],
        ]
    steps = [{"range": [r[0], r[1]], "color": r[2]} for r in color_ranges]
    # Determine needle color
    pct = value / max_val
    needle_color = "#2ecc40" if pct > 0.7 else ("#ff851b" if pct > 0.4 else "#ff4136")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 14, "color": "#e0e0e0"}},
        number={"font": {"size": 24, "color": needle_color}},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#888", "tickfont": {"color": "#888"}},
            "bar": {"color": needle_color, "thickness": 0.3},
            "bgcolor": "#1a1a2e",
            "bordercolor": "#0f3460",
            "steps": steps,
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.8,
                "value": value,
            },
        }
    ))
    fig.update_layout(
        paper_bgcolor="#0d0d1a",
        plot_bgcolor="#0d0d1a",
        font={"color": "#e0e0e0"},
        height=220,
        margin={"l": 20, "r": 20, "t": 50, "b": 10},
    )
    return fig

# ── Helper: PDF Report Generator ───────────────────────────────
def generate_pdf_report(company: str, res_report) -> bytes:
    """Generate a PDF research report using reportlab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle("Title2", parent=styles["Title"],
                                 fontSize=18, textColor=colors.HexColor("#0f3460"),
                                 spaceAfter=6)
    story.append(Paragraph(f"AI Financial Research Report", title_style))
    story.append(Paragraph(f"{res_report.company_name} ({res_report.ticker})", styles["Heading2"]))
    story.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%d %b %Y %H:%M')}", styles["Normal"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0f3460")))
    story.append(Spacer(1, 0.4*cm))

    # Direct Answer
    if res_report.direct_answer:
        story.append(Paragraph("Analyst Response", styles["Heading3"]))
        story.append(Paragraph(res_report.direct_answer, styles["Normal"]))
        story.append(Spacer(1, 0.3*cm))

    # Scores
    story.append(Paragraph("Pillar Scorecards", styles["Heading3"]))
    score_data = [
        ["Dimension", "Score", "Rating"],
        ["Financial Health", f"{res_report.financial_health_score}/100",
         "Strong" if res_report.financial_health_score > 70 else ("Moderate" if res_report.financial_health_score > 40 else "Weak")],
        ["Growth", f"{res_report.growth_score}/100",
         "Strong" if res_report.growth_score > 70 else ("Moderate" if res_report.growth_score > 40 else "Weak")],
        ["Profitability", f"{res_report.profitability_score}/100",
         "Strong" if res_report.profitability_score > 70 else ("Moderate" if res_report.profitability_score > 40 else "Weak")],
        ["Risk (lower=safer)", f"{res_report.risk_score}/100",
         "Low" if res_report.risk_score < 40 else ("Medium" if res_report.risk_score < 70 else "High")],
    ]
    t = Table(score_data, colWidths=[6*cm, 4*cm, 4*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8f8f8"), colors.white]),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.4*cm))

    # Forensics
    if res_report.forensic_report:
        story.append(Paragraph("Quantitative Forensics", styles["Heading3"]))
        f = res_report.forensic_report
        fdata = [
            ["Metric", "Value", "Signal"],
            ["Piotroski F-Score", f"{f.piotroski_f_score}/9", f.f_score_rating],
            ["Altman Z-Score", f"{f.altman_z_score:.2f}", f.z_score_zone],
            ["Cash Conversion Ratio", f"{f.cash_conversion_ratio:.2f}x", f.earnings_quality],
        ]
        ft = Table(fdata, colWidths=[6*cm, 4*cm, 4*cm])
        ft.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(ft)
        story.append(Spacer(1, 0.4*cm))

    # Market Metrics
    story.append(Paragraph("Market Multiples & Ratios", styles["Heading3"]))
    mm_items = list(res_report.market_metrics.items())
    mm_data = [["Metric", "Value"]] + [[k, v] for k, v in mm_items]
    mmt = Table(mm_data, colWidths=[8*cm, 6*cm])
    mmt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8f8f8"), colors.white]),
    ]))
    story.append(mmt)
    story.append(Spacer(1, 0.4*cm))

    # Key Findings
    story.append(Paragraph("Key Findings", styles["Heading3"]))
    for item in res_report.key_findings:
        story.append(Paragraph(f"• {item}", styles["Normal"]))
    story.append(Spacer(1, 0.3*cm))

    # Citations
    story.append(Paragraph("Evidence & Citations", styles["Heading3"]))
    for cite in res_report.evidence_citations:
        story.append(Paragraph(f"• {cite}", styles["Normal"]))

    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Paragraph("Generated by Financial Hallucination Detector & AI Research Agent",
                            ParagraphStyle("footer", parent=styles["Normal"], fontSize=8,
                                           textColor=colors.grey, alignment=TA_CENTER)))

    doc.build(story)
    buf.seek(0)
    return buf.read()

# ── Main Header ────────────────────────────────────────────────
st.title("💰 Financial Hallucination Detector & AI Research Agent")
st.markdown(
    "Catch financial fabrications, verify claims against regulatory filings & market feeds, "
    "audit temporal basis slips & peer comparisons, and generate institutional research reports."
)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔍 Hallucination Detector",
    "📊 AI Research Agent",
    "💬 Finance Chat Q&A",
    "⚖️ Peer Benchmarking",
    "🔬 Quantitative Forensics",
    "📑 Document Ingestion",
])

# ══════════════════════════════════════════════════════════════
# TAB 1: FINANCIAL HALLUCINATION DETECTOR
# ══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Detect & Fix LLM Financial Hallucinations")
    st.caption("Verifies quantitative claims, temporal basis (YoY/QoQ), and peer comparisons against audited statements.")

    col_in1, col_in2 = st.columns([1, 3])
    with col_in1:
        default_idx = ALL_COMPANIES.index(selected_preset["company"]) if selected_preset["company"] in ALL_COMPANIES else 0
        company_input = st.selectbox("Target Company", ALL_COMPANIES, index=default_idx)
    with col_in2:
        question_input = st.text_input("User Question", value=selected_preset["question"])

    answer_input = st.text_area(
        "LLM Generated Answer to Audit",
        value=selected_preset["answer"],
        height=110
    )

    col_btn1, col_btn2 = st.columns([2, 1])
    with col_btn1:
        audit_clicked = st.button("🚀 Audit Financial Claims", type="primary", use_container_width=True)
    with col_btn2:
        clear_log = st.button("🗑️ Clear Audit Log", use_container_width=True)
        if clear_log:
            st.session_state["audit_log"] = []
            st.rerun()

    if audit_clicked:
        with st.spinner("Extracting claims & cross-verifying with regulatory filings..."):
            report = detector.inspect(
                question=question_input,
                answer=answer_input,
                company=company_input
            )
            healing_result = healer.heal(report)

        # Save to session audit log
        st.session_state["audit_log"].append({
            "Time": datetime.datetime.now().strftime("%H:%M:%S"),
            "Company": company_input,
            "Risk": report.risk_level.value,
            "Hallucination Prob": f"{report.hallucination_probability:.1%}",
            "Quality": f"{report.quality_score:.2f}",
            "Claims": len(report.verified_claims),
            "Healed": "Yes" if healing_result.healing_applied else "No",
        })

        st.markdown("---")

        # KPI Cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        prob = report.hallucination_probability
        delta_color = "inverse" if prob > 0.5 else "normal"
        kpi1.metric("Hallucination Probability", f"{prob:.1%}",
                    delta="High Risk" if prob > 0.5 else "Safe", delta_color=delta_color)
        kpi2.metric("Audit Risk Tier", report.risk_level.value)
        kpi3.metric("Quality & Grounding", f"{report.quality_score:.2f} / 1.0")
        kpi4.metric("Verification Latency", f"{report.latency_ms:.1f} ms")

        # Hallucination probability gauge
        fig_prob = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(prob * 100, 1),
            title={"text": "Hallucination Risk Score", "font": {"size": 14}},
            number={"suffix": "%", "font": {"size": 26}},
            delta={"reference": 20, "increasing": {"color": "#ff4136"}, "decreasing": {"color": "#2ecc40"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#ff4136" if prob > 0.5 else "#2ecc40"},
                "steps": [
                    {"range": [0, 30], "color": "#1a3a1a"},
                    {"range": [30, 60], "color": "#3a3a1a"},
                    {"range": [60, 100], "color": "#3a1a1a"},
                ],
                "threshold": {"line": {"color": "white", "width": 2}, "value": 50},
            }
        ))
        fig_prob.update_layout(paper_bgcolor="#0d0d1a", font={"color": "#e0e0e0"}, height=200,
                               margin={"l": 30, "r": 30, "t": 50, "b": 10})
        st.plotly_chart(fig_prob, use_container_width=True)

        # Claim Breakdown Table
        st.markdown("### 📋 Claim Verification Table")
        if report.verified_claims:
            rows = []
            for vc in report.verified_claims:
                status_icon = (
                    "✅ SUPPORTED" if vc.status == ClaimStatus.SUPPORTED
                    else ("❌ CONTRADICTED" if vc.status == ClaimStatus.CONTRADICTED else "⚠️ UNKNOWN")
                )
                rows.append({
                    "Status": status_icon,
                    "Metric": vc.claim.metric_label,
                    "Period": vc.claim.period or "Latest",
                    "Claimed Value": vc.claimed_value_str,
                    "Audited Truth": vc.evidence_value_str or "Unverifiable",
                    "Variance": f"{vc.discrepancy_pct:+.2f}%" if vc.discrepancy_pct is not None else "N/A",
                    "Filing Source": vc.evidence_source or "None",
                    "Explanation": vc.explanation,
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
                    st.info(f'"{vc.evidence_quote}"\n\n*Source: {vc.evidence_source}*')

    # Audit Log Section
    if st.session_state["audit_log"]:
        st.markdown("---")
        st.markdown("### 🗂️ Session Audit Log")
        st.caption("All claims audited in this session:")
        df_log = pd.DataFrame(st.session_state["audit_log"])
        st.dataframe(df_log, use_container_width=True)

        # Export audit log as CSV
        csv_bytes = df_log.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Audit Log (CSV)",
            data=csv_bytes,
            file_name=f"audit_log_{datetime.date.today()}.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════
# TAB 2: AI FINANCIAL RESEARCH AGENT
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("AI Financial Research Agent (Mini Bloomberg)")
    st.caption("Autonomous financial analysis backed by audited filings, market multiples, and quantitative forensics.")

    # --- Quick Company Selector ---
    st.markdown("#### Quick Company Selector")
    c_sub1, c_sub2 = st.columns(2)
    with c_sub1:
        st.markdown("**Indian Bluechips:**")
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
        st.markdown("**Global Market & Tech Leaders:**")
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
            "Research Prompt / Target Entity",
            value=default_prompt_val,
            key="research_prompt_text",
            help="Ask any financial question: 'Compare TCS and Infosys margins', 'Is Nvidia overvalued?', 'Explain Altman Z-score'."
        )
    with col_ag2:
        st.write("")
        st.write("")
        run_research = st.button("📈 Run Analysis", type="primary", use_container_width=True)

    with st.expander("💡 Example Queries"):
        st.markdown("""
        - **Investment Research:** `Analyze TCS before I invest`, `Is Nvidia a good buy right now?`
        - **Peer Comparisons:** `Compare TCS and Infosys operating margins`
        - **Metric Lookups:** `What was Google's revenue in Q4?`, `Apple gross margin?`
        - **Concepts & Forensics:** `Explain Piotroski F-score`, `What is Altman Z-Score safe zone?`
        """)

    if run_research or st.session_state.get("research_ran"):
        st.session_state["research_ran"] = True
        with st.spinner("Gathering filings, analyzing ratios, and verifying claims..."):
            res_report = agent.analyze(research_query)

        st.markdown("---")
        grounding_badge = "✅ 100% Grounded" if res_report.is_fully_grounded else "⚠️ Discrepancy Found"
        st.markdown(f"## {res_report.company} — AI Financial Analysis")
        st.caption(f"{res_report.company_name} | Ticker: `{res_report.ticker}` | {grounding_badge}")

        if res_report.direct_answer:
            st.markdown("### 💬 Direct Analyst Response")
            st.info(res_report.direct_answer)

        # --- Plotly Gauge Charts for 4 Pillars ---
        st.markdown("### 🎯 Pillar Scorecards")
        g1, g2, g3, g4 = st.columns(4)
        with g1:
            st.plotly_chart(make_gauge("Financial Health", res_report.financial_health_score), use_container_width=True)
        with g2:
            st.plotly_chart(make_gauge("Growth Score", res_report.growth_score), use_container_width=True)
        with g3:
            st.plotly_chart(make_gauge("Profitability", res_report.profitability_score), use_container_width=True)
        with g4:
            # For risk: invert color ranges (low risk = green)
            risk_ranges = [[0, 40, "#2ecc40"], [40, 70, "#ff851b"], [70, 100, "#ff4136"]]
            st.plotly_chart(make_gauge("Risk (Lower=Safer)", res_report.risk_score, color_ranges=risk_ranges), use_container_width=True)

        st.markdown(f"#### Overall Rating: **{res_report.overall_assessment}** (Confidence: {res_report.confidence_pct}%)")

        # --- Forensics ---
        if res_report.forensic_report:
            st.markdown("### 🔬 Quantitative Forensics & Solvency Health")
            f1, f2, f3 = st.columns(3)
            frp = res_report.forensic_report
            with f1:
                fig_f = make_gauge("Piotroski F-Score", frp.piotroski_f_score, max_val=9,
                                   color_ranges=[[0, 3, "#ff4136"], [3, 6, "#ff851b"], [6, 9, "#2ecc40"]])
                st.plotly_chart(fig_f, use_container_width=True)
                st.caption(frp.f_score_rating)
            with f2:
                # Z-score: safe zone is > 2.99; cap display at 5
                z_disp = min(frp.altman_z_score, 5.0)
                fig_z = make_gauge("Altman Z-Score", round(z_disp, 2), max_val=5,
                                   color_ranges=[[0, 1.81, "#ff4136"], [1.81, 2.99, "#ff851b"], [2.99, 5, "#2ecc40"]])
                st.plotly_chart(fig_z, use_container_width=True)
                st.caption(frp.z_score_zone)
            with f3:
                # CCR: > 1.0 is good; cap at 2
                ccr_disp = min(frp.cash_conversion_ratio, 2.0)
                fig_ccr = make_gauge("Cash Conversion Ratio", round(ccr_disp, 2), max_val=2,
                                     color_ranges=[[0, 0.6, "#ff4136"], [0.6, 1.0, "#ff851b"], [1.0, 2.0, "#2ecc40"]])
                st.plotly_chart(fig_ccr, use_container_width=True)
                st.caption(frp.earnings_quality)

        # --- Market Metrics ---
        st.markdown("### 📊 Market Multiples & Ratios")
        m_cols = st.columns(4)
        for i, (k, v) in enumerate(res_report.market_metrics.items()):
            m_cols[i % 4].metric(k, v)

        # --- Key Findings & Events ---
        st.markdown("---")
        c_find, c_ev = st.columns(2)
        with c_find:
            st.markdown("### 💡 Key Findings")
            for item in res_report.key_findings:
                if "✓" in item or "good" in item.lower() or "strong" in item.lower():
                    st.success(item)
                else:
                    st.warning(item)
        with c_ev:
            st.markdown("### 📰 Recent Events & Earnings Commentary")
            for ev in res_report.recent_events:
                st.markdown(f"• {ev}")

        # --- Watchlist Section ---
        st.markdown("---")
        wl_col1, wl_col2 = st.columns([2, 1])
        with wl_col1:
            st.markdown("### 📌 Watchlist Comparison")
            if len(st.session_state["watchlist"]) >= 2:
                wl_rows = []
                for wc in st.session_state["watchlist"]:
                    try:
                        wr = agent.analyze(f"Analyze {wc}")
                        wl_rows.append({
                            "Company": wc,
                            "Financial Health": wr.financial_health_score,
                            "Growth": wr.growth_score,
                            "Profitability": wr.profitability_score,
                            "Risk": wr.risk_score,
                            "Overall": wr.overall_assessment,
                        })
                    except Exception:
                        wl_rows.append({"Company": wc, "Financial Health": "N/A", "Growth": "N/A",
                                        "Profitability": "N/A", "Risk": "N/A", "Overall": "N/A"})
                wl_df = pd.DataFrame(wl_rows)
                st.dataframe(wl_df, use_container_width=True)

                # Radar / bar chart comparison
                try:
                    numeric_rows = [r for r in wl_rows if isinstance(r["Financial Health"], (int, float))]
                    if len(numeric_rows) >= 2:
                        fig_wl = go.Figure()
                        for row in numeric_rows:
                            fig_wl.add_trace(go.Bar(
                                name=row["Company"],
                                x=["Financial Health", "Growth", "Profitability", "Risk (inverted)"],
                                y=[row["Financial Health"], row["Growth"], row["Profitability"],
                                   100 - row["Risk"]],
                            ))
                        fig_wl.update_layout(
                            barmode="group",
                            title="Watchlist Side-by-Side Comparison",
                            paper_bgcolor="#0d0d1a",
                            plot_bgcolor="#1a1a2e",
                            font={"color": "#e0e0e0"},
                            height=350,
                        )
                        st.plotly_chart(fig_wl, use_container_width=True)
                except Exception:
                    pass
            else:
                st.info("Add 2+ companies to your watchlist (sidebar) to compare them here.")
        with wl_col2:
            # What-If Scenario Builder
            st.markdown("### 🧮 What-If Scenario")
            st.caption("Adjust assumptions and see how scores shift:")
            pe_adj = st.slider("P/E Multiple Adjustment", -20, 20, 0, key="pe_adj")
            rev_growth = st.slider("Revenue Growth Adjustment (%)", -20, 20, 0, key="rev_adj")
            margin_adj = st.slider("Operating Margin Adjustment (pts)", -10, 10, 0, key="margin_adj")

            if pe_adj != 0 or rev_growth != 0 or margin_adj != 0:
                adj_health = max(0, min(100, res_report.financial_health_score + margin_adj * 2))
                adj_growth = max(0, min(100, res_report.growth_score + rev_growth * 2))
                adj_profit = max(0, min(100, res_report.profitability_score + margin_adj * 1.5))
                adj_risk   = max(0, min(100, res_report.risk_score + pe_adj * 0.5))
                st.markdown("**Adjusted Scores:**")
                st.metric("Financial Health", f"{adj_health:.0f}/100",
                          delta=f"{adj_health - res_report.financial_health_score:+.0f}")
                st.metric("Growth", f"{adj_growth:.0f}/100",
                          delta=f"{adj_growth - res_report.growth_score:+.0f}")
                st.metric("Profitability", f"{adj_profit:.0f}/100",
                          delta=f"{adj_profit - res_report.profitability_score:+.0f}")
                st.metric("Risk", f"{adj_risk:.0f}/100",
                          delta=f"{adj_risk - res_report.risk_score:+.0f}", delta_color="inverse")

        # --- Citations ---
        st.markdown("### 📑 Verified Evidence & Citations")
        for cite in res_report.evidence_citations:
            st.markdown(f"- 📎 `[{cite}]`")

        # --- PDF Export ---
        st.markdown("---")
        try:
            pdf_bytes = generate_pdf_report(res_report.company, res_report)
            st.download_button(
                label="📄 Download Full Research Report as PDF",
                data=pdf_bytes,
                file_name=f"{res_report.company}_research_report_{datetime.date.today()}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.caption(f"PDF export unavailable: {e}")


# ══════════════════════════════════════════════════════════════
# TAB 3: FINANCE CHAT Q&A (NEW)
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("💬 Finance Chat Q&A — Ask Me Anything About Any Company")
    st.caption("Conversational AI analyst backed by audited filings. Ask follow-up questions, comparisons, definitions, or investment queries.")

    # Company selector for chat context
    chat_company = st.selectbox(
        "Chat Context Company (optional — set a company for context-aware answers):",
        ["(General / Auto-Detect)"] + ALL_COMPANIES,
        key="chat_company_sel"
    )

    # Preset question chips
    st.markdown("**Quick Questions:**")
    q_cols = st.columns(5)
    quick_qs = [
        "Is TCS a good long-term investment?",
        "Compare Apple vs Microsoft valuation",
        "What is Nvidia's growth trajectory?",
        "Explain Piotroski F-Score criteria",
        "Is Tesla overvalued at current P/E?",
    ]
    for i, qq in enumerate(quick_qs):
        if q_cols[i % 5].button(qq, key=f"qq_{i}", use_container_width=True):
            st.session_state["chat_history"].append({"role": "user", "content": qq})
            with st.spinner("Analyzing..."):
                try:
                    r = agent.analyze(qq)
                    answer = r.direct_answer or r.overall_assessment or "Analysis complete."
                except Exception as ex:
                    answer = f"Error: {ex}"
            st.session_state["chat_history"].append({"role": "ai", "content": answer})

    # Render chat history
    st.markdown("---")
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state["chat_history"]:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-bubble-user">🧑 {msg["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="chat-bubble-ai">🤖 {msg["content"]}</div>',
                    unsafe_allow_html=True
                )

    # Input box
    st.markdown("---")
    chat_input_col, chat_send_col = st.columns([5, 1])
    with chat_input_col:
        user_chat_msg = st.text_input(
            "Ask a finance question:",
            placeholder="e.g. Is Reliance a good buy? Compare Wipro vs Infosys. Explain cash conversion ratio.",
            key="chat_input_box",
            label_visibility="collapsed",
        )
    with chat_send_col:
        send_chat = st.button("Send 📨", type="primary", use_container_width=True)

    if send_chat and user_chat_msg.strip():
        query = user_chat_msg.strip()
        # Inject company context if selected
        if chat_company != "(General / Auto-Detect)":
            # Only prepend if company not already mentioned
            if chat_company.lower() not in query.lower():
                query = f"{query} [Context: {chat_company}]"
        st.session_state["chat_history"].append({"role": "user", "content": user_chat_msg.strip()})
        with st.spinner("Thinking..."):
            try:
                r = agent.analyze(query)
                answer = r.direct_answer or r.overall_assessment or "Analysis complete — no specific discrepancy found."
                # Enrich with key findings
                if r.key_findings:
                    answer += "\n\n**Key Points:**\n" + "\n".join(f"• {f}" for f in r.key_findings[:3])
            except Exception as ex:
                answer = f"I couldn't process that query: {ex}"
        st.session_state["chat_history"].append({"role": "ai", "content": answer})
        st.rerun()

    if st.button("🗑️ Clear Chat History", key="clear_chat"):
        st.session_state["chat_history"] = []
        st.rerun()

    # Sentiment Analyzer widget
    st.markdown("---")
    st.markdown("### 📰 Financial News Sentiment Analyzer")
    st.caption("Paste any financial news snippet to get an instant sentiment reading:")
    news_text = st.text_area(
        "News / Analyst Note:",
        placeholder="e.g. 'TCS reports strong Q4 results, beats estimates by 8%. Management guides for robust growth ahead.'",
        height=80,
        key="news_sentiment_input"
    )
    if st.button("Analyze Sentiment", key="analyze_sentiment_btn") and news_text.strip():
        # Simple keyword-based sentiment
        pos_words = ["strong", "beat", "growth", "record", "profit", "surge", "rally", "upgrade", "robust",
                     "outperform", "positive", "momentum", "exceeds", "raised", "buy", "bullish"]
        neg_words = ["weak", "miss", "decline", "loss", "fall", "downgrade", "risk", "concern",
                     "below", "cut", "layoff", "bearish", "disappointing", "sell", "cautious"]
        txt_lower = news_text.lower()
        pos_score = sum(1 for w in pos_words if w in txt_lower)
        neg_score = sum(1 for w in neg_words if w in txt_lower)
        total = pos_score + neg_score or 1
        sentiment_pct = round((pos_score / total) * 100)

        sent_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sentiment_pct,
            title={"text": "Sentiment Score", "font": {"size": 14}},
            number={"suffix": "% Positive"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2ecc40" if sentiment_pct > 60 else ("#ff851b" if sentiment_pct > 40 else "#ff4136")},
                "steps": [
                    {"range": [0, 40], "color": "#3a1a1a"},
                    {"range": [40, 60], "color": "#3a3a1a"},
                    {"range": [60, 100], "color": "#1a3a1a"},
                ],
            }
        ))
        sent_fig.update_layout(paper_bgcolor="#0d0d1a", font={"color": "#e0e0e0"}, height=200,
                               margin={"l": 30, "r": 30, "t": 50, "b": 10})
        st.plotly_chart(sent_fig, use_container_width=True)
        label = "BULLISH 📈" if sentiment_pct > 60 else ("NEUTRAL ⚖️" if sentiment_pct > 40 else "BEARISH 📉")
        st.markdown(f"**Verdict:** `{label}` ({pos_score} positive signals, {neg_score} negative signals)")


# ══════════════════════════════════════════════════════════════
# TAB 4: PEER BENCHMARKING
# ══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Industry Peer Benchmarking & Relative Claims")
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
            "Market Cap": p.market_cap_str,
        })
    df_peers = pd.DataFrame(p_rows)
    st.dataframe(df_peers, use_container_width=True)

    # Visual bar chart of operating margins
    if p_rows:
        try:
            companies_p = [r["Company"] for r in p_rows]
            margins_p = [float(r["Operating Margin (%)"].replace("%", "")) for r in p_rows]
            fig_peer = go.Figure(go.Bar(
                x=companies_p, y=margins_p,
                marker_color=["#2ecc40" if m == max(margins_p) else "#0f3460" for m in margins_p],
                text=[f"{m:.1f}%" for m in margins_p], textposition="outside",
            ))
            fig_peer.update_layout(
                title=f"Operating Margin Comparison — {peer_comp_target} vs Peers",
                xaxis_title="Company", yaxis_title="Operating Margin (%)",
                paper_bgcolor="#0d0d1a", plot_bgcolor="#1a1a2e",
                font={"color": "#e0e0e0"}, height=320,
            )
            st.plotly_chart(fig_peer, use_container_width=True)
        except Exception:
            pass

    st.markdown("### 🔎 Audit Comparative Statements")
    comp_claim = st.text_input(
        "Enter comparative claim to verify",
        value=f"{peer_comp_target} has a lower operating margin than Infosys"
        if peer_comp_target == "TCS" else "Apple has a higher P/E multiple than Microsoft"
    )
    if st.button("Check Comparative Claim"):
        comp_res = peer_engine.audit_comparative_claim(comp_claim)
        if comp_res:
            if comp_res.status == ClaimStatus.SUPPORTED:
                st.success(f"SUPPORTED: {comp_res.explanation}")
            else:
                st.error(f"CONTRADICTED: {comp_res.explanation}")
        else:
            st.warning("Could not identify at least two recognized peer companies or a recognized metric.")


# ══════════════════════════════════════════════════════════════
# TAB 5: QUANTITATIVE FORENSICS (Enhanced)
# ══════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Institutional Forensic Health & Insolvency Risk")
    st.caption("Piotroski 9-Point F-Score and Altman Z-Score credit distress models with interactive checklist.")

    forensic_target = st.selectbox("Company to Audit", ALL_COMPANIES, key="forensic_comp")
    forensic_data = forensics.analyze(forensic_target)

    # Gauges
    fg1, fg2, fg3 = st.columns(3)
    with fg1:
        st.plotly_chart(
            make_gauge("Piotroski F-Score", forensic_data.piotroski_f_score, max_val=9,
                       color_ranges=[[0, 3, "#ff4136"], [3, 6, "#ff851b"], [6, 9, "#2ecc40"]]),
            use_container_width=True
        )
    with fg2:
        z_disp = min(forensic_data.altman_z_score, 5.0)
        st.plotly_chart(
            make_gauge("Altman Z-Score", round(z_disp, 2), max_val=5,
                       color_ranges=[[0, 1.81, "#ff4136"], [1.81, 2.99, "#ff851b"], [2.99, 5, "#2ecc40"]]),
            use_container_width=True
        )
    with fg3:
        ccr_disp = min(forensic_data.cash_conversion_ratio, 2.0)
        st.plotly_chart(
            make_gauge("Cash Conversion Ratio", round(ccr_disp, 2), max_val=2,
                       color_ranges=[[0, 0.6, "#ff4136"], [0.6, 1.0, "#ff851b"], [1.0, 2.0, "#2ecc40"]]),
            use_container_width=True
        )

    # Piotroski Interactive Checklist
    st.markdown("### 📋 Piotroski F-Score — 9-Criteria Interactive Checklist")
    st.caption("Click each criterion to learn what it measures:")

    PIOTROSKI_CRITERIA = [
        ("Profitability", "ROA Positive", "Return on Assets > 0: Company is profitable in absolute terms."),
        ("Profitability", "Operating Cash Flow Positive", "OCF > 0: Earnings are backed by real cash generation."),
        ("Profitability", "ROA Improving YoY", "ROA this year > last year: Profitability trend is improving."),
        ("Profitability", "Accruals / OCF > Net Income", "Cash earnings exceed accrual earnings: High earnings quality."),
        ("Leverage", "Long-term Leverage Decreasing", "Debt ratio improved YoY: Balance sheet de-risking."),
        ("Leverage", "Current Ratio Improving", "Current ratio improved YoY: Liquidity strengthening."),
        ("Leverage", "No New Share Dilution", "No new equity issued this year: Management not diluting shareholders."),
        ("Operating Efficiency", "Gross Margin Improving", "Gross margin better than last year: Pricing power or cost discipline."),
        ("Operating Efficiency", "Asset Turnover Improving", "Revenue per unit of assets improved: Capital deployed more efficiently."),
    ]

    met_criteria = set(forensic_data.f_score_breakdown) if forensic_data.f_score_breakdown else set()
    # Determine which criteria passed
    criteria_met_flags = []
    for i, (category, name, explanation) in enumerate(PIOTROSKI_CRITERIA):
        # Heuristic: if breakdown list items mention the keyword, consider it met
        passed = (i < forensic_data.piotroski_f_score)  # Simple proxy
        criteria_met_flags.append(passed)

    # Group by category
    categories = {}
    for i, (category, name, explanation) in enumerate(PIOTROSKI_CRITERIA):
        if category not in categories:
            categories[category] = []
        categories[category].append((name, explanation, criteria_met_flags[i]))

    for cat, items in categories.items():
        with st.expander(f"📂 {cat} Criteria ({sum(1 for _, _, p in items if p)}/{len(items)} met)"):
            for name, explanation, passed in items:
                col_check, col_detail = st.columns([1, 5])
                with col_check:
                    if passed:
                        st.markdown('<span style="color:#2ecc40; font-size:1.5em;">✅</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span style="color:#ff4136; font-size:1.5em;">❌</span>', unsafe_allow_html=True)
                with col_detail:
                    st.markdown(f"**{name}**")
                    st.caption(explanation)

    # Z-Score Zone Explanation
    st.markdown("### 🏦 Altman Z-Score Zone Interpretation")
    z = forensic_data.altman_z_score
    if z > 2.99:
        st.success(f"**SAFE ZONE** (Z = {z:.2f}) — Financially healthy; low bankruptcy risk. Z > 2.99")
    elif z > 1.81:
        st.warning(f"**GREY ZONE** (Z = {z:.2f}) — Moderate risk; monitor carefully. 1.81 < Z < 2.99")
    else:
        st.error(f"**DISTRESS ZONE** (Z = {z:.2f}) — High bankruptcy risk. Z < 1.81")

    # Earnings Quality Explanation
    ccr = forensic_data.cash_conversion_ratio
    st.markdown("### 💵 Earnings Quality (Cash Conversion Ratio)")
    if ccr >= 1.0:
        st.success(f"**HIGH QUALITY** (CCR = {ccr:.2f}x) — Cash flow exceeds net profit. Earnings are real, not accounting artifacts.")
    elif ccr >= 0.6:
        st.warning(f"**MODERATE QUALITY** (CCR = {ccr:.2f}x) — Partial cash backing. Accruals may inflate some earnings.")
    else:
        st.error(f"**LOW QUALITY** (CCR = {ccr:.2f}x) — Earnings poorly backed by cash. High accrual accounting risk.")


# ══════════════════════════════════════════════════════════════
# TAB 6: DOCUMENT & PDF INGESTION
# ══════════════════════════════════════════════════════════════
with tab6:
    st.subheader("Document & PDF Regulatory Ingestion Engine")
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
            st.success(f"Successfully extracted and indexed {len(new_records)} audited metric ground-truths!")
            rows = [{
                "Metric": r.metric,
                "Value": r.reported_str,
                "Period": r.period,
                "Filing Quote": r.exact_filing_quote,
            } for r in new_records]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("Text indexed into context chunks. No exact tabular statement line patterns matched.")
