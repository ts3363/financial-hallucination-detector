"""
financial/rag_knowledge_base.py
================================
Retrieval-Augmented Generation (RAG) Knowledge Base for Finance.
Contains a comprehensive financial knowledge corpus covering:
  - Financial concepts, ratios, and formulas
  - Investment strategies and frameworks
  - Market mechanics and instruments
  - Accounting standards and terminology
  - Regulatory and compliance concepts
  - Behavioral finance and risk management
  - Sector-specific analysis frameworks

Uses TF-IDF vector similarity for chunk retrieval to ground LLM answers
in verified financial knowledge, reducing hallucination.
"""

from __future__ import annotations
import re
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import Counter


@dataclass
class KnowledgeChunk:
    """A single knowledge chunk with metadata."""
    chunk_id: str
    category: str
    title: str
    content: str
    keywords: List[str] = field(default_factory=list)
    source: str = "Financial Knowledge Base (Curated)"


# ── COMPREHENSIVE FINANCIAL KNOWLEDGE CORPUS ──────────────────
FINANCIAL_KNOWLEDGE_CORPUS: List[KnowledgeChunk] = [
    # ═══════════════════════════════════════════════════════════
    # VALUATION RATIOS & METRICS
    # ═══════════════════════════════════════════════════════════
    KnowledgeChunk(
        chunk_id="val_001", category="Valuation",
        title="Price-to-Earnings (P/E) Ratio",
        content=(
            "The Price-to-Earnings (P/E) ratio is calculated as Market Price per Share divided by "
            "Earnings per Share (EPS). It tells investors how much they are paying for each rupee/dollar "
            "of earnings. A high P/E may indicate overvaluation or high growth expectations, while a low "
            "P/E may signal undervaluation or declining prospects. Trailing P/E uses past 12-month earnings; "
            "Forward P/E uses estimated future earnings. Industry comparison is essential — tech stocks "
            "typically trade at higher P/E multiples (25-50x) than utilities (10-18x) or banks (8-15x). "
            "The PEG ratio (P/E divided by earnings growth rate) adjusts P/E for growth; PEG < 1 suggests "
            "undervaluation relative to growth."
        ),
        keywords=["pe ratio", "price to earnings", "valuation", "eps", "peg ratio", "overvalued", "undervalued", "multiple"]
    ),
    KnowledgeChunk(
        chunk_id="val_002", category="Valuation",
        title="Price-to-Book (P/B) Ratio",
        content=(
            "The Price-to-Book (P/B) ratio compares a company's market price to its book value per share. "
            "Book value = Total Assets minus Total Liabilities (shareholders' equity). P/B < 1 means the "
            "stock trades below its net asset value — potentially undervalued or in financial distress. "
            "P/B > 3 is common for asset-light businesses (IT services, software) where intellectual capital "
            "isn't fully captured on the balance sheet. Banks and financial institutions are commonly valued "
            "using P/B because their assets (loans) are marked closer to market value. Indian IT companies "
            "like TCS and Infosys often trade at P/B of 10-15x due to high ROE and low tangible assets."
        ),
        keywords=["price to book", "pb ratio", "book value", "net asset value", "nav", "undervalued"]
    ),
    KnowledgeChunk(
        chunk_id="val_003", category="Valuation",
        title="Enterprise Value and EV/EBITDA",
        content=(
            "Enterprise Value (EV) = Market Capitalization + Total Debt - Cash and Cash Equivalents. "
            "EV represents the total takeover price of a company. EV/EBITDA ratio is a more comprehensive "
            "valuation multiple than P/E because it is capital-structure neutral (unaffected by leverage), "
            "accounts for cash reserves, and excludes non-cash depreciation. EV/EBITDA below 10x is often "
            "considered attractive; above 20x suggests premium valuation. It is especially useful for "
            "comparing companies with different debt levels. For capital-intensive industries like telecom, "
            "infrastructure, and energy, EV/EBITDA is the primary valuation benchmark."
        ),
        keywords=["enterprise value", "ev", "ebitda", "ev/ebitda", "takeover", "capital structure", "valuation multiple"]
    ),
    KnowledgeChunk(
        chunk_id="val_004", category="Valuation",
        title="Discounted Cash Flow (DCF) Valuation",
        content=(
            "DCF valuation estimates intrinsic value by discounting projected future free cash flows back "
            "to present value using a discount rate (typically WACC — Weighted Average Cost of Capital). "
            "Steps: (1) Project Free Cash Flow to Firm (FCFF) for 5-10 years, (2) Calculate Terminal Value "
            "using perpetuity growth method: TV = FCF*(1+g)/(WACC-g), (3) Discount all cash flows to "
            "present value, (4) Subtract net debt to get equity value, (5) Divide by shares outstanding. "
            "Key assumptions: revenue growth rate, operating margins, capex, working capital changes, "
            "terminal growth rate (typically 2-4%), and WACC (typically 8-12%). DCF is highly sensitive "
            "to terminal value and discount rate assumptions — a 1% change in WACC can swing value 15-25%."
        ),
        keywords=["dcf", "discounted cash flow", "intrinsic value", "wacc", "free cash flow", "terminal value", "fcff"]
    ),
    KnowledgeChunk(
        chunk_id="val_005", category="Valuation",
        title="Dividend Discount Model (DDM)",
        content=(
            "The Dividend Discount Model values a stock as the present value of all expected future dividends. "
            "Gordon Growth Model: Price = D1 / (r - g), where D1 = next year's dividend, r = required return, "
            "g = dividend growth rate. Only works when g < r. Best suited for mature, stable dividend-paying "
            "companies (utilities, banks, consumer staples). Not applicable to non-dividend-paying growth stocks "
            "like many tech companies. Two-stage DDM uses a high-growth phase followed by a stable phase. "
            "Dividend yield = Annual Dividend / Stock Price. A high dividend yield (>4%) may indicate value "
            "opportunity or financial stress. Indian blue chips like TCS, Infosys, and HDFC Bank are known "
            "for consistent dividend payouts."
        ),
        keywords=["dividend", "ddm", "dividend discount model", "gordon growth", "dividend yield", "payout"]
    ),

    # ═══════════════════════════════════════════════════════════
    # PROFITABILITY METRICS
    # ═══════════════════════════════════════════════════════════
    KnowledgeChunk(
        chunk_id="prof_001", category="Profitability",
        title="Return on Equity (ROE)",
        content=(
            "Return on Equity (ROE) = Net Income / Shareholders' Equity. It measures how efficiently a "
            "company generates profits from shareholders' capital. ROE > 15% is generally considered good; "
            "> 25% is excellent. The DuPont decomposition breaks ROE into three components: "
            "ROE = Net Margin x Asset Turnover x Equity Multiplier. This reveals whether high ROE comes "
            "from profitability (margins), efficiency (asset utilization), or leverage (debt). High ROE "
            "driven primarily by leverage is riskier than ROE from margins. TCS has an ROE of ~51% driven "
            "by high margins and low capital requirements. Apple's ROE exceeds 140% partly due to massive "
            "share buybacks reducing equity. A consistently declining ROE is a warning sign."
        ),
        keywords=["roe", "return on equity", "dupont", "profitability", "equity", "net income", "shareholder"]
    ),
    KnowledgeChunk(
        chunk_id="prof_002", category="Profitability",
        title="Operating Margin and Gross Margin",
        content=(
            "Gross Margin = (Revenue - Cost of Goods Sold) / Revenue. It shows how much profit remains "
            "after direct production costs. Operating Margin = Operating Income / Revenue, which also "
            "deducts selling, general & administrative (SG&A) expenses. Net Margin = Net Income / Revenue "
            "after all expenses including interest and taxes. Margin trends matter more than absolute levels. "
            "Expanding margins signal pricing power, scale benefits, or cost discipline. Contracting margins "
            "may indicate competitive pressure, input cost inflation, or heavy investments. IT services "
            "companies typically have operating margins of 20-28%. Apple's gross margin is ~46%, among the "
            "highest in consumer electronics. Software companies like Microsoft enjoy 40%+ operating margins."
        ),
        keywords=["operating margin", "gross margin", "net margin", "profitability", "cogs", "sga", "margin expansion"]
    ),
    KnowledgeChunk(
        chunk_id="prof_003", category="Profitability",
        title="Return on Assets (ROA) and Return on Capital Employed (ROCE)",
        content=(
            "Return on Assets (ROA) = Net Income / Total Assets. It measures how productively all assets "
            "generate earnings. ROA > 5% is acceptable for most industries; > 10% is strong. "
            "Return on Capital Employed (ROCE) = EBIT / Capital Employed, where Capital Employed = "
            "Total Assets - Current Liabilities. ROCE is a better measure for capital-intensive businesses. "
            "ROCE > WACC indicates the company creates value. Comparing ROCE across companies within the "
            "same sector reveals capital efficiency. Reliance Industries has a ROCE of ~9% (capital-heavy), "
            "while TCS achieves ROCE > 50% (asset-light IT model)."
        ),
        keywords=["roa", "return on assets", "roce", "capital employed", "ebit", "capital efficiency"]
    ),

    # ═══════════════════════════════════════════════════════════
    # FINANCIAL HEALTH & SOLVENCY
    # ═══════════════════════════════════════════════════════════
    KnowledgeChunk(
        chunk_id="health_001", category="Financial Health",
        title="Piotroski F-Score (9-Point Financial Health)",
        content=(
            "The Piotroski F-Score is a 9-point integer scoring system that assesses a company's financial "
            "strength. Each criterion scores 1 (pass) or 0 (fail). The 9 criteria across 3 pillars: "
            "\n\nPROFITABILITY (4 points): "
            "(1) Positive Net Income (ROA > 0), "
            "(2) Positive Operating Cash Flow, "
            "(3) Improving ROA year-over-year, "
            "(4) Cash Flow from Operations > Net Income (accrual quality). "
            "\n\nLEVERAGE & LIQUIDITY (3 points): "
            "(5) Decrease in long-term debt ratio, "
            "(6) Increase in current ratio, "
            "(7) No new share issuance (zero dilution). "
            "\n\nOPERATING EFFICIENCY (2 points): "
            "(8) Improving gross margin, "
            "(9) Improving asset turnover ratio. "
            "\n\nInterpretation: F-Score 8-9 = financially robust (top decile); 5-7 = moderate; "
            "0-3 = financially distressed (bottom quartile). Studies show high F-Score stocks outperform "
            "low F-Score stocks by 7-10% annually."
        ),
        keywords=["piotroski", "f-score", "f score", "financial health", "9 criteria", "profitability", "leverage", "operating efficiency"]
    ),
    KnowledgeChunk(
        chunk_id="health_002", category="Financial Health",
        title="Altman Z-Score (Bankruptcy Prediction)",
        content=(
            "The Altman Z-Score predicts corporate bankruptcy probability using 5 weighted financial ratios: "
            "Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5, where: "
            "X1 = Working Capital / Total Assets (liquidity), "
            "X2 = Retained Earnings / Total Assets (cumulative profitability), "
            "X3 = EBIT / Total Assets (operating efficiency), "
            "X4 = Market Value of Equity / Book Value of Total Liabilities (solvency), "
            "X5 = Sales / Total Assets (asset efficiency). "
            "\n\nInterpretation zones: Z > 2.99 = Safe Zone (virtually zero default risk within 24 months); "
            "1.81 < Z < 2.99 = Grey Zone (moderate risk, needs monitoring); "
            "Z < 1.81 = Distress Zone (high bankruptcy probability, immediate investigation required). "
            "The model has 72-80% accuracy for predicting bankruptcy 2 years in advance."
        ),
        keywords=["altman", "z-score", "z score", "bankruptcy", "distress", "solvency", "default risk", "safe zone"]
    ),
    KnowledgeChunk(
        chunk_id="health_003", category="Financial Health",
        title="Debt-to-Equity Ratio and Interest Coverage",
        content=(
            "Debt-to-Equity (D/E) Ratio = Total Debt / Shareholders' Equity. A D/E of 0 means no debt; "
            "D/E > 1 means more debt than equity. Acceptable D/E varies by industry: IT services (< 0.3), "
            "banks (8-12, regulatory leverage), utilities (1-2), infrastructure (1-3). "
            "Interest Coverage Ratio = EBIT / Interest Expense. ICR > 3 is healthy; ICR < 1.5 is risky "
            "(company may struggle to service debt). Net Debt = Total Debt - Cash & Equivalents. "
            "Negative net debt means the company holds more cash than it owes. TCS (D/E 0.08) and Google "
            "(D/E 0.11) are virtually debt-free. Apple (D/E 1.52) carries substantial debt for share buybacks, "
            "but its massive cash generation (>$100B annually) makes this sustainable."
        ),
        keywords=["debt to equity", "leverage", "interest coverage", "debt", "solvency", "net debt", "icr"]
    ),
    KnowledgeChunk(
        chunk_id="health_004", category="Financial Health",
        title="Cash Conversion Ratio and Free Cash Flow",
        content=(
            "Cash Conversion Ratio (CCR) = Operating Cash Flow / Net Profit. CCR > 1.0 means cash earnings "
            "exceed accrual earnings — a sign of high-quality, sustainable profits. CCR < 0.5 raises red flags "
            "about revenue recognition, working capital strain, or aggressive accounting. "
            "Free Cash Flow (FCF) = Operating Cash Flow - Capital Expenditure. FCF represents money available "
            "for dividends, buybacks, acquisitions, or debt repayment. FCF yield = FCF per share / Stock Price. "
            "Companies with consistently positive and growing FCF are more resilient. Tech giants like Microsoft "
            "and Google generate FCF margins of 25-35%."
        ),
        keywords=["cash conversion", "ccr", "free cash flow", "fcf", "operating cash flow", "earnings quality", "accrual"]
    ),
    KnowledgeChunk(
        chunk_id="health_005", category="Financial Health",
        title="Current Ratio and Quick Ratio (Liquidity)",
        content=(
            "Current Ratio = Current Assets / Current Liabilities. It measures short-term liquidity — "
            "ability to pay obligations due within 1 year. CR > 1.5 is comfortable; CR < 1.0 means "
            "current liabilities exceed current assets (liquidity risk). "
            "Quick Ratio (Acid Test) = (Current Assets - Inventory) / Current Liabilities. More conservative "
            "than current ratio as it excludes inventory which may be illiquid. QR > 1.0 is healthy. "
            "For IT services companies, current ratios are typically 2-3x since they have minimal inventory. "
            "For manufacturing or retail, current ratios of 1.2-1.5x are acceptable."
        ),
        keywords=["current ratio", "quick ratio", "acid test", "liquidity", "working capital", "current assets"]
    ),

    # ═══════════════════════════════════════════════════════════
    # INVESTMENT STRATEGIES
    # ═══════════════════════════════════════════════════════════
    KnowledgeChunk(
        chunk_id="strat_001", category="Investment Strategy",
        title="Value Investing Principles",
        content=(
            "Value investing, pioneered by Benjamin Graham and popularized by Warren Buffett, involves "
            "buying stocks trading below their intrinsic value with a margin of safety. Key principles: "
            "(1) Margin of Safety — buy at significant discount to intrinsic value to protect against "
            "errors in analysis. (2) Circle of Competence — only invest in businesses you understand. "
            "(3) Economic Moat — prefer companies with durable competitive advantages (brand, network "
            "effects, switching costs, patents, cost advantages). (4) Long-term horizon — hold for years, "
            "not months. (5) Focus on earnings power and cash flow, not market momentum. "
            "Value metrics: low P/E, low P/B, high dividend yield, high FCF yield. "
            "Contrarian approach: buy when others are fearful."
        ),
        keywords=["value investing", "buffett", "graham", "intrinsic value", "margin of safety", "moat", "contrarian"]
    ),
    KnowledgeChunk(
        chunk_id="strat_002", category="Investment Strategy",
        title="Growth Investing and GARP",
        content=(
            "Growth investing focuses on companies with above-average revenue and earnings growth, even "
            "if current valuation multiples are high. Growth investors prioritize: revenue growth > 15-20% "
            "annually, expanding margins, large addressable market, strong competitive position, and "
            "management quality. They accept higher P/E ratios for faster growth. "
            "GARP (Growth at a Reasonable Price) combines value and growth by seeking companies with strong "
            "growth at reasonable valuations. The PEG ratio (P/E / Earnings Growth Rate) is the key GARP metric. "
            "PEG < 1 = potentially undervalued growth; PEG 1-2 = fairly valued; PEG > 2 = expensive. "
            "Examples: Nvidia trades at high P/E but PEG may be reasonable given 100%+ earnings growth. "
            "Peter Lynch popularized GARP and managed the Fidelity Magellan Fund to 29% annual returns."
        ),
        keywords=["growth investing", "garp", "peg ratio", "growth rate", "peter lynch", "high growth", "revenue growth"]
    ),
    KnowledgeChunk(
        chunk_id="strat_003", category="Investment Strategy",
        title="Systematic Investment Plan (SIP) and Rupee Cost Averaging",
        content=(
            "A Systematic Investment Plan (SIP) involves investing a fixed amount at regular intervals "
            "(monthly, weekly) regardless of market conditions. Benefits: (1) Rupee/Dollar Cost Averaging — "
            "buys more units when prices are low, fewer when high, reducing average cost. "
            "(2) Disciplined investing removes emotional timing decisions. (3) Power of compounding "
            "over long periods. (4) Accessible — start with as little as Rs 500/month in India. "
            "SIP in index funds (Nifty 50, S&P 500) has historically delivered 12-15% CAGR over 10+ years. "
            "SIP vs Lump Sum: SIP performs better in volatile/declining markets; lump sum outperforms in "
            "consistently rising markets. For most retail investors, monthly SIP is recommended."
        ),
        keywords=["sip", "systematic investment", "rupee cost averaging", "dollar cost averaging", "mutual fund", "compounding"]
    ),
    KnowledgeChunk(
        chunk_id="strat_004", category="Investment Strategy",
        title="Portfolio Diversification and Asset Allocation",
        content=(
            "Diversification reduces portfolio risk by spreading investments across different asset classes, "
            "sectors, and geographies. Core principle: correlation matters — assets that don't move together "
            "reduce volatility. Asset allocation models: (1) Conservative: 60-70% bonds, 20-30% equity, "
            "10% alternatives. (2) Balanced: 50% equity, 40% bonds, 10% alternatives. (3) Aggressive: "
            "70-80% equity, 10-20% bonds, 10% alternatives. Rule of thumb: equity allocation = 100 - age. "
            "Sector diversification: don't concentrate > 25% in one sector. Geographic diversification: "
            "allocate 20-40% internationally. Rebalancing quarterly or annually maintains target allocation. "
            "Over-diversification (>30 stocks) can dilute returns without reducing risk further."
        ),
        keywords=["diversification", "portfolio", "asset allocation", "risk", "correlation", "bonds", "equity", "rebalancing"]
    ),
    KnowledgeChunk(
        chunk_id="strat_005", category="Investment Strategy",
        title="Mutual Funds vs ETFs vs Direct Equity",
        content=(
            "Mutual Funds: Professionally managed, diversified baskets of stocks/bonds. Active funds try to "
            "beat the benchmark; only ~15-20% succeed over 10 years. Expense ratios: 0.5-2.0%. In India, "
            "ELSS funds offer tax benefits under Section 80C. "
            "ETFs (Exchange Traded Funds): Passively track an index (Nifty 50, S&P 500), trade like stocks "
            "on exchanges, lower expense ratios (0.02-0.5%), real-time pricing. "
            "Direct Equity: Buying individual stocks requires research, higher risk, potential for higher "
            "returns. Suitable for informed investors willing to analyze financial statements. "
            "Index funds and ETFs have consistently outperformed most active fund managers over 10+ year "
            "periods. For most retail investors: 60-70% in broad index funds/ETFs, 20-30% in direct equity, "
            "10% in debt/alternatives."
        ),
        keywords=["mutual fund", "etf", "index fund", "direct equity", "expense ratio", "active", "passive", "nifty", "sensex"]
    ),

    # ═══════════════════════════════════════════════════════════
    # MARKET MECHANICS & INSTRUMENTS
    # ═══════════════════════════════════════════════════════════
    KnowledgeChunk(
        chunk_id="mkt_001", category="Market Mechanics",
        title="Stock Market Basics — NSE, BSE, NYSE, NASDAQ",
        content=(
            "Stock exchanges are regulated marketplaces where shares are bought and sold. "
            "India: NSE (National Stock Exchange, benchmark: Nifty 50) and BSE (Bombay Stock Exchange, "
            "benchmark: Sensex/BSE 30). Both regulated by SEBI. "
            "US: NYSE (New York Stock Exchange, largest by market cap) and NASDAQ (tech-heavy exchange). "
            "Key market indices: Nifty 50 (50 largest Indian companies by market cap and liquidity), "
            "Sensex (30 largest on BSE), S&P 500 (500 largest US companies), NASDAQ Composite (tech-weighted), "
            "Dow Jones Industrial Average (30 blue-chips). "
            "Market cap tiers: Large-cap (> Rs 20,000 Cr in India / > $10B in US), "
            "Mid-cap (Rs 5,000-20,000 Cr / $2-10B), Small-cap (< Rs 5,000 Cr / < $2B). "
            "Trading hours: NSE/BSE 9:15 AM - 3:30 PM IST; NYSE 9:30 AM - 4:00 PM EST."
        ),
        keywords=["stock market", "nse", "bse", "nifty", "sensex", "nyse", "nasdaq", "exchange", "trading", "sebi"]
    ),
    KnowledgeChunk(
        chunk_id="mkt_002", category="Market Mechanics",
        title="IPO (Initial Public Offering) Process",
        content=(
            "An IPO is when a private company first sells shares to the public on a stock exchange. "
            "Process: (1) Hire investment banks (book running lead managers), (2) File DRHP/prospectus "
            "with SEBI/SEC, (3) Roadshow to institutional investors, (4) Set price band, (5) Book building "
            "(collect investor bids), (6) Allotment and listing. "
            "Types: Fresh Issue (company raises new capital) vs Offer for Sale (existing investors sell shares). "
            "Key metrics: Grey Market Premium (GMP) signals pre-listing demand, oversubscription ratio "
            "shows investor interest. QIB (Qualified Institutional Buyers), NII (Non-Institutional Investors), "
            "and Retail categories get separate quotas. "
            "Post-IPO lock-in periods prevent insiders from selling immediately (typically 6-12 months). "
            "In India, IPO allotment is through a lottery system for retail investors."
        ),
        keywords=["ipo", "initial public offering", "listing", "drhp", "book building", "gmp", "allotment", "public issue"]
    ),
    KnowledgeChunk(
        chunk_id="mkt_003", category="Market Mechanics",
        title="Derivatives — Futures and Options",
        content=(
            "Derivatives are financial contracts whose value is derived from an underlying asset. "
            "Futures: Agreement to buy/sell an asset at a predetermined price on a future date. "
            "Used for hedging (risk reduction) or speculation. Lot size and margin requirements apply. "
            "Options: Give the RIGHT (not obligation) to buy (Call) or sell (Put) at a strike price "
            "before expiry. Options premium = Intrinsic Value + Time Value. "
            "Key Greeks: Delta (price sensitivity), Gamma (delta change rate), Theta (time decay), "
            "Vega (volatility sensitivity). "
            "Strategies: Covered Call (own stock + sell call), Protective Put (own stock + buy put), "
            "Straddle (buy call + put at same strike — profit from big moves in either direction), "
            "Iron Condor (profit from range-bound markets). "
            "In India, options expire on the last Thursday of the month (monthly) or every Thursday (weekly)."
        ),
        keywords=["derivatives", "futures", "options", "call", "put", "strike price", "hedging", "greeks", "delta", "theta"]
    ),
    KnowledgeChunk(
        chunk_id="mkt_004", category="Market Mechanics",
        title="Bonds, Fixed Income, and Yield Curve",
        content=(
            "Bonds are fixed-income debt instruments where the issuer borrows money and pays periodic "
            "interest (coupon) plus principal at maturity. Types: Government bonds (G-Secs, Treasuries — "
            "lowest risk), Corporate bonds (higher yield, more risk), Municipal bonds. "
            "Bond pricing: inversely related to interest rates — when rates rise, bond prices fall. "
            "Yield = Annual coupon / Current market price. Yield to Maturity (YTM) accounts for capital "
            "gains/losses if held to maturity. Duration measures bond price sensitivity to rate changes. "
            "Yield curve: plots yields across maturities. Normal (upward sloping), Inverted (short-term > "
            "long-term — recession signal), Flat (uncertainty). "
            "Credit ratings: AAA/Aaa (highest quality) to D (default). India: CRISIL, ICRA, CARE. "
            "Global: S&P, Moody's, Fitch."
        ),
        keywords=["bonds", "fixed income", "yield", "coupon", "ytm", "yield curve", "treasury", "duration", "credit rating"]
    ),

    # ═══════════════════════════════════════════════════════════
    # ACCOUNTING & FINANCIAL STATEMENTS
    # ═══════════════════════════════════════════════════════════
    KnowledgeChunk(
        chunk_id="acct_001", category="Accounting",
        title="Three Financial Statements and How They Connect",
        content=(
            "The three core financial statements: "
            "(1) Income Statement (Profit & Loss): Shows revenue, expenses, and profit over a period. "
            "Revenue - COGS = Gross Profit. Gross Profit - Operating Expenses = Operating Profit (EBIT). "
            "EBIT - Interest - Tax = Net Profit. "
            "(2) Balance Sheet: Snapshot of assets, liabilities, and equity at a point in time. "
            "Assets = Liabilities + Shareholders' Equity. Current vs Non-current classification. "
            "(3) Cash Flow Statement: Tracks actual cash movements in three sections: Operating (business "
            "activities), Investing (capex, acquisitions), Financing (debt, dividends, buybacks). "
            "Connections: Net income from P&L flows into retained earnings on the balance sheet. "
            "Depreciation reduces P&L profit but is added back in cash flow (non-cash). "
            "Capex appears in investing cash flow and increases balance sheet assets."
        ),
        keywords=["financial statements", "income statement", "balance sheet", "cash flow", "profit and loss", "revenue", "assets", "liabilities"]
    ),
    KnowledgeChunk(
        chunk_id="acct_002", category="Accounting",
        title="EBITDA and Adjusted Earnings",
        content=(
            "EBITDA = Earnings Before Interest, Taxes, Depreciation & Amortization. It approximates "
            "operating cash flow generation before financing and accounting adjustments. "
            "EBITDA Margin = EBITDA / Revenue. Used widely for valuation (EV/EBITDA) and comparing "
            "profitability across companies with different capital structures and tax environments. "
            "Limitations: EBITDA ignores capital expenditures required to maintain the business, "
            "working capital changes, and doesn't represent actual cash available. "
            "Adjusted EBITDA excludes one-time charges (restructuring, litigation, impairment) to show "
            "'normalized' operating performance. Be cautious — companies may over-adjust to inflate margins. "
            "Indian companies report standalone and consolidated results; always prefer consolidated "
            "for the full picture including subsidiaries."
        ),
        keywords=["ebitda", "adjusted earnings", "depreciation", "amortization", "operating profit", "normalized"]
    ),
    KnowledgeChunk(
        chunk_id="acct_003", category="Accounting",
        title="Earnings Per Share (EPS) and Share Buybacks",
        content=(
            "EPS = Net Income / Weighted Average Shares Outstanding. Basic EPS uses current shares; "
            "Diluted EPS accounts for potential dilution from stock options, convertibles, warrants. "
            "EPS growth is a key driver of stock price appreciation. "
            "Share buybacks: Company repurchases its own shares from the open market, reducing shares "
            "outstanding and boosting EPS mechanically. Benefits: tax-efficient return of capital "
            "(vs dividends), signals management confidence, supports stock price. "
            "Risks: If funded by debt or at inflated prices, buybacks destroy value. Apple and Microsoft "
            "are among the largest global buyback programs. In India, buyback tax is 20% on distributed "
            "income. Companies like TCS, Infosys, and Wipro regularly conduct buybacks."
        ),
        keywords=["eps", "earnings per share", "buyback", "share repurchase", "dilution", "weighted average"]
    ),

    # ═══════════════════════════════════════════════════════════
    # RISK MANAGEMENT
    # ═══════════════════════════════════════════════════════════
    KnowledgeChunk(
        chunk_id="risk_001", category="Risk Management",
        title="Beta, Volatility, and Systematic Risk",
        content=(
            "Beta measures a stock's sensitivity to overall market movements. Beta = 1.0 means the stock "
            "moves with the market. Beta > 1 = more volatile than market (e.g. Tesla beta 2.35, Nvidia 1.68). "
            "Beta < 1 = less volatile (e.g. TCS 0.58, HDFC 0.88). Negative beta = moves opposite to market. "
            "Systematic risk (market risk) cannot be diversified away — it affects the entire market "
            "(recessions, interest rate changes, geopolitical events). Unsystematic risk (company-specific) "
            "CAN be diversified away. CAPM: Expected Return = Risk-Free Rate + Beta × (Market Return - "
            "Risk-Free Rate). In India, risk-free rate ≈ 7% (10-year G-Sec); in US ≈ 4-5% (10-year Treasury). "
            "Volatility (standard deviation of returns) is a common risk measure. VIX Index measures "
            "market-wide expected volatility (the 'fear index')."
        ),
        keywords=["beta", "volatility", "systematic risk", "capm", "market risk", "vix", "standard deviation"]
    ),
    KnowledgeChunk(
        chunk_id="risk_002", category="Risk Management",
        title="Sharpe Ratio and Risk-Adjusted Returns",
        content=(
            "Sharpe Ratio = (Portfolio Return - Risk-Free Rate) / Standard Deviation of Returns. "
            "It measures excess return per unit of risk. Sharpe > 1.0 is good; > 2.0 is excellent; < 0 "
            "means the investment underperformed risk-free assets. "
            "Sortino Ratio: Similar to Sharpe but uses only downside deviation (penalizes only negative "
            "volatility, not upside). Treynor Ratio = (Return - Risk-Free Rate) / Beta (uses systematic "
            "risk only). Information Ratio = (Portfolio Return - Benchmark Return) / Tracking Error "
            "(measures active fund manager skill). "
            "Maximum Drawdown: Largest peak-to-trough decline in portfolio value — measures worst-case "
            "loss. A good portfolio targets Sharpe > 1, Maximum Drawdown < 20%."
        ),
        keywords=["sharpe ratio", "risk adjusted", "sortino", "treynor", "drawdown", "risk free rate"]
    ),

    # ═══════════════════════════════════════════════════════════
    # BEHAVIORAL FINANCE
    # ═══════════════════════════════════════════════════════════
    KnowledgeChunk(
        chunk_id="behav_001", category="Behavioral Finance",
        title="Common Cognitive Biases in Investing",
        content=(
            "Behavioral finance studies how psychological biases affect investment decisions: "
            "(1) Confirmation Bias — seeking information that confirms existing beliefs, ignoring contrary evidence. "
            "(2) Anchoring — fixating on an initial price or value (e.g., refusing to sell below purchase price). "
            "(3) Loss Aversion — losses hurt ~2x more than equivalent gains feel good; leads to holding losers too long. "
            "(4) Herd Mentality — following the crowd into bubbles or panic selling. "
            "(5) Recency Bias — overweighting recent events/performance in decisions. "
            "(6) Overconfidence — overestimating one's analytical abilities. "
            "(7) Disposition Effect — selling winners too early, holding losers too long. "
            "(8) Sunk Cost Fallacy — continuing a bad investment because of money already lost. "
            "Mitigation: Use systematic frameworks (checklists, quantitative screens), maintain investment journals, "
            "follow predetermined entry/exit rules."
        ),
        keywords=["behavioral finance", "bias", "cognitive", "loss aversion", "herd mentality", "anchoring", "overconfidence"]
    ),

    # ═══════════════════════════════════════════════════════════
    # SECTOR ANALYSIS
    # ═══════════════════════════════════════════════════════════
    KnowledgeChunk(
        chunk_id="sector_001", category="Sector Analysis",
        title="IT Services Sector Analysis (India)",
        content=(
            "Indian IT services sector (TCS, Infosys, Wipro, HCL Tech, Tech Mahindra) is a $250+ billion "
            "industry with 5+ million employees. Key metrics to track: Revenue growth (constant currency), "
            "Operating margins (EBIT margin, typically 20-28%), Attrition rate, Deal Total Contract Value (TCV), "
            "and Revenue per employee. Growth drivers: Digital transformation, cloud migration, AI/ML adoption, "
            "enterprise automation. Risks: Visa regulation changes (H-1B), client spending cuts during recessions, "
            "AI disruption replacing some services. TCS is India's largest IT company by market cap and revenue. "
            "Infosys known for higher growth focus. Wipro for strategic acquisitions. "
            "Typical valuation: P/E 20-30x, EV/EBITDA 15-22x. Seasonally, Q3 is weak (furloughs), Q4 is strong "
            "(client budget flush)."
        ),
        keywords=["it services", "tcs", "infosys", "wipro", "hcl", "tech mahindra", "digital transformation", "attrition", "outsourcing"]
    ),
    KnowledgeChunk(
        chunk_id="sector_002", category="Sector Analysis",
        title="Banking Sector Analysis (India)",
        content=(
            "Indian banking sector includes PSU banks (SBI, Bank of Baroda, PNB) and private banks "
            "(HDFC Bank, ICICI Bank, Kotak, Axis). Key metrics: Net Interest Margin (NIM) — net interest "
            "income / average earning assets, typically 3-4% for well-run banks. Gross NPA (Non-Performing "
            "Assets) — loans where borrowers have defaulted; healthy < 2%. CASA Ratio — Current Account + "
            "Savings Account deposits as % of total deposits; higher = lower cost of funds. "
            "Credit Growth — loan portfolio expansion rate. Provision Coverage Ratio — provisions made against "
            "bad loans. ROA for banks: > 1% is good; > 1.5% is excellent. "
            "HDFC Bank is India's most valuable bank with consistently low NPAs and high NIMs. "
            "Key risks: rising interest rates compress NIMs, asset quality deterioration in economic downturns."
        ),
        keywords=["banking", "hdfc", "icici", "sbi", "nim", "npa", "casa", "credit growth", "loan", "interest rate"]
    ),
    KnowledgeChunk(
        chunk_id="sector_003", category="Sector Analysis",
        title="Technology Sector — AI and Semiconductors (Global)",
        content=(
            "The global technology sector is driven by AI/ML, cloud computing, and semiconductor cycles. "
            "Nvidia dominates AI GPU hardware with >80% market share in data center AI accelerators. "
            "Revenue growth > 100% YoY driven by AI training and inference demand. "
            "Microsoft leads in enterprise cloud (Azure) and AI integration (Copilot, OpenAI partnership). "
            "Google/Alphabet: Search advertising (80% of revenue), Google Cloud (growing segment), Waymo. "
            "Apple: Consumer hardware ecosystem (iPhone 50%+ of revenue), growing Services segment. "
            "Amazon: E-commerce + AWS (cloud infrastructure leader by market share). "
            "Meta: Social media advertising + Reality Labs (metaverse investments, heavy capex). "
            "Tesla: EV manufacturer + energy storage; high beta, volatile; trades on growth expectations. "
            "Key risks: Antitrust regulation, AI compute cost escalation, semiconductor cyclicality."
        ),
        keywords=["technology", "ai", "semiconductor", "nvidia", "microsoft", "google", "apple", "amazon", "meta", "tesla", "cloud"]
    ),

    # ═══════════════════════════════════════════════════════════
    # TAX & REGULATORY
    # ═══════════════════════════════════════════════════════════
    KnowledgeChunk(
        chunk_id="tax_001", category="Tax & Regulatory",
        title="Capital Gains Tax (India)",
        content=(
            "India's capital gains tax structure: "
            "Equity shares (listed): STCG (Short-Term, < 12 months holding) taxed at 15%. "
            "LTCG (Long-Term, > 12 months) taxed at 10% on gains exceeding Rs 1 lakh per year. "
            "Equity mutual funds: Same as listed shares. "
            "Debt mutual funds: Taxed at individual slab rate regardless of holding period (post-2023 changes). "
            "Unlisted shares: STCG at slab rate (< 24 months), LTCG at 20% with indexation (> 24 months). "
            "Securities Transaction Tax (STT): 0.1% on equity delivery, 0.0125% on intraday sell, "
            "0.05% on futures, 0.0625% on options. "
            "Tax-saving investments: ELSS mutual funds (80C, 3-year lock-in), PPF, NPS."
        ),
        keywords=["capital gains", "tax", "stcg", "ltcg", "stt", "section 80c", "elss", "india tax"]
    ),
    KnowledgeChunk(
        chunk_id="tax_002", category="Tax & Regulatory",
        title="SEBI Regulations and Market Governance (India)",
        content=(
            "SEBI (Securities and Exchange Board of India) regulates securities markets to protect investors "
            "and ensure fair, transparent markets. Key regulations: "
            "Insider Trading: SEBI (Prohibition of Insider Trading) Regulations, 2015 — trading on Unpublished "
            "Price Sensitive Information (UPSI) is illegal. Company insiders must declare holdings and trades. "
            "LODR (Listing Obligations and Disclosure Requirements): Listed companies must make timely disclosures "
            "of financial results, board changes, and material events. "
            "Mutual Fund Regulations: Expense ratio caps (2.25% max for equity AUM < Rs 500 Cr), exit load limits, "
            "risk labeling (riskometer). "
            "Market circuit breakers: Trading halts when Nifty/Sensex moves 10%, 15%, or 20% in either direction. "
            "FPI (Foreign Portfolio Investor) limits and ownership disclosure requirements."
        ),
        keywords=["sebi", "regulation", "insider trading", "lodr", "disclosure", "circuit breaker", "compliance", "fpi"]
    ),

    # ═══════════════════════════════════════════════════════════
    # MACROECONOMICS
    # ═══════════════════════════════════════════════════════════
    KnowledgeChunk(
        chunk_id="macro_001", category="Macroeconomics",
        title="Interest Rates, Inflation, and Central Bank Policy",
        content=(
            "Central banks (RBI in India, Federal Reserve in US) set benchmark interest rates to control "
            "inflation and economic growth. In India: Repo Rate (rate at which RBI lends to banks) is the "
            "key policy rate. Reverse Repo is the rate RBI pays banks for parking funds. "
            "When inflation is high, central banks RAISE rates — this increases borrowing costs, slows spending, "
            "strengthens currency, and typically causes stock markets to fall (especially growth stocks). "
            "When economy is weak, central banks CUT rates — this stimulates borrowing, spending, and investment. "
            "Impact on stocks: Rate hikes hurt high-P/E growth stocks most (future earnings worth less in present "
            "value). Banking stocks benefit from rate hikes through wider NIMs. Bond prices fall when rates rise. "
            "CPI (Consumer Price Index) is the primary inflation measure. RBI targets 4% inflation (+/- 2%). "
            "Fed targets 2% inflation."
        ),
        keywords=["interest rate", "inflation", "rbi", "federal reserve", "repo rate", "monetary policy", "cpi"]
    ),
    KnowledgeChunk(
        chunk_id="macro_002", category="Macroeconomics",
        title="GDP Growth and Economic Indicators",
        content=(
            "GDP (Gross Domestic Product) = Total value of goods and services produced in a country. "
            "India's GDP growth: 6-7% annually (one of the fastest-growing major economies). "
            "US GDP growth: typically 2-3% annually. Key economic indicators: "
            "PMI (Purchasing Managers Index): > 50 = expansion, < 50 = contraction. "
            "IIP (Index of Industrial Production): Measures factory output in India. "
            "Unemployment Rate: Lower is better for consumer spending and corporate earnings. "
            "Trade Balance: Exports - Imports. India runs a trade deficit (imports > exports). "
            "FDI/FII flows: Foreign investment into India signals global confidence. "
            "Correlation with markets: GDP growth drives corporate earnings, which drive stock prices long-term. "
            "But markets are forward-looking and may diverge from current GDP in anticipation of future trends."
        ),
        keywords=["gdp", "economic indicators", "pmi", "unemployment", "trade deficit", "fdi", "iip", "growth rate"]
    ),

    # ═══════════════════════════════════════════════════════════
    # TECHNICAL ANALYSIS
    # ═══════════════════════════════════════════════════════════
    KnowledgeChunk(
        chunk_id="tech_001", category="Technical Analysis",
        title="Technical Analysis — Moving Averages and Trends",
        content=(
            "Technical analysis studies price and volume patterns to predict future price movements. "
            "Key tools: Moving Averages — SMA (Simple Moving Average) and EMA (Exponential Moving Average). "
            "50-day MA: short-term trend. 200-day MA: long-term trend. "
            "Golden Cross: 50-day MA crosses ABOVE 200-day MA — bullish signal. "
            "Death Cross: 50-day MA crosses BELOW 200-day MA — bearish signal. "
            "Support: Price level where buying demand historically prevents further decline. "
            "Resistance: Price level where selling pressure historically prevents further rise. "
            "Breakout: When price moves above resistance (bullish) or below support (bearish) with high volume. "
            "RSI (Relative Strength Index): 0-100 scale. > 70 = overbought (may fall), < 30 = oversold (may rise). "
            "MACD (Moving Average Convergence Divergence): Measures momentum and trend direction. "
            "Bollinger Bands: Volatility bands around a moving average; price near upper band = overbought."
        ),
        keywords=["technical analysis", "moving average", "golden cross", "death cross", "rsi", "macd", "support", "resistance", "bollinger"]
    ),

    # ═══════════════════════════════════════════════════════════
    # PERSONAL FINANCE
    # ═══════════════════════════════════════════════════════════
    KnowledgeChunk(
        chunk_id="pf_001", category="Personal Finance",
        title="Emergency Fund and Financial Planning Basics",
        content=(
            "Financial planning fundamentals: (1) Emergency Fund — save 6-12 months of living expenses in "
            "liquid assets (savings account, liquid mutual fund) before investing. "
            "(2) Insurance first — adequate term life insurance (10-15x annual income) and health insurance "
            "(Rs 10-20 lakh family cover minimum) before any investments. "
            "(3) Clear high-interest debt (credit cards at 24-42% APR, personal loans at 10-16%) before investing. "
            "(4) 50-30-20 Rule: 50% of income for needs, 30% for wants, 20% for savings/investments. "
            "(5) Start investing early — compounding over 30 years: Rs 5,000/month SIP at 12% CAGR grows "
            "to ~Rs 1.76 crore. Starting 10 years later yields only ~Rs 50 lakhs. "
            "(6) NPS (National Pension System) offers additional Rs 50,000 tax deduction under 80CCD(1B)."
        ),
        keywords=["emergency fund", "financial planning", "insurance", "term life", "health insurance", "50-30-20", "compounding", "nps"]
    ),
    KnowledgeChunk(
        chunk_id="pf_002", category="Personal Finance",
        title="Retirement Planning and FIRE Movement",
        content=(
            "Retirement corpus estimation: Annual Expenses × 25 (based on 4% safe withdrawal rate). "
            "For Rs 50,000 monthly expenses: need ~Rs 1.5 crore corpus (adjusted for inflation). "
            "FIRE (Financial Independence, Retire Early): Save 50-70% of income, invest aggressively, "
            "retire in 10-15 years. Variants: LeanFIRE (minimal expenses), FatFIRE (comfortable lifestyle), "
            "CoastFIRE (save enough that compounding handles it; work optionally). "
            "Investment mix for retirement: Equity-heavy in early years (80% equity, 20% debt), "
            "gradually shift to conservative as retirement approaches (40% equity, 60% debt/bonds). "
            "PPF (Public Provident Fund): 7.1% tax-free returns, 15-year lock-in, Rs 1.5 lakh annual limit. "
            "EPF (Employee Provident Fund): 8.25% returns, employer-matched, tax-free on retirement. "
            "Inflation-adjusted planning: Use real returns (nominal return - inflation rate)."
        ),
        keywords=["retirement", "fire", "corpus", "4% rule", "ppf", "epf", "pension", "inflation", "withdrawal"]
    ),

    # ═══════════════════════════════════════════════════════════
    # HALLUCINATION-SPECIFIC KNOWLEDGE
    # ═══════════════════════════════════════════════════════════
    KnowledgeChunk(
        chunk_id="hall_001", category="AI & Hallucination",
        title="What is Financial Hallucination in AI",
        content=(
            "Financial hallucination occurs when an AI language model fabricates, misquotes, or inaccurately "
            "states financial data that doesn't match audited regulatory filings. Common types: "
            "(1) Revenue/Profit Confabulation — stating an incorrect number (e.g., claiming TCS revenue "
            "was Rs 65,000 Cr when audited filings show Rs 61,237 Cr). "
            "(2) Temporal Basis Flip — mixing up Year-over-Year (YoY) and Quarter-over-Quarter (QoQ) growth "
            "rates, which have vastly different values. "
            "(3) Peer Ranking Errors — incorrectly stating one company has higher margins than another. "
            "(4) Outdated Data — using last year's figures for current queries. "
            "(5) Ratio Fabrication — inventing financial ratios that aren't supported by filings. "
            "Mitigation: RAG (Retrieval-Augmented Generation) grounds answers in verified source documents, "
            "cross-verification against audited databases, and explicit citation of filing sources."
        ),
        keywords=["hallucination", "confabulation", "ai", "llm", "rag", "grounding", "verification", "fabrication"]
    ),
    KnowledgeChunk(
        chunk_id="hall_002", category="AI & Hallucination",
        title="RAG (Retrieval-Augmented Generation) for Finance",
        content=(
            "RAG combines a retrieval system with a generative model to produce grounded, factual answers. "
            "Architecture: (1) Knowledge Base — curated corpus of financial facts, definitions, regulatory "
            "filings, and audited metrics. (2) Retriever — when a query arrives, finds the most relevant "
            "knowledge chunks using TF-IDF, BM25, or embedding similarity. (3) Generator — synthesizes a "
            "natural language answer grounded in the retrieved context, with citations. "
            "Benefits for finance: Reduces hallucinated numbers, provides traceable sources, ensures "
            "answers are based on audited data rather than model memorization. "
            "Our system uses: TF-IDF cosine similarity retrieval over 40+ curated financial knowledge chunks, "
            "combined with audited metric records from 13 major companies (TCS, Infosys, Reliance, HDFC, "
            "Wipro, Tata Motors, Apple, Microsoft, Google, Nvidia, Amazon, Meta, Tesla)."
        ),
        keywords=["rag", "retrieval augmented generation", "knowledge base", "retrieval", "grounding", "tf-idf", "embedding"]
    ),
]


class FinancialRAGEngine:
    """
    TF-IDF based Retrieval-Augmented Generation engine for financial knowledge.
    Retrieves relevant knowledge chunks and synthesizes grounded answers.
    """

    def __init__(self, extra_chunks: Optional[List[KnowledgeChunk]] = None):
        self.chunks = list(FINANCIAL_KNOWLEDGE_CORPUS)
        if extra_chunks:
            self.chunks.extend(extra_chunks)
        # Build TF-IDF index
        self._doc_freqs: Dict[str, int] = {}
        self._tf_vectors: List[Dict[str, float]] = []
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer: lowercase, alphanumeric tokens, min length 2."""
        return [w for w in re.findall(r'[a-z][a-z0-9/\-]+', text.lower()) if len(w) >= 2]

    def _build_index(self):
        """Build TF-IDF vectors for all chunks."""
        # Combine title, content, and keywords for each chunk
        doc_texts = []
        for c in self.chunks:
            text = f"{c.title} {c.content} {' '.join(c.keywords)}"
            doc_texts.append(text)

        # Compute document frequencies
        self._doc_freqs = {}
        all_doc_tokens = []
        for text in doc_texts:
            tokens = self._tokenize(text)
            all_doc_tokens.append(tokens)
            unique_tokens = set(tokens)
            for tok in unique_tokens:
                self._doc_freqs[tok] = self._doc_freqs.get(tok, 0) + 1

        # Compute TF-IDF vectors
        n_docs = len(self.chunks)
        self._tf_vectors = []
        for tokens in all_doc_tokens:
            tf = Counter(tokens)
            total = len(tokens) or 1
            tfidf_vec = {}
            for tok, count in tf.items():
                tf_val = count / total
                idf_val = math.log((n_docs + 1) / (self._doc_freqs.get(tok, 0) + 1)) + 1
                tfidf_vec[tok] = tf_val * idf_val
            self._tf_vectors.append(tfidf_vec)

    def _cosine_similarity(self, vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        """Compute cosine similarity between two sparse vectors."""
        common_keys = set(vec_a.keys()) & set(vec_b.keys())
        if not common_keys:
            return 0.0
        dot = sum(vec_a[k] * vec_b[k] for k in common_keys)
        norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
        norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[KnowledgeChunk, float]]:
        """
        Retrieve the top-k most relevant knowledge chunks for a query.
        Returns list of (chunk, similarity_score) tuples.
        """
        # Build query vector
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        n_docs = len(self.chunks)
        query_tf = Counter(query_tokens)
        total = len(query_tokens)
        query_vec = {}
        for tok, count in query_tf.items():
            tf_val = count / total
            idf_val = math.log((n_docs + 1) / (self._doc_freqs.get(tok, 0) + 1)) + 1
            query_vec[tok] = tf_val * idf_val

        # Also boost keyword-level matches
        scores = []
        for i, chunk in enumerate(self.chunks):
            sim = self._cosine_similarity(query_vec, self._tf_vectors[i])
            # Keyword bonus: if query tokens directly match chunk keywords
            kw_set = set(kw.lower() for kw in chunk.keywords)
            q_lower = query.lower()
            kw_bonus = sum(0.15 for kw in kw_set if kw in q_lower)
            final_score = sim + kw_bonus
            scores.append((chunk, final_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def answer_query(
        self,
        query: str,
        company_context: Optional[str] = None,
        company_data: Optional[Dict[str, str]] = None,
        top_k: int = 3,
    ) -> Dict[str, any]:
        """
        RAG-powered answer generation.
        Retrieves relevant knowledge chunks, then synthesizes a grounded answer
        combining retrieved financial concepts with company-specific data.

        Returns dict with: answer, retrieved_chunks, confidence, sources
        """
        retrieved = self.retrieve(query, top_k=top_k)

        if not retrieved or retrieved[0][1] < 0.05:
            return {
                "answer": (
                    "I don't have enough verified financial knowledge to answer this question accurately. "
                    "Please try rephrasing or ask about financial concepts, ratios, investment strategies, "
                    "market mechanics, or specific company analysis."
                ),
                "retrieved_chunks": [],
                "confidence": 0.0,
                "sources": [],
                "grounded": False,
            }

        # Build context from retrieved chunks
        context_parts = []
        sources = []
        for chunk, score in retrieved:
            if score > 0.02:
                context_parts.append(f"[{chunk.category}: {chunk.title}]\n{chunk.content}")
                sources.append({"title": chunk.title, "category": chunk.category,
                                "relevance": round(score, 3), "source": chunk.source})

        # Synthesize answer from retrieved context
        answer = self._synthesize_answer(query, context_parts, company_context, company_data)
        confidence = min(0.95, retrieved[0][1] * 2.5 + 0.3)

        return {
            "answer": answer,
            "retrieved_chunks": [(c.title, round(s, 3)) for c, s in retrieved if s > 0.02],
            "confidence": round(confidence, 2),
            "sources": sources,
            "grounded": True,
        }

    def _synthesize_answer(
        self,
        query: str,
        context_parts: List[str],
        company_context: Optional[str],
        company_data: Optional[Dict[str, str]],
    ) -> str:
        """
        Synthesize a natural language answer grounded in retrieved context.
        Uses template-based generation with context injection.
        """
        q_lower = query.lower()

        # Join the most relevant context
        combined_context = "\n\n".join(context_parts[:3])

        # Extract the most relevant paragraph from context
        # Split context into sentences and find the most relevant ones
        all_sentences = []
        for part in context_parts[:3]:
            # Get the content part (after the header)
            lines = part.split("\n", 1)
            content = lines[1] if len(lines) > 1 else lines[0]
            sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', content) if len(s.strip()) > 20]
            all_sentences.extend(sents)

        # Score sentences by relevance to query
        query_words = set(self._tokenize(query))
        scored_sentences = []
        for sent in all_sentences:
            sent_words = set(self._tokenize(sent))
            overlap = len(query_words & sent_words)
            scored_sentences.append((sent, overlap))
        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        # Pick top relevant sentences (3-6 for a good answer)
        top_sentences = [s for s, _ in scored_sentences[:6]]

        # Build the answer
        answer_parts = []

        # Header based on query type
        if any(w in q_lower for w in ["what is", "what are", "explain", "define", "meaning", "tell me about"]):
            # Definitional query
            topic = re.sub(r'^(what\s+is|what\s+are|explain|define|tell\s+me\s+about)\s+', '', q_lower, flags=re.IGNORECASE).strip().rstrip('?')
            answer_parts.append(f"**{topic.title()} — Explained:**\n")
        elif any(w in q_lower for w in ["how to", "how do", "how can", "should i", "when to"]):
            answer_parts.append("**Here's what the financial knowledge base says:**\n")
        elif any(w in q_lower for w in ["compare", "vs", "versus", "difference", "better"]):
            answer_parts.append("**Comparative Analysis:**\n")
        else:
            answer_parts.append("**Financial Analysis Response:**\n")

        # Add the most relevant content
        if top_sentences:
            answer_parts.append("\n".join(top_sentences[:4]))
        else:
            # Fallback: use first chunk content directly
            answer_parts.append(context_parts[0].split("\n", 1)[-1][:600])

        # Add company-specific data if available
        if company_data:
            answer_parts.append("\n\n**Company-Specific Data:**")
            for key, val in list(company_data.items())[:6]:
                answer_parts.append(f"• {key}: {val}")

        # Add source attribution
        answer_parts.append("\n\n*This answer is grounded in the RAG financial knowledge base — not generated from model memory.*")

        return "\n".join(answer_parts)

    def add_knowledge(self, chunk: KnowledgeChunk):
        """Add a new knowledge chunk and rebuild index."""
        self.chunks.append(chunk)
        self._build_index()

    def get_all_categories(self) -> List[str]:
        """Return all unique categories in the knowledge base."""
        return sorted(set(c.category for c in self.chunks))

    def get_chunks_by_category(self, category: str) -> List[KnowledgeChunk]:
        """Return all chunks in a given category."""
        return [c for c in self.chunks if c.category.lower() == category.lower()]
