# hallucination-detector
Production RAG hallucination detection + self-healing pipeline — 5 checks, 3 healing strategies, zero external APIs

# hallucination-detector

> A production RAG hallucination detection and self-healing pipeline — 5 checks, 3 healing strategies, zero external APIs.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-70%20passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()
[![spaCy](https://img.shields.io/badge/spaCy-optional-orange)](https://spacy.io/)

Most RAG tutorials stop at: retrieve documents, stuff them into a prompt, call the model.  
This library handles what comes next — catching when the model contradicts its own retrieved sources, fixing the answer before it reaches the user, and routing the result based on a quality score.

Read the full write-up on Towards Data Science →  
**[RAG Hallucinates — I Built a Self-Healing Layer That Fixes It in Real Time](https://towardsdatascience.com/rag-hallucinates-i-built-a-self-healing-layer-that-fixes-it-in-real-time/)**

---

## The Problem

RAG retrieves the correct document. The LLM still generates the wrong answer.

In my system, the model repeatedly returned answers that directly contradicted the retrieved context — for example, stating a 30-day return policy when the source clearly specified 14 days. Retrieval was working as expected. The failure happened at generation.

There was no error, no alert, and nothing in the logs to indicate the response was wrong.

This library is built to detect and fix that class of failure before it reaches the user.

---

## Pipeline

```
LLM Output
    │
    ▼
┌─────────────────────────────┐
│  Check 1: Confidence Score  │  Is the answer assertive?
├─────────────────────────────┤
│  Check 2: Faithfulness      │  Is it grounded in sources?
├─────────────────────────────┤
│  Check 3: Contradiction     │  Does it conflict with context?
├─────────────────────────────┤
│  Check 4: Entity Check      │  Are names and citations real?
├─────────────────────────────┤
│  Check 5: Drift Monitor     │  Has this answer changed over time?
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│     Quality Score Engine    │
│  faithfulness  × 0.40       │
│  consistency   × 0.30       │
│  confidence    × 0.20       │
│  latency       × 0.10       │
│  drift penalty − 0.20       │
└─────────────────────────────┘
    │
    ├──► ACCEPT         (score ≥ 0.75, no healing needed)
    ├──► HEALED_ACCEPT  (healing applied, re-inspection passed)
    ├──► FALLBACK       (score < 0.50, not healed)
    └──► DISCARD        (healing failed, safe decline served)
```

---

## What It Does

| Component | Job |
|---|---|
| `ConfidenceScorer` | Detects assertive answers with low grounding — the most dangerous pattern |
| `FaithfulnessScorer` | Checks what fraction of claims are traceable to retrieved context |
| `ContradictionDetector` | Catches numeric, negation, and temporal conflicts with source documents |
| `EntityHallucinationDetector` | Flags person names, citations, and orgs absent from all context chunks |
| `AnswerDriftMonitor` | Tracks answer fingerprints in SQLite — detects silent degradation over time |
| `HallucinationHealer` | Fixes wrong answers in-place before delivery; serves safe decline if healing fails |
| `QualityScore` | Weighted composite score routing answers to one of four delivery tiers |

---

## Installation

```bash
git clone https://github.com/Emmimal/hallucination-detector.git
cd hallucination-detector
pip install spacy
python -m spacy download en_core_web_sm
```

No other dependencies. SQLite is standard library. spaCy is used for named entity recognition — without it, the system falls back to regex NER automatically with a warning.

---

## Quick Start

```python
from hallucination_detector import (
    HallucinationDetector, HallucinationHealer,
    DetectorConfig, QualityScore
)

config   = DetectorConfig(db_path="drift.db", log_flagged=True)
detector = HallucinationDetector(config)
healer   = HallucinationHealer(detector)

# Run on every LLM answer before delivery
report = detector.inspect(question, context_chunks, llm_answer)
score  = QualityScore.compute(report)

if score.routing == "accept":
    return llm_answer

# Attempt in-place healing
result = healer.heal(question, context_chunks, llm_answer, report)
score  = QualityScore.compute(report, healing_result=result)

if score.routing == "healed_accept":
    return result.healed_answer

return fallback_response
```

---

## Running the Demos

Five production scenarios covering every failure mode and healing strategy:

```bash
python demo.py
```

| Demo | Failure Mode | Healing Strategy |
|------|-------------|-----------------|
| 1 | Confident lie — 30 days vs 14 days policy | Contradiction patch |
| 2 | Hallucinated citation — Dr. James Harrison | Grounding rewrite |
| 3 | Billing contradiction — $10/month vs $120/year | Contradiction patch + billing normalization |
| 4 | Silent price drift — $49.99 → $39.99 over time | Grounding rewrite |
| 5 | Clean answer — no healing needed | Passes through unchanged |

Each demo prints the full detect → score → heal output so you can see exactly what changed and why.

---

## Running the Tests

```bash
pip install pytest
pytest tests/ -v
```

Expected output:

```
TestConfidenceScorer             5 passed
TestFaithfulnessScorer           5 passed
TestContradictionDetector        7 passed
TestEntityHallucinationDetector  5 passed
TestAnswerDriftMonitor           6 passed
TestHallucinationDetector       24 passed
TestQualityScore                18 passed

70 passed
```

Every named production failure has a test assertion. The thread-safety test runs 20 concurrent `inspect()` calls. The SQLite persistence test writes drift history with one monitor instance and detects it with a fresh instance on the same file — because that is exactly what happens across rolling deployments.

---

## Async (FastAPI)

```python
report = await detector.ainspect(question, context_chunks, llm_answer)
```

`ainspect()` runs the full pipeline in a thread pool executor — safe for FastAPI and any async framework. Concurrent calls are fully thread-safe; the 20-thread test covers this.

---

## Block on Critical Risk

```python
if report.is_hallucinating:
    raise HallucinationBlocked(report)
    # HallucinationBlocked.report carries the full dict for your monitoring layer
```

---

## Strict Mode (Legal / Medical)

```python
config = DetectorConfig(
    faithfulness_threshold=0.70,          # up from 0.50
    faithfulness_overlap_threshold=0.70,  # up from 0.40
    confidence_threshold=0.60,            # down from 0.75 — flag earlier
    drift_threshold=0.25,                 # down from 0.35 — more sensitive
    db_path="drift_production.db",
    log_flagged=True,
)
```

---

## Structured JSON Logging

```python
from hallucination_detector import configure_logging
import logging

configure_logging(level=logging.WARNING)
# Every flagged response emits a structured JSON WARNING with the full report
```

---

## Configuration Reference

```python
DetectorConfig(
    confidence_threshold=0.75,          # Flag when answer sounds this assertive
    faithfulness_threshold=0.50,        # Flag when fewer than this fraction of claims are grounded
    faithfulness_overlap_threshold=0.40,# Keyword overlap required per claim to count as grounded
    drift_threshold=0.35,               # Similarity delta above which drift is flagged
    db_path="hallucination_drift.db",   # SQLite file for drift history (":memory:" for tests)
    window_size=50,                     # Past answers retained per question
    log_flagged=True,                   # Emit WARNING log when is_hallucinating=True
)
```

**Tuning thresholds:**

| Domain | `confidence_threshold` | `faithfulness_overlap_threshold` | `drift_threshold` |
|---|---|---|---|
| General | 0.75 | 0.40 | 0.35 |
| High-stakes (legal, medical) | 0.60 | 0.70 | 0.25 |
| Noisy / conversational | 0.80 | 0.35 | 0.45 |

---

## Healing Strategies

| Strategy | Triggered When | What It Does |
|---|---|---|
| `contradiction_patch` | Numeric or billing contradiction found | Replaces wrong values in-place from context; falls back to grounding rewrite if faithfulness remains below 0.50 |
| `entity_scrub` | Hallucinated names or citations | Removes offending sentences; appends transparency note |
| `grounding_rewrite` | Faithfulness < 0.30 or drift detected | Rebuilds answer from top context sentences with context-derived prefix |
| `safe_decline` | Healing fails re-inspection | Serves a safe decline rather than a wrong answer |

---

## Confidence Recalibration After Healing

| Strategy | Formula | Rationale |
|---|---|---|
| `contradiction_patch` | `min(original + 0.15, 0.80)` | Deterministic fix from verified source |
| `entity_scrub` | `original × 0.85` | Remaining text is still the model's output |
| `grounding_rewrite` | Re-run `ConfidenceScorer` | Hedging prefix ("According to…") scores lower naturally |

---

## Performance

Measured on Python 3.12, CPU only, no GPU:

| Operation | Latency | Notes |
|---|---|---|
| Confidence scoring | < 1ms | Regex pattern matching |
| Faithfulness scoring | ~2ms | Keyword overlap calculation |
| Contradiction detection | ~1ms | Regex + number extraction |
| Entity detection — spaCy | ~45ms | `en_core_web_sm` NER |
| Entity detection — regex | < 1ms | Fallback path, no spaCy required |
| Drift record + check | ~3ms | SQLite write + similarity query |
| Full `inspect()` — regex NER | < 10ms | Pure Python path |
| Full `inspect()` — spaCy NER | < 50ms | Production path |

If you need sub-10ms end-to-end, the regex NER fallback is a one-line config change. You trade some entity detection precision for latency.

---

---

## Financial Features: Financial Hallucination Detector & AI Research Agent

LLMs can confidently state incorrect financial figures, revenue numbers, and margins:
> **User:** *"What was TCS's revenue in Q4?"*  
> **LLM:** *"TCS reported ₹65,000 crore."*  
> **Audited Truth:** *"Consolidated revenue stood at ₹61,237 crore."*

The financial extension (`financial/`) extracts quantitative claims, cross-verifies them against official regulatory filings (SEC, NSE/BSE) and live market feeds, computes claim-level hallucination probabilities, and provides in-place self-healing with verified footnote citations.

### Financial Detection Pipeline

```
             LLM Answer
                 │
          Extract Claims  (Entities, Metrics, Fiscal Periods, Currency & Scales)
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
  Financial Reports    Market Data (yfinance / fundamental cache)
       │                   │
       └─────────┬─────────┘
                 ▼
          Claim Verification (Exact match / Rounding tolerance / Currency conversion)
                 │
       ┌─────────┼─────────┐
       ▼         ▼         ▼
   Supported  Contradicted  Unknown
       │         │         │
       └─────────┼─────────┘
                 ▼
          Hallucination Score & Self-Healed Answer
```

### AI Financial Research Agent (Mini Bloomberg)

Autonomously analyzes companies across 4 core scorecards:
- **Financial Health** (0-100) - Debt/Equity, current ratio, cash generation
- **Growth** (0-100) - YoY revenue, deal book TCV, earnings momentum
- **Profitability** (0-100) - Operating margin, gross margin, return on equity (ROE)
- **Risk** (0-100) - Beta volatility, valuation stretch, concentration

Features included:
- **Multi-Company Coverage**: Full audited ground truth for 13 major Indian and global giants (TCS, INFOSYS, RELIANCE, HDFC, WIPRO, TATAMOTORS, APPLE, MICROSOFT, GOOGLE, NVIDIA, AMAZON, META, TESLA), plus dynamic parsing for any arbitrary company or ticker.
- **Natural Language Financial Q&A**: Answers company valuation questions, cross-company peer comparisons, specific metric lookups, and financial concepts.
- **Quantitative Forensics**: Piotroski 9-Point F-Score, Altman Z-Score solvency model, and Cash Conversion Ratio.
- **Peer Benchmarking**: Cross-company margin and multiple ranking across 6 industry sectors.
- **Temporal Growth Verifier**: Catches YoY vs QoQ basis slips and quarter-to-annual confabulations.
- **Document & PDF Ingestion**: Dynamic regex & statement parser indexing uploaded filings directly into memory.
- **Self-Audited Reports**: The agent passes its own synthesized reports through the FinancialHallucinationDetector to ensure 0% hallucination rate before delivery.

### Quick Start (Financial Features)

Run the console scenarios:
`ash
python demo_financial.py --scenario 1  # TCS Rs 65,000 Cr revenue hallucination
python demo_financial.py --scenario all
`

Launch the interactive Streamlit Web Dashboard:
`ash
streamlit run financial/app.py
`

Run financial test suite (30 tests):
`ash
pytest test_financial.py -v
`

## Project Structure

`
hallucination-detector/
├── hallucination_detector.py          # Core RAG detector, healer, quality scorer
├── demo.py                            # Five general production scenarios
├── test_hallucination_detector.py     # 70 unit and integration tests
├── demo_financial.py                  # Financial demonstration scenarios
├── test_financial.py                  # Financial test suite (30 unit & integration tests)
├── requirements.txt
├── README.md
└── financial/                         # Financial extension package
    ├── __init__.py                    # Public API exports
    ├── config.py                      # Financial tolerances, metrics, scale & currency mappings
    ├── claim_extractor.py             # Financial regex & NLP claim extraction
    ├── sample_data.py                 # Audited ground truth filings (13 pre-loaded companies)
    ├── data_sources.py                # FinancialReportStore & MarketDataProvider (yfinance + cache)
    ├── detector.py                    # FinancialHallucinationDetector
    ├── healer.py                      # FinancialHealer (in-place patching & footnote citations)
    ├── sentiment.py                   # FinSentimentAnalyzer (financial lexicon & transcript sentiment)
    ├── forensics.py                   # Piotroski F-Score, Altman Z-Score & Cash Conversion Ratio
    ├── peer_benchmark.py              # Sector peer comparison matrix & comparative claim auditor
    ├── temporal_verifier.py           # Temporal YoY vs QoQ basis flip detector
    ├── rag_indexer.py                 # Regulatory statement & PDF ingestion engine
    ├── research_agent.py              # FinancialResearchAgent (mini Bloomberg analyst)
    └── app.py                         # Interactive 5-tab Streamlit Web Dashboard
`
---

## When to Use This

**Worth it when you have:**
- A RAG system where wrong answers have real consequences (customer support, legal, medical, finance)
- Multi-turn deployments where answer drift is a risk
- Any production system where you can't afford six weeks of a hallucination nobody notices

**Skip it when you have:**
- Single-turn queries against a small, fixed knowledge base
- Hard latency requirements under 10ms (use regex NER path) or under 1ms (this is not the right tool)
- A fully deterministic retrieval domain where keyword matching is sufficient and auditable

---

## Known Limits

- **Confident, consistent hallucinations.** If the model always says "30 days" and the context also says "30 days," all checks pass. This system assumes retrieved context is correct. It cannot detect bad retrieval — only answers that deviate from what was retrieved.
- **Semantic paraphrase.** At 40% keyword overlap, a carefully phrased fabrication can pass the faithfulness check. Raise `faithfulness_overlap_threshold` to 0.70 for high-stakes domains.
- **Drift as a trailing indicator.** The drift monitor requires at least three prior answers before it fires. Some bad answers will be served before detection kicks in.
- **Token estimation.** Uses 1 token ≈ 4 characters. Misfires for code and non-Latin scripts.

---

## Related

This library pairs naturally with [context-engine](https://github.com/Emmimal/context-engine) — which controls what enters the context window — and [hallucination-detector](https://github.com/Emmimal/hallucination-detector) — which checks what comes out of the LLM. Together they cover both sides of the RAG reliability problem.

---

## License

MIT
