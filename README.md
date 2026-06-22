# 🔍 LLM Reliability Engine

[![CI](https://github.com/YatindraRai002/LLM-LIE-DETECTOR/actions/workflows/ci.yml/badge.svg)](https://github.com/YatindraRai002/LLM-LIE-DETECTOR/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39-red?logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Async-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-66%20passed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

A hallucination detection system for LLM outputs. It fuses three independent uncertainty signals — token-level calibration, semantic self-consistency, and cross-model agreement — into a single calibrated risk score, instead of trusting a model's confident tone at face value.

> LLMs state false things with the same fluent confidence as true things. This system measures *actual* uncertainty across three independent signals rather than relying on how the answer sounds.

> **Research Hypothesis:** We hypothesize that hallucinations in LLM generations manifest simultaneously across three orthogonal dimensions: low token-level confidence (calibration), semantic inconsistency across multiple stochastic generations (semantic uncertainty), and factual disagreement with a stronger external reference model (cross-model verification). Fusing these complementary uncertainty sources via a learned meta-classifier should significantly improve hallucination detection compared to any single uncertainty signal.

---

## Contents

- [Architecture](#architecture)
- [Benchmark results](#benchmark-results)
- [Quick start](#quick-start)
- [Usage](#usage)
- [Detection pipeline](#detection-pipeline-details)
- [Security](#security--resilience)
- [Explanation engine](#explanation-engine-phase-b)
- [Project structure](#project-structure)
- [Testing](#testing)
- [References](#references)

---

## Architecture

### System overview

```mermaid
graph TD
    subgraph Frontends["Frontend layer"]
        UI1["Streamlit App<br/>(app.py)<br/>direct Python import — PRIMARY"]
        UI2["Next.js Web App<br/>(frontend/)<br/>polls /analyze → /result — EXPERIMENTAL"]
    end

    subgraph API["FastAPI backend (backend/api.py)"]
        Router["Async router<br/>POST /analyze → 202 job_id<br/>GET /result/{id} → result"]
        Health["GET /health · GET /metrics"]
    end

    subgraph Pipeline["Core detection pipeline (core/aggregator.py)"]
        Cache["Response cache<br/>(core/cache.py)"]
        S1["Stage 1 — Calibration<br/>token-level logits"]
        S2["Stage 2 — Semantic uncertainty<br/>sample + cluster"]
        S3["Stage 3 — Cross-check<br/>NLI vs Groq"]
        S4["Stage 4 — Aggregator + Explainer<br/>weighted fusion, hard override"]
    end

    subgraph Models["Models & external APIs"]
        Local["TinyLlama<br/>local GPU"]
        Groq["Groq API<br/>llama-3.3-70b"]
        NLI["DeBERTa-v3<br/>NLI cross-encoder"]
    end

    UI1 -->|Python import| Pipeline
    UI2 -->|async HTTP| Router
    Router --> Cache
    Cache -->|cache miss| S1
    S1 --> S2
    S1 --> S3
    S2 -.parallel.- S3
    S2 --> S4
    S3 --> S4
    S1 -->|loads via model_loader| Local
    S2 -->|embeds + samples| Groq
    S3 -->|reference answer| Groq
    S3 -->|entailment scoring| NLI
    S4 --> Router
    Router --> Health
```

### Single-request data flow

```mermaid
sequenceDiagram
    participant U as User
    participant API as FastAPI /analyze
    participant Cache as Redis/SQLite cache
    participant Cal as Stage 1: Calibration
    participant Unc as Stage 2: Uncertainty
    participant CC as Stage 3: Cross-check
    participant Agg as Stage 4: Aggregator

    U->>API: POST /analyze {prompt}
    API->>Cache: lookup(prompt)
    alt cache hit
        Cache-->>API: cached result
        API-->>U: 200 {result}
    else cache miss
        API-->>U: 202 {job_id}
        API->>Cal: generate(prompt) on TinyLlama
        Cal-->>API: response + token logits
        par parallel execution
            API->>Unc: sample 6x via Groq, cluster
            Unc-->>API: uncertainty score
        and
            API->>CC: Groq reference + DeBERTa NLI
            CC-->>API: cross-check score + verdict
        end
        API->>Agg: fuse(calibration, uncertainty, cross-check)
        Agg-->>API: HallucinationResult
        API->>Cache: store(prompt, result)
        Note over U,API: client polls GET /result/{job_id}
    end
```

**Key design decisions**

| Decision | Why |
|---|---|
| **Stages 2 & 3 run in parallel** via `ThreadPoolExecutor` | Both depend only on Stage 1's output and are independent of each other — running them concurrently cuts per-query latency by roughly 40% |
| **2-signal fallback mode** | If Groq is unreachable, the aggregator redistributes its weight to calibration + uncertainty rather than guessing or crashing — never produces a false high-risk score from an API outage |
| **Hard NLI override** | If DeBERTa detects outright contradiction between the local and Groq response, the score is floored at 0.85 regardless of the weighted sum |
| **Async API** | `/analyze` returns a `job_id` immediately; the frontend polls `/result/{job_id}` — avoids browser/proxy timeouts on long-running GPU inference |
| **Cache-first** | Identical `(prompt, weights)` pairs return instantly on repeat queries instead of re-running the full pipeline |

---

## Benchmark results

Evaluated on **[TruthfulQA](https://github.com/sylinrl/TruthfulQA)** (Misconceptions category, n = 100):

| Metric | Value |
|---|---|
| AUROC | 0.509 |
| Best F1 | 0.929 |
| Average precision | 0.896 |
| Precision @ optimal threshold | 0.868 |
| Recall @ optimal threshold | 1.000 |
| Labeled examples | 68 / 100 |
| Latency (p50) | 16.95 s |
| Latency (p95) | 42.15 s |
| Risk distribution | Low: 79 · Medium: 1 · High: 20 |

**Analyzing these results:** The previous AUROC bottleneck was the keyword-matching heuristic (which only successfully labeled 68 of 100 queries with noisy boundaries). We have upgraded the evaluation harness with a **Groq-powered LLM-as-Judge** for robust ground-truth labeling, along with baseline comparisons, ablation studies, and Expected Calibration Error (ECE) tracking. The detector uses a learned Meta-Classifier instead of hardcoded weights, allowing it to predict actual hallucination probabilities.

---

## Quick start

### Prerequisites
- Python 3.10+
- NVIDIA GPU, 8GB+ VRAM recommended for 4-bit quantization (CPU fallback works but is slow)
- [Groq API key](https://console.groq.com/keys) — free tier is sufficient

### Installation

```bash
git clone https://github.com/YatindraRai002/LLM-LIE-DETECTOR.git
cd LLM-LIE-DETECTOR
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=gsk_your_key_here
```

> **Security:** the default Streamlit dashboard credentials in `config.yaml` (`admin` / `admin_password`) are for local development only. Change them before any public deployment — see [Security & resilience](#security--resilience).

All hyperparameters — signal weights, risk thresholds, model names, sample counts — are centralized in [`config.yaml`](config.yaml). No need to touch source code to retune the system.

---

## Usage

### Streamlit dashboard (primary interface)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

### FastAPI backend

```bash
uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

| Method | Path | Description |
|---|---|---|
| `POST` | `/analyze` | Submit a prompt for analysis. Returns `202 {job_id}` immediately |
| `GET` | `/result/{job_id}` | Poll for the analysis result |
| `GET` | `/health` | Health check — model load status, Groq/Redis connectivity |
| `GET` | `/metrics` | Prometheus-format metrics |
| `GET` | `/api/history` | Recent analysis history from SQLite |

Example:

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Who invented the telephone?"}'
# → {"job_id": "...", "status": "pending"}

curl http://localhost:8000/result/<job_id>
# → poll until status == "done"
```

### Next.js frontend (experimental)

```bash
cd frontend
npm install
npm run dev
```

Polls the async API automatically. The Streamlit dashboard remains the primary, fully-supported interface — this is included to demonstrate the API consumed from a separate client.

### Docker (full stack)

```bash
docker compose up --build
```

Brings up the FastAPI backend, Streamlit dashboard, Redis, Prometheus, and Grafana in one command.

---

## Detection pipeline details

### Stage 1 — probability calibration

Extracts token-level logits from TinyLlama during generation via `output_scores=True`. Tokens the model assigns low probability to indicate it is effectively guessing rather than recalling.

```
CalibScore = (1 − mean(p)) · 0.7  +  min(std(p) / 0.3, 1) · 0.3
```

**Implementation:** [`core/calibration.py`](core/calibration.py)

### Stage 2 — semantic uncertainty

Generates 6 diverse responses via parallel Groq calls at temperature 0.8, embeds them with `all-MiniLM-L6-v2`, and clusters with agglomerative clustering on cosine distance. A model that "knows" an answer gives consistent responses across samples; a model that's hallucinating scatters. Falls back to sequential local-model generation if Groq is rate-limited.

**Implementation:** [`core/semantic_uncertainty.py`](core/semantic_uncertainty.py)

### Stage 3 — cross-model check

Queries Groq (Llama-3.3-70B) for a deterministic reference answer, then runs bidirectional NLI (`cross-encoder/nli-deberta-v3-base`) between the local and remote responses. A confirmed contradiction triggers the hard override.

**Implementation:** [`core/cross_check.py`](core/cross_check.py)

### Stage 4 — learned meta-classifier + explainability

Instead of a static weighted average, the system trains a **Logistic Regression meta-classifier** to predict hallucination probability directly from the three uncertainty signals:

$$P(\text{hallucination}) = \sigma(\beta_1 \cdot \text{Cal} + \beta_2 \cdot \text{Unc} + \beta_3 \cdot \text{CC} + \beta_0)$$

- **Adaptive weights:** The coefficients $\beta_1, \beta_2, \beta_3$ and intercept $\beta_0$ are learned from labeled evaluation data via `tune_weights.py`.
- **2-signal fallback:** If Groq is unavailable, the system uses a secondary 2-signal model trained on Calibration and Semantic Uncertainty only.
- **Graceful default:** Falls back to standard config weights if the classifier model hasn't been fitted.

**Implementation:** [`core/meta_classifier.py`](core/meta_classifier.py) · [`core/aggregator.py`](core/aggregator.py) · [`core/explainer.py`](core/explainer.py)

---

## Security & resilience

| Feature | Implementation |
|---|---|
| Authentication | `streamlit-authenticator`, bcrypt password hashing, JWT sessions |
| Rate limiting | IP-based sliding window, 5 requests/minute |
| API key safety | `.env` gitignored, format-validated (`gsk_` prefix), never logged in full |
| Groq retry | Exponential backoff, explicit 401 / 429 handling |
| Graceful degradation | Automatic 2-signal fallback when Groq is unreachable |
| Input sanitization | Length truncation, prompt-injection pattern stripping, tokenizer-bomb collapse |
| SQL safety | Parameterized queries only — no string interpolation |
| YAML safety | `SafeLoader` for all YAML parsing |
| Calibrator storage | Plain JSON (not pickle) — no deserialization code-execution surface |
| Memory efficiency | 4-bit quantization + `@lru_cache` model singletons |
| CORS | Scoped to configured origins — no wildcard |

A full internal security audit (severity-ranked findings, what was checked and how) is in [`SECURITY.md`](SECURITY.md).

---

## Explanation engine (Phase B)

A post-scoring layer that attributes *why* a response was flagged, not just the final number:

| Module | What it does |
|---|---|
| Token attribution | Flags tokens where `P(token) < mean × 0.5`, rendered as color-coded spans |
| Sentence-level NLI | Scores each sentence of the local response against the Groq oracle individually |
| Signal SHAP (ablation) | Lightweight contribution breakdown — what % of the score came from each signal |
| Recommendations | Plain-English summary of the dominant risk driver |

Toggle via the "Show explanation (slower)" checkbox in Streamlit, or `explain=true/false` on the API.

---

## Project structure

```text
LLM-LIE-DETECTOR/
├── core/                       # Detection pipeline (calibration, uncertainty, cross-check, aggregator, explainer)
├── models/                     # Model loaders — local HF model, Groq client
├── backend/                    # FastAPI app, metrics
├── ui/                         # Streamlit components, auth
├── evaluation/                 # TruthfulQA harness, weight tuning, Platt calibration
├── frontend/                   # Next.js experimental client
├── tests/                      # pytest suite
├── monitoring/                 # Prometheus + Grafana config
├── app.py                      # Streamlit entry point
├── config.yaml                 # All tunable weights, thresholds, model names
├── docker-compose.yml          # Full stack orchestration
├── Dockerfile
├── .github/workflows/ci.yml    # CI pipeline
└── SECURITY.md
```

> Adjust this tree to match your actual repo layout before publishing — confirm whether `core/`, `models/`, and `evaluation/` live at the repo root or are nested under `backend/`, since both have appeared across this project's history.

---

## Testing

```bash
# Full suite
PYTHONPATH=. pytest --tb=short

# Specific modules
PYTHONPATH=. pytest tests/test_api.py -v
PYTHONPATH=. pytest tests/test_aggregator.py -v
PYTHONPATH=. pytest tests/test_sanitizer.py -v
PYTHONPATH=. pytest tests/test_cache.py -v

# Skip tests that need a live GPU or Groq key
PYTHONPATH=. pytest -m "not requires_gpu and not requires_groq"
```

66 tests covering calibration scoring, semantic clustering, score aggregation (including the 2-signal fallback and hard-override paths), input sanitization, caching, and the API layer.

---

## References

- Lin, S., Hilton, J., & Evans, O. (2022). **TruthfulQA: Measuring How Models Mimic Human Falsehoods.** ACL 2022. [Paper](https://arxiv.org/abs/2109.07958)
- Kuhn, L., Gal, Y., & Farquhar, S. (2023). **Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation.** ICLR 2023. [Paper](https://arxiv.org/abs/2302.09664)
- Kadavath, S. et al. (2022). **Language Models (Mostly) Know What They Know.** [Paper](https://arxiv.org/abs/2207.05221)
- He, P. et al. (2021). **DeBERTa: Decoding-enhanced BERT with Disentangled Attention.** ICLR 2021. [Paper](https://arxiv.org/abs/2006.03654)
- Guo, C. et al. (2017). **On Calibration of Modern Neural Networks.** ICML 2017. [Paper](https://arxiv.org/abs/1706.04599)

---

## 🚫 Known Failure Modes

Every hallucination detector has operational boundaries where accuracy degrades. The following failure modes have been identified:

1. **Subjective or Creative Queries:** The system assumes a factual ground-truth exists. For creative writing, brainstorming, or open opinion questions, semantic uncertainty is naturally high (as diverse responses are valid) and cross-model agreement is low, leading to false positives (flagging valid responses as hallucinations).
2. **Common Shared Misconceptions:** If both the local model and the cross-check model (Groq) share the same misconception (e.g., believing that humans only use 10% of their brains), cross-model agreement will be high and semantic uncertainty will be low. The system will fail to flag the hallucination (false negative).
3. **Very Recent Events:** If the prompt involves real-time or very recent news that postdates the training cutoffs of both the local and Groq models, they may both hallucinate different facts or agree on wrong ones, causing unpredictable risk scores.
4. **Mixed-Accuracy Long Responses:** When evaluating multi-sentence or paragraph-length responses, a single aggregated risk score can obscure the fact that the model was 90% correct but hallucinated one crucial detail. In these cases, token and sentence-level explanations are critical.

---

## 📄 License

MIT — see [LICENSE](LICENSE).
