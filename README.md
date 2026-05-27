# 🔍 LLM Lie Detector

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39.0-red?logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-orange?logo=pytorch&logoColor=white)
![Security](https://img.shields.io/badge/Security-Hardened-success)

A professional-grade hallucination detection system that identifies when Large Language Models are "lying" by fusing three independent uncertainty signals.

> **Why this matters:** LLMs are prone to confidently stating false facts (hallucinating). Our system acts as an automated polygraph test for LLM outputs by measuring internal confidence, output determinism, and factual alignment across multiple evaluations.

---

## 🚀 Core Architecture
The system moves beyond simple text analysis by implementing a multi-signal fusion pipeline:

### 1. Calibration Scoring (Logit Analysis)
- **How it works**: Extracts the raw token-level probabilities (logits) from the local LLM during generation.
- **Signal**: If the model is "unsure" (low probability for the chosen token), the calibration score increases.
- **Implementation**: `core/calibration.py`

### 2. Semantic Uncertainty (Clustering)
- **How it works**: Generates $N$ diverse responses using high-temperature sampling. These are embedded into vector space and grouped using **Agglomerative Clustering**.
- **Signal**: If the model produces many different semantic meanings (high entropy), it is flagged as uncertain.
- **Implementation**: `core/semantic_uncertainty.py`

### 3. Cross-Check (Multi-Model Agreement)
- **How it works**: Compares the local model's response with a high-capability remote model (via **Groq API**) using a **DeBERTa-v3 NLI** cross-encoder.
- **Signal**: If the two models contradict each other, the cross-check uncertainty score increases.
- **Implementation**: `core/cross_check.py`

### 4. Weighted Aggregator
- **How it works**: Fuses the three signals using a weighted average:
  $$\text{Final Score} = w_1 \cdot \text{Calib} + w_2 \cdot \text{Uncertainty} + w_3 \cdot \text{CrossCheck}$$
- **Implementation**: `core/aggregator.py`

---

## 🛠️ Setup & Installation

### 1. Prerequisites
- Python 3.10+
- NVIDIA GPU (8GB+ VRAM recommended for 4-bit quantization)
- Groq API Key

### 2. Installation
```bash
git clone https://github.com/YatindraRai002/LLM-LIE-DETECTOR.git
cd LLM-LIE-DETECTOR
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```

Tuning hyperparameters (like weights and thresholds) and managing authentication parameters can be done in `config.yaml`.

---

## 💻 Usage

### Launch the Diagnostic Dashboard
The system includes a full, secure Streamlit UI for deep-dive analysis.
```bash
streamlit run app.py
```
> **Note**: The dashboard is secured. Default test credentials:
> **Username:** `admin` | **Password:** `admin_password`

### Evaluation & Optimization
- **Benchmark**: Run the TruthfulQA evaluation to test the system's accuracy.
  ```bash
  PYTHONPATH=. python evaluation/truthfulqa_eval.py
  ```
- **Profile**: Analyze latency bottlenecks.
  ```bash
  PYTHONPATH=. python evaluation/profiler.py
  ```
- **Optimize**: Tune weights based on eval results.
  ```bash
  PYTHONPATH=. python evaluation/tune_weights.py
  ```

---

## 🛡️ Security & Stability
- **Authentication**: Dashboard access is secured via `streamlit-authenticator` using robust bcrypt hashing and JWT session cookies.
- **Rate Limiting (Global)**: Enforces an IP-based global memory sliding window to prevent API spam and abuse (Max 5 requests/minute).
- **Rate Limiting (Upstream)**: Implements asynchronous exponential backoff for remote provider (Groq API) calls to prevent HTTP 429 crashes.
- **Input Sanitization**: Employs strict prompt truncation and algorithmic sanitization to prevent prompt-injection and tokenizer bombs.
- **Memory Efficiency**: Utilizes `@lru_cache` and deep 4-bit quantization to maximize VRAM utilization for local models.

---

## 📁 Project Structure
- `app.py`: Streamlit dashboard entry point.
- `core/`: The underlying detection logic (Calibration, Uncertainty, Cross-Check, Aggregator).
- `models/`: HuggingFace model loading and remote Groq API client integration.
- `evaluation/`: TruthfulQA benchmarking and pipeline profiling scripts.
- `ui/`: Dashboard modules, authentication, and visual PCA plots.
- `scripts/`: Dev tools like API smoke tests.
