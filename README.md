# 🔍 LLM Lie Detector

A professional-grade hallucination detection system that identifies when Large Language Models are "lying" by fusing three independent uncertainty signals.

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

## 🛠️ Setup & Installation

### 1. Prerequisites
- Python 3.10+
- NVIDIA GPU (8GB+ VRAM recommended for 4-bit quantization)
- Groq API Key

### 2. Installation
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 3. Configuration
Create a \`.env\` file in the root directory:
\`\`\`env
GROQ_API_KEY=your_groq_api_key_here
\`\`\`

Tuning hyperparameters (like weights and thresholds) can be done in \`config.yaml\`.

## 💻 Usage

### Launch the Diagnostic Dashboard
The system includes a full Streamlit UI for deep-dive analysis.
\`\`\`bash
streamlit run app.py
\`\`\`

### Evaluation & Optimization
- **Benchmark**: Run the TruthfulQA evaluation to test the system's accuracy.
  \`\`\`bash
  PYTHONPATH=. python evaluation/truthfulqa_eval.py
  \`\`\`
- **Profile**: Analyze latency bottlenecks.
  \`\`\`bash
  PYTHONPATH=. python evaluation/profiler.py
  \`\`\`
- **Optimize**: Tune weights based on eval results.
  \`\`\`bash
  PYTHONPATH=. python evaluation/tune_weights.py
  \`\`\`

## 🛡️ Security & Stability
- **Rate Limiting**: Exponential backoff for Groq API calls.
- **Memory Efficiency**: `@lru_cache` and 4-bit quantization to maximize VRAM utilization.
- **Input Safety**: Strict prompt truncation and payload sanitization to prevent API overflows.
- **Robustness**: Fallback defaults for missing configuration keys.

## 📁 Project Structure
- \`app.py\`: Streamlit entry point.
- \`core/\`: The detection logic (Calibration, Uncertainty, Cross-Check, Aggregator).
- \`models/\`: Model loading and Groq API client.
- \`evaluation/\`: Benchmark and profiling tools.
- \`ui/\`: Dashboard components and PCA visualizations.
