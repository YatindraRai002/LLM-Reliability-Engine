# Security & Stability Enhancements

## 1. API Resilience (Rate Limiting & Payload)
- **Exponential Backoff**: Implemented a `@retry_with_backoff` decorator in `models/groq_client.py`. This automatically handles HTTP 429 (Too Many Requests) by pausing and retrying with increasing delays.
- **Timeout Control**: Set a strict 30s timeout on Groq API requests to prevent the application from hanging indefinitely on a stalled request.
- **Payload Balancing**: Centralised prompt sanitisation (max 10k characters) in `core/sanitizer.py`, used by both `app.py` and `core/aggregator.py`.
- **UI Rate Limiting**: Added a 5-second cooldown between analyses in `app.py` to prevent button-spamming that could overload the local model or exhaust Groq API quota.

## 2. Input Sanitisation & Injection Defence
- **Centralised Sanitiser** (`core/sanitizer.py`):
  - Truncation to 10k characters.
  - Collapse of excessive repeated special characters (tokeniser abuse mitigation).
  - Detection and logging of known prompt-injection patterns (e.g. "ignore previous instructions", `<script>` tags).
- **XSS Prevention**: HTML/JavaScript fragments are flagged and logged.
- **Non-string coercion**: Non-string inputs are safely coerced to empty strings.

## 3. Authentication
- **Dashboard Auth** (`ui/auth.py`): Optional password gate controlled via `DASHBOARD_PASSWORD` in `.env`.
  - Uses `hmac.compare_digest()` to prevent timing-based side-channel attacks.
  - Disabled by default (no password = open access for local dev).

## 4. Data Integrity & Robustness
- **Sanitisation**: Added a `sanitize()` helper in `core/aggregator.py` that ensures all inputs to the final score are bounded between $[0, 1]$ and are valid floats. This prevents `NaN` or `Inf` results from crashing the UI.
- **Safe Config Loading**: Implemented a safe `load_config()` wrapper with fallback defaults, ensuring the app doesn't crash if a specific key is missing from `config.yaml`.

## 5. Infrastructure Security
- **Secret Management**: Strictly enforced the use of `.env` for API keys. Added `.env` to `.gitignore` to prevent accidental leaks to version control.
- **Resource Protection**: Used `@st.cache_resource` to prevent memory leaks and redundant VRAM allocation during model loading.
- **Dependency Pinning**: All dependencies in `requirements.txt` are pinned to exact versions (`==`) to prevent supply-chain attacks and ensure reproducible builds.
- **Expanded `.gitignore`**: Now covers `.venv/`, `.cache/`, IDE directories, and other transient artifacts.
