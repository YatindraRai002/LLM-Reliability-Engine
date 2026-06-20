import time
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from ui.components import render_results
from ui.auth import check_auth
from core.sanitizer import sanitize_prompt
from ui.analytics import render_analytics_dashboard

st.set_page_config(
    page_title="LLM Lie Detector",
    page_icon="🔍",
    layout="wide"
)

if not check_auth():
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []
if "current_result" not in st.session_state:
    st.session_state.current_result = None

RATE_LIMIT_SECONDS = 5

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Detector", "Analytics"])

if page == "Detector":
    st.title("LLM Lie Detector")
    st.caption(
        "Detects hallucinations using calibration scoring, "
        "semantic uncertainty, and multi-model cross-checking."
    )

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query = st.text_input(
            "Enter a question for the LLM",
            placeholder="e.g. Who invented the telephone?",
            max_chars=10_000,
        )
    with col_btn:
        st.write("")
        run_btn = st.button("Analyze", use_container_width=True)

    show_explanation = st.checkbox(
        "Show explanation (slower)",
        value=False,
        help=(
            "Runs Phase B explanation engine: token-level confidence highlighting, "
            "sentence-level contradiction detection (NLI), and signal SHAP analysis. "
            "Adds ~5–15 s depending on response length."
        ),
    )

    if run_btn and query:
        try:
            client_ip = st.context.headers.get("X-Forwarded-For", "unknown-session")
        except Exception:
            client_ip = "unknown-session"

        @st.cache_resource(show_spinner=False)
        def get_rate_limits():
            from collections import defaultdict
            return defaultdict(list)

        rate_limits = get_rate_limits()
        now = time.time()

        rate_limits[client_ip] = [
            ts for ts in rate_limits[client_ip] if now - ts < 60
        ]

        if len(rate_limits[client_ip]) >= 5:
            st.error(
                "🚨 Global rate limit exceeded (Max 5 requests per minute). "
                "Please try again later."
            )
        else:
            rate_limits[client_ip].append(now)
            safe_query = sanitize_prompt(query)
            if not safe_query:
                st.error("Invalid or empty query after sanitization.")
            else:
                spinner_msg = (
                    "Analyzing + building explanation…"
                    if show_explanation
                    else "Analyzing…"
                )
                with st.spinner(spinner_msg):
                    try:
                        from core.aggregator import run_full_pipeline

                        result_dict = run_full_pipeline(
                            safe_query,
                            explain=show_explanation,
                        )

                        from types import SimpleNamespace
                        if "result" in result_dict and isinstance(
                            result_dict["result"], object
                        ) and hasattr(result_dict["result"], "to_dict"):
                            pass

                        st.session_state.current_result = result_dict
                        st.session_state.history.append(result_dict)
                        st.session_state["_last_run_ts"] = time.time()
                    except Exception as e:
                        st.error(f"Pipeline Error: {e}")

    if st.session_state.current_result:
        render_results(st.session_state.current_result)

elif page == "Analytics":
    render_analytics_dashboard()
