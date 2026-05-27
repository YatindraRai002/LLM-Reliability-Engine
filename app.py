import time
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from ui.components import render_results
from ui.auth import check_auth
from core.aggregator import run_full_pipeline
from core.sanitizer import sanitize_prompt
from models.model_loader import get_open_model, get_embedding_model, get_nli_model
from ui.analytics import render_analytics_dashboard

st.set_page_config(
    page_title="LLM Lie Detector",
    page_icon="🔍",
    layout="wide"
)

# ── Authentication Gate ──────────────────────────────────────────────
if not check_auth():
    st.stop()

# ── Model Loading ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models... (first run only)")
def load_all_models():
    try:
        get_open_model()
        get_embedding_model()
        get_nli_model()
        return True
    except Exception as e:
        st.error(f"Resource Error: {e}")
        return False

if "history" not in st.session_state:
    st.session_state.history = []
if "current_result" not in st.session_state:
    st.session_state.current_result = None

# Rate-limit: minimum seconds between analyse requests
RATE_LIMIT_SECONDS = 5

models_ready = load_all_models()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Detector", "Analytics"])

if page == "Detector":
    st.title("LLM Lie Detector")
    st.caption("Detects hallucinations using calibration scoring, semantic uncertainty, and multi-model cross-checking.")

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query = st.text_input(
            "Enter a question for the LLM",
            placeholder="e.g. Who invented the telephone?",
            max_chars=10000
        )
    with col_btn:
        st.write("") 
        run_btn = st.button("Analyze", use_container_width=True)

    if run_btn and query:
        if not models_ready:
            st.error("Models are not loaded. Please check your system resources.")
        else:
            # ── Global IP Rate Limiting ──────────────────────────────────
            try:
                # Streamlit 1.38+ supports st.context.headers for reading IPs natively
                client_ip = st.context.headers.get("X-Forwarded-For", "unknown-session")
            except Exception:
                client_ip = "unknown-session"
                
            @st.cache_resource(show_spinner=False)
            def get_rate_limits():
                from collections import defaultdict
                return defaultdict(list)
                
            rate_limits = get_rate_limits()
            now = time.time()
            
            # Clean up timestamps older than 60 seconds
            rate_limits[client_ip] = [ts for ts in rate_limits[client_ip] if now - ts < 60]
            
            # Allowing maximum 5 requests per minute per user
            if len(rate_limits[client_ip]) >= 5:
                st.error("🚨 Global rate limit exceeded (Max 5 requests per minute). Please try again later.")
            else:
                rate_limits[client_ip].append(now)
                safe_query = sanitize_prompt(query)
                if not safe_query:
                    st.error("Invalid or empty query after sanitization.")
                else:
                    with st.spinner("Running production-grade detection pipeline..."):
                        try:
                            # Execution via the synchronous wrapper run_full_pipeline
                            # which handles the asyncio event loop internally.
                            result_dict = run_full_pipeline(safe_query)

                            if "error" in result_dict:
                                st.error(result_dict["error"])
                            else:
                                st.session_state.current_result = result_dict
                                st.session_state.history.append(result_dict)
                                st.session_state["_last_run_ts"] = time.time()
                        except Exception as e:
                            st.error(f"Pipeline Execution Error: {e}")

    if st.session_state.current_result:
        render_results(st.session_state.current_result)
        
elif page == "Analytics":
    render_analytics_dashboard()
