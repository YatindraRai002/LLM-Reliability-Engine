import time
import requests
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

# ── Authentication Gate ──────────────────────────────────────────────
if not check_auth():
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []
if "current_result" not in st.session_state:
    st.session_state.current_result = None

# Rate-limit: minimum seconds between analyse requests
RATE_LIMIT_SECONDS = 5

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
                with st.spinner("Analyzing via Backend API..."):
                    try:
                        # Make API request to the FastAPI backend
                        response = requests.post(
                            "http://localhost:8000/api/analyze",
                            json={"query": safe_query}
                        )
                        if response.status_code != 200:
                            st.error(f"Backend API Error: {response.text}")
                        else:
                            result_dict = response.json()
                            if "error" in result_dict:
                                st.error(result_dict["error"])
                            else:
                                from types import SimpleNamespace
                                if "result" in result_dict and isinstance(result_dict["result"], dict):
                                    result_dict["result"] = SimpleNamespace(**result_dict["result"])
                                st.session_state.current_result = result_dict
                                st.session_state.history.append(result_dict)
                                st.session_state["_last_run_ts"] = time.time()
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to the FastAPI backend. Is it running on port 8000?")
                    except Exception as e:
                        st.error(f"Frontend Execution Error: {e}")

    if st.session_state.current_result:
        render_results(st.session_state.current_result)
        
elif page == "Analytics":
    render_analytics_dashboard()
