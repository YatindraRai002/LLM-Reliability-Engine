import streamlit as st
import plotly.graph_objects as go
import numpy as np
from sklearn.decomposition import PCA

def render_results(result_dict: dict):
    """
    Main rendering function for the detection results.
    """
    result = result_dict["result"]
    
    # ---- Risk Score Banner ----
    color_map = {"low": "🟢", "medium": "🟡", "high": "🔴"}
    st.markdown(f"### {color_map[result.label]} Hallucination Risk: `{result.score:.3f}` — {result.label.upper()}")
    st.info(result.explanation)
    
    # ---- Three metric columns ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final score", f"{result.score:.3f}")
    c2.metric("Calibration", f"{result.calibration_score:.3f}")
    c3.metric("Uncertainty", f"{result.uncertainty_score:.3f}")
    c4.metric("Cross-check", f"{result.cross_check_score:.3f}")
    
    # ---- Score breakdown bar chart ----
    fig_scores = go.Figure(go.Bar(
        x=["Calibration", "Semantic uncertainty", "Cross-check"],
        y=[result.calibration_score, result.uncertainty_score, result.cross_check_score],
        marker_color=["#4C3DB5", "#A03520", "#0D6B50"],
        text=[f"{s:.3f}" for s in [result.calibration_score, result.uncertainty_score, result.cross_check_score]],
        textposition="outside"
    ))
    fig_scores.update_layout(
        title="Signal breakdown",
        yaxis_range=[0, 1],
        height=300,
        margin=dict(t=40, b=20)
    )
    st.plotly_chart(fig_scores, use_container_width=True)
    
    # ---- Tabs for detailed views ----
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Model responses", "Uncertainty landscape", "Token confidence", "Explanations", "System Calibration"
    ])
    
    with tab1:
        render_response_comparison(result_dict)
    
    with tab2:
        render_uncertainty_landscape(result_dict["uncertainty_detail"])
    
    with tab3:
        render_token_confidence(result_dict["calibration_detail"])

    with tab4:
        render_explanations(result_dict)

    with tab5:
        render_system_calibration()

def render_response_comparison(result_dict: dict):
    cc = result_dict["cross_check_detail"]
    col_a, col_b = st.columns(2)

    # Use 'open_model_response' if available, otherwise fall back to the correction detail draft
    local_response = cc.get("open_model_response") or result_dict.get("correction_detail", {}).get("draft", "Response not available")

    _tab_responses(result_dict["cross_check_detail"])

def _tab_responses(cc: dict):
    st.subheader("Response comparison")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Local model** (TinyLlama)")
        response = cc.get("local_response", "—")
        st.info(response if response else "No response generated")
    
    with c2:
        st.markdown(f"**{cc.get('groq_model', 'Groq')}**")
        
        if not cc.get("groq_available", True):
            # Show helpful error, not just "not available"
            error = cc.get("error", "Unknown error")
            error_type = cc.get("error_type", "unknown")
            
            if error_type == "invalid_key" or "401" in str(error) or "Invalid API Key" in str(error):
                st.error(
                    "🔑 **Invalid Groq API Key**\n\n"
                    "1. Go to https://console.groq.com/keys\n"
                    "2. Create a new API key\n"
                    "3. Update `GROQ_API_KEY` in your `.env` file\n"
                    "4. Restart Streamlit (`Ctrl+C` then `streamlit run app.py`)"
                )
            else:
                st.warning(f"⚠️ Groq cross-check unavailable")
                with st.expander("Error details"):
                    st.code(f"Error: {error}")
            
            st.caption("Running in 2-signal mode (calibration + uncertainty only)")
            return
        
        groq_resp = cc.get("groq_response", "—")
        st.info(groq_resp)
    
    # NLI verdict — only show when Groq was available
    if cc.get("groq_available", False):
        verdict = cc.get("verdict", "unknown")
        icon    = {"agree": "🟢", "neutral": "🟡", "contradict": "🔴"}.get(verdict, "⚪")
        agr     = cc.get("symmetric_agreement", 0.0)
        st.markdown(
            f"**NLI verdict:** {icon} `{verdict.upper()}`  |  "
            f"Symmetric agreement: `{agr:.3f}`"
        )
        
        ab = cc.get("ab_detail", {})
        ba = cc.get("ba_detail", {})
        if ab and ba:
            from ui.visualizations import nli_scores_chart
            st.plotly_chart(nli_scores_chart(ab, ba), use_container_width=True)

def render_uncertainty_landscape(sem_detail: dict):
    """Render the PCA scatter plot of response embeddings."""
    import streamlit as st
    from ui.visualizations import uncertainty_scatter

    st.subheader("Semantic uncertainty landscape")

    # ── Metrics row ──────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Uncertainty score",
        f"{sem_detail.get('uncertainty_score', 0):.3f}"
    )
    c2.metric(
        "Semantic clusters",
        sem_detail.get("n_semantic_clusters", 1),
        help="More clusters = more inconsistent answers"
    )
    c3.metric(
        "Mean similarity",
        f"{sem_detail.get('mean_pairwise_similarity', 1.0):.3f}",
        help="Higher = responses are more consistent"
    )

    # ── Scatter plot ──────────────────────────────────────────────────────
    # CORRECT key: embeddings_2d (NOT "embeddings")
    embeddings_2d = sem_detail.get("embeddings_2d", [])   # ← .get() with default
    cluster_labels = sem_detail.get("cluster_labels", [])
    responses = sem_detail.get("responses", [])

    if not embeddings_2d or not cluster_labels or not responses:
        st.info(
            "Uncertainty landscape not available. "
            "This happens when fewer than 2 samples were generated."
        )
    else:
        st.plotly_chart(
            uncertainty_scatter(
                embeddings_2d,
                cluster_labels,
                responses,
                sem_detail.get("uncertainty_score", 0.5),
            ),
            use_container_width=True,
        )
        st.caption(
            "Each point = one sampled response. "
            "Tight cluster = consistent model. "
            "Scattered = uncertain."
        )

    # ── Sampled responses expander ────────────────────────────────────────
    if responses:
        with st.expander(
            f"All {len(responses)} sampled responses", expanded=False
        ):
            for i, (resp, lbl) in enumerate(
                zip(responses, cluster_labels or [0] * len(responses))
            ):
                st.markdown(f"**Sample {i+1}** (cluster {lbl}):")
                st.write(resp)
                st.divider()

def render_token_confidence(cal_detail: dict):
    """Bar chart of per-token confidence probabilities."""
    probs = cal_detail.get("token_probs", None)
    
    if not probs:
        st.info("Token probability data not available for this response.")
        st.markdown("**Generated response:**")
        st.write(cal_detail.get("response", "No response recorded."))
        return
        
    x_labels = [f"t{i}" for i in range(len(probs))]
    
    fig = go.Figure(go.Bar(
        x=x_labels,
        y=probs,
        marker_color=["#0D6B50" if p > 0.7 else "#8A5C0A" if p > 0.4 else "#8B2020" for p in probs]
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="50% threshold")
    fig.update_layout(
        title="Token-level confidence (green=confident, red=uncertain)",
        yaxis_range=[0, 1],
        height=300,
        xaxis_title="Token position",
        yaxis_title="P(token)"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"**Generated response:** {cal_detail.get('response', 'N/A')}")


# ── Explanations Tab ─────────────────────────────────────────────────

def render_explanations(result_dict: dict):
    """Render the Explanations tab with token highlighting, contradictions, and signal SHAP."""
    explanation = result_dict.get("explanation_detail")

    # explain might be None, empty dict, or missing keys
    if not explanation or not isinstance(explanation, dict):
        st.info("Explanation data is not available for this result.")
        st.caption("This happens when the explainer module didn't run.")
        return
    
    # Check if explainer ran but produced empty output
    if not any([
        explanation.get("flagged_spans"),
        explanation.get("signal_pct"),
        explanation.get("recommendations"),
        explanation.get("highlighted_html")
    ]):
        st.info("No significant explanation signals found for this response.")
        return

    # ---- 1. Color-coded Response ----
    st.subheader("🔍 Token-Level Confidence")
    if explanation.get("highlighted_html"):
        st.markdown(explanation["highlighted_html"], unsafe_allow_html=True)
        st.caption("Each token is colored by the model's confidence. Red = uncertain, Green = confident.")
    else:
        st.info("Token-level highlighting is not available.")

    st.markdown("---")

    # ---- 2. Signal Contribution (SHAP-style) ----
    st.subheader("📊 Signal Contribution Analysis")
    if explanation.get("signal_pct"):
        signal = explanation["signal_pct"]

        fig = go.Figure(go.Bar(
            x=[signal.get("calibration", 0), signal.get("semantic_uncertainty", 0), signal.get("cross_check", 0)],
            y=["Calibration", "Semantic Uncertainty", "Cross-Check"],
            orientation="h",
            marker_color=["#4C3DB5", "#A03520", "#0D6B50"],
            text=[f"{signal.get('calibration', 0):.1f}%", f"{signal.get('semantic_uncertainty', 0):.1f}%", f"{signal.get('cross_check', 0):.1f}%"],
            textposition="outside",
        ))
        fig.update_layout(
            title="Which signal drove the hallucination score?",
            xaxis_title="Contribution (%)",
            xaxis_range=[0, 100],
            height=250,
            margin=dict(l=10, r=10, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Signal contribution data is not available.")

    st.markdown("---")

    # ---- 3. Contradicting Sentences ----
    st.subheader("🔴 Contradicting Sentences")
    if explanation.get("contradicting_sentences"):
        for cs in explanation["contradicting_sentences"]:
            st.markdown(
                f'<div style="background-color:#FCEBEB; border-left:4px solid #A32D2D; '
                f'padding:10px; margin:5px 0; border-radius:4px;">'
                f'<strong style="color:#A32D2D;">Contradiction Score: {cs.get("contradiction_score", 0):.3f}</strong><br>'
                f'<span style="color:#333;">{cs.get("sentence", "")}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.success("No contradicting sentences detected.")

    st.markdown("---")

    # ---- 4. Recommendations ----
    st.subheader("💡 Recommendations")
    if explanation.get("recommendations"):
        for rec in explanation["recommendations"]:
            st.markdown(f"- {rec}")
    else:
        st.success("✅ No actionable recommendations.")


def render_system_calibration():
    """Render the System Calibration tab with a reliability diagram."""
    import os
    import json
    from evaluation.calibration_plots import plot_calibration
    
    st.subheader("📈 System Calibration (Reliability Diagram)")
    st.markdown("This shows how well the model's confidence scores align with actual empirical correctness.")
    
    eval_path = os.path.join(os.path.dirname(__file__), "..", "eval_results.json")
    if not os.path.exists(eval_path):
        st.info("No evaluation data found. Run the evaluation harness to generate a reliability diagram.")
        st.code("PYTHONPATH=. python evaluation/truthfulqa_eval.py --n 20")
        return
        
    try:
        with open(eval_path, "r") as f:
            results = json.load(f)
            
        labeled = [r for r in results if r.get("correctness") is not None]
        if not labeled:
            st.info("No labeled evaluation data found.")
            return
            
        raw_scores = [r["hallucination_score"] for r in labeled]
        labels = [0 if r["correctness"] else 1 for r in labeled]
        
        # We can also plot the calibrated scores if available, but for simplicity we plot raw vs empirical
        fig = plot_calibration(labels, raw_scores, title="Reliability Diagram (Raw Scores vs Emp. Correctness)")
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error rendering calibration plot: {e}")
