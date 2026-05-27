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

    with col_a:
        st.markdown("**Local Model Response**")
        # Use highlighted HTML if explanation data is available
        explanation = result_dict.get("explanation_detail")
        if explanation and hasattr(explanation, "highlighted_html") and explanation.highlighted_html:
            st.markdown(explanation.highlighted_html, unsafe_allow_html=True)
        else:
            st.write(local_response)

    with col_b:
        st.markdown("**Groq Response**")
        st.write(cc.get("groq_response", "Groq response not available"))

    verdict_colors = {"agree": "green", "neutral": "orange", "contradict": "red", "collapse": "red"}
    v = cc.get("verdict", "neutral")
    st.markdown(f"**NLI verdict:** :{verdict_colors.get(v, 'gray')}[{v.upper()}]  |  Agreement score: `{cc.get('symmetric_agreement', 0.0):.3f}`")

def render_uncertainty_landscape(sem_detail: dict):
    """UMAP/PCA projection of N response embeddings."""
    embeddings = np.array(sem_detail["embeddings"])
    labels = sem_detail["cluster_labels"]
    responses = sem_detail["responses"]
    
    if len(embeddings) < 2:
        st.warning("Not enough samples to visualize landscape.")
        return

    # PCA to 2D
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    
    fig = go.Figure()
    unique_labels = list(set(labels))
    colors = ["#4C3DB5", "#A03520", "#0D6B50", "#8A5C0A", "#8B2020", "#0E4A8A"]
    
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        fig.add_trace(go.Scatter(
            x=coords[mask, 0],
            y=coords[mask, 1],
            mode="markers+text",
            name=f"Cluster {label}",
            marker=dict(size=12, color=colors[i % len(colors)]),
            text=[f"R{j}" for j, m in enumerate(mask) if m],
            hovertext=[responses[j][:100] + "..." for j, m in enumerate(mask) if m]
        ))
    
    fig.update_layout(
        title=f"Response clusters — {sem_detail['n_semantic_clusters']} distinct semantic groups",
        height=400,
        xaxis_title="PCA dim 1",
        yaxis_title="PCA dim 2"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Each point is one sampled response. Tight cluster = consistent model. Scattered = uncertain.")

def render_token_confidence(cal_detail: dict):
    """Bar chart of per-token confidence probabilities."""
    probs = cal_detail.get("token_probs", [])
    if not probs:
        st.warning("No token probabilities available.")
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

    if explanation is None:
        st.info("Explanation data is not available for this result.")
        return

    # ---- 1. Color-coded Response ----
    st.subheader("🔍 Token-Level Confidence")
    if hasattr(explanation, "highlighted_html") and explanation.highlighted_html:
        st.markdown(explanation.highlighted_html, unsafe_allow_html=True)
        st.caption("Each token is colored by the model's confidence. Red = uncertain, Green = confident.")
    else:
        st.info("Token-level highlighting is not available.")

    st.markdown("---")

    # ---- 2. Signal Contribution (SHAP-style) ----
    st.subheader("📊 Signal Contribution Analysis")
    if hasattr(explanation, "signal_pct") and explanation.signal_pct:
        signal = explanation.signal_pct

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
    if hasattr(explanation, "contradicting_sentences") and explanation.contradicting_sentences:
        for cs in explanation.contradicting_sentences:
            st.markdown(
                f'<div style="background-color:#FCEBEB; border-left:4px solid #A32D2D; '
                f'padding:10px; margin:5px 0; border-radius:4px;">'
                f'<strong style="color:#A32D2D;">Contradiction Score: {cs.contradiction_score:.3f}</strong><br>'
                f'<span style="color:#333;">{cs.sentence}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.success("No contradicting sentences detected.")

    st.markdown("---")

    # ---- 4. Recommendations ----
    st.subheader("💡 Recommendations")
    if hasattr(explanation, "recommendations") and explanation.recommendations:
        for rec in explanation.recommendations:
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
