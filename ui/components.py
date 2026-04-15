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
    tab1, tab2, tab3 = st.tabs(["Model responses", "Uncertainty landscape", "Token confidence"])
    
    with tab1:
        render_response_comparison(result_dict)
    
    with tab2:
        render_uncertainty_landscape(result_dict["uncertainty_detail"])
    
    with tab3:
        render_token_confidence(result_dict["calibration_detail"])

def render_response_comparison(result_dict: dict):
    cc = result_dict["cross_check_detail"]
    col_a, col_b = st.columns(2)

    # Use 'open_model_response' if available, otherwise fall back to the correction detail draft
    local_response = cc.get("open_model_response") or result_dict.get("correction_detail", {}).get("draft", "Response not available")

    with col_a:
        st.markdown("**Local Model Response**")
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
    probs = cal_detail["token_probs"]
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
    st.write(f"**Generated response:** {cal_detail['response']}")
