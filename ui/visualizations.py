def uncertainty_scatter(embeddings_2d, cluster_labels, responses, uncertainty_score):
    import numpy as np
    import plotly.graph_objects as go
    
    if not embeddings_2d or not cluster_labels:
        fig = go.Figure()
        fig.add_annotation(text="No embedding data available", showarrow=False)
        return fig
    
    coords = np.array(embeddings_2d, dtype=np.float32)
    labels = np.array(cluster_labels)
    
    # Detect degenerate case: all points at same location (tiny variance)
    coord_range = coords.max() - coords.min() if len(coords) > 1 else 0
    if coord_range < 1e-10:
        # All responses are identical — add small jitter for visibility
        np.random.seed(42)
        coords = coords + np.random.normal(0, 0.01, coords.shape)
        jitter_note = " (responses nearly identical — jitter added for visibility)"
    else:
        jitter_note = ""
    
    fig = go.Figure()
    unique_labels = sorted(set(labels.tolist()))
    colors = ["#4C3DB5","#A03520","#0D6B50","#8A5C0A","#8B2020","#0E4A8A"]
    
    for i, lbl in enumerate(unique_labels):
        mask  = labels == lbl
        hover = [
            (r[:100]+"...") if len(r)>100 else r
            for r, m in zip(responses, mask) if m
        ]
        idxs = [j for j, m in enumerate(mask) if m]
        
        fig.add_trace(go.Scatter(
            x=coords[mask, 0], y=coords[mask, 1],
            mode="markers+text",
            name=f"Cluster {lbl} ({mask.sum()})",
            marker=dict(
                size=14,
                color=colors[i % len(colors)],
                line=dict(width=1, color="rgba(255,255,255,0.4)")
            ),
            text=[f"R{j}" for j in idxs],
            textposition="top center",
            textfont=dict(size=10),
            hovertext=hover,
            hovertemplate="<b>R%{text}</b><br>%{hovertext}<extra></extra>",
        ))
    
    n_clust = len(unique_labels)
    fig.update_layout(
        title=(
            f"Response clusters — {n_clust} semantic group{'s' if n_clust>1 else ''} "
            f"(uncertainty={uncertainty_score:.3f}){jitter_note}"
        ),
        xaxis_title="PCA dim 1",
        yaxis_title="PCA dim 2",
        height=370,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.18),
    )
    return fig

def nli_scores_chart(ab: dict, ba: dict):
    import plotly.graph_objects as go
    
    categories = ['Contradiction', 'Entailment', 'Neutral']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Local → Groq',
        x=categories,
        y=[ab.get('contradiction', 0), ab.get('entailment', 0), ab.get('neutral', 0)],
        marker_color='#4C3DB5'
    ))
    
    fig.add_trace(go.Bar(
        name='Groq → Local',
        x=categories,
        y=[ba.get('contradiction', 0), ba.get('entailment', 0), ba.get('neutral', 0)],
        marker_color='#0D6B50'
    ))
    
    fig.update_layout(
        barmode='group',
        title='Bidirectional NLI Probabilities',
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", y=-0.2)
    )
    return fig

