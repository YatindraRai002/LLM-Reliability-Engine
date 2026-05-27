"""
Calibration plots for the LLM Lie Detector.
Generates reliability diagrams comparing predicted probabilities
to empirical correctness frequencies.
"""
import plotly.graph_objects as go
import numpy as np
from sklearn.calibration import calibration_curve

def plot_calibration(
    y_true: list[int],
    y_prob: list[float],
    n_bins: int = 10,
    title: str = "Reliability Diagram",
) -> go.Figure:
    """
    Generates a Plotly calibration curve (reliability diagram).

    Args:
        y_true: Binary labels (1 = hallucination, 0 = correct)
        y_prob: Predicted probabilities
        n_bins: Number of bins for the curve
        title: Title of the plot
    """
    if not y_true or not y_prob:
        return go.Figure()

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy='uniform'
    )

    fig = go.Figure()

    # Perfectly calibrated reference line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Perfectly calibrated',
        line=dict(color='gray', dash='dash')
    ))

    # Empirical calibration curve
    fig.add_trace(go.Scatter(
        x=mean_predicted_value,
        y=fraction_of_positives,
        mode='lines+markers',
        name='Model calibration',
        marker=dict(size=8, color='#4C3DB5'),
        line=dict(color='#4C3DB5', width=2)
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Mean predicted probability",
        yaxis_title="Fraction of positives (empirical)",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1],
        width=500,
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(x=0.05, y=0.95)
    )

    return fig
