import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SQLITE_DB_PATH = os.path.join(CURRENT_DIR, "..", "results.db")

def load_analytics_data():
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        df = pd.read_sql_query("SELECT * FROM results", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Failed to load analytics data: {e}")
        return pd.DataFrame()

def render_analytics_dashboard():
    st.header("📊 Usage Analytics & Hallucination Trends")
    st.markdown("Analyze the history of scanned prompts, hallucination scores, and model performance over time.")
    
    df = load_analytics_data()
    
    if df.empty:
        st.info("No data available yet. Run some queries through the detector!")
        return

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 1. High-level metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Queries", len(df))
    
    high_risk_count = len(df[df['label'] == 'high'])
    c2.metric("High Risk Detected", high_risk_count)
    
    avg_score = df['score'].mean()
    c3.metric("Avg Hallucination Score", f"{avg_score:.2f}")
    
    c4.metric("Avg Token Confidence", f"{df['cal'].mean():.2f}")
    
    st.markdown("---")
    
    # 2. Score distribution over time
    st.subheader("Hallucination Scores Over Time")
    fig_time = px.scatter(
        df, x="timestamp", y="score", color="label",
        hover_data=["prompt"],
        color_discrete_map={"low": "green", "medium": "orange", "high": "red"}
    )
    st.plotly_chart(fig_time, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    # 3. Label Distribution
    with col1:
        st.subheader("Risk Distribution")
        fig_pie = px.pie(
            df, names="label", 
            color="label",
            color_discrete_map={"low": "green", "medium": "orange", "high": "red"}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    # 4. Signal Correlation
    with col2:
        st.subheader("Uncertainty vs Cross-Check")
        fig_scatter = px.scatter(
            df, x="unc", y="cc", color="label",
            labels={"unc": "Semantic Uncertainty", "cc": "Cross-Check Score"},
            color_discrete_map={"low": "green", "medium": "orange", "high": "red"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")
    
    # 5. Raw Data Table
    st.subheader("Recent Queries")
    st.dataframe(
        df.sort_values(by="timestamp", ascending=False).drop(columns=["id", "weights"]).head(50),
        use_container_width=True
    )
