import os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

import streamlit as st

st.set_page_config(page_title="Sentiment Analysis Dashboard", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); }
    .stMetric { background: rgba(255,255,255,0.05); border-radius: 12px; padding: 16px; backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); }
    h1 { background: linear-gradient(90deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Social Media Sentiment Analysis Dashboard")
st.markdown("**Mohammed Badawi – 2105656** | Big Data & AI Project")
st.divider()

@st.cache_data(ttl=30)
def load_from_mongodb():
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB]
        coll = db[MONGO_COLLECTION_PROCESSED]
        data = list(coll.find({}, {"_id": 0}).limit(50000))
        if data:
            return pd.DataFrame(data)
    except:
        pass
    proc = os.path.join(PROCESSED_DATA_DIR, "full_processed.csv")
    if os.path.exists(proc):
        return pd.read_csv(proc)
    return pd.DataFrame()

df = load_from_mongodb()

if df.empty:
    st.error("No data available. Run preprocessing first.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    source_filter = st.selectbox("📡 Data Source", ["All"] + sorted(df["source"].unique().tolist()) if "source" in df.columns else ["All"])
with col2:
    sentiment_filter = st.selectbox("💬 Sentiment", ["All"] + sorted(df["sentiment"].unique().tolist()) if "sentiment" in df.columns else ["All"])

filtered = df.copy()
if source_filter != "All" and "source" in df.columns:
    filtered = filtered[filtered["source"] == source_filter]
if sentiment_filter != "All" and "sentiment" in df.columns:
    filtered = filtered[filtered["sentiment"] == sentiment_filter]

st.divider()
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("📝 Total Posts", f"{len(filtered):,}")
with k2:
    pos = len(filtered[filtered["sentiment"]=="positive"]) if "sentiment" in filtered.columns else 0
    st.metric("😊 Positive", f"{pos:,}")
with k3:
    neg = len(filtered[filtered["sentiment"]=="negative"]) if "sentiment" in filtered.columns else 0
    st.metric("😞 Negative", f"{neg:,}")
with k4:
    neu = len(filtered[filtered["sentiment"]=="neutral"]) if "sentiment" in filtered.columns else 0
    st.metric("😐 Neutral", f"{neu:,}")

st.divider()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("📊 Sentiment Distribution")
    if "sentiment" in filtered.columns:
        counts = filtered["sentiment"].value_counts()
        colors_map = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#3498db"}
        colors = [colors_map.get(s, "#95a5a6") for s in counts.index]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(counts.values, labels=counts.index, colors=colors, autopct="%1.1f%%",
               startangle=90, textprops={"fontsize": 12})
        ax.set_title("Sentiment Breakdown", fontweight="bold")
        st.pyplot(fig)
        plt.close()

with col_b:
    st.subheader("📈 Posts by Source")
    if "source" in filtered.columns:
        src_counts = filtered["source"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(src_counts.index, src_counts.values, color=["#1DA1F2", "#FF4500"], edgecolor="white")
        ax.set_title("Posts per Source", fontweight="bold")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close()

st.divider()
st.subheader("📋 Recent Posts")
display_cols = [c for c in ["text", "sentiment", "source", "clean_text"] if c in filtered.columns]
if display_cols:
    st.dataframe(filtered[display_cols].head(20), use_container_width=True)

st.divider()
st.subheader("🤖 Model Performance Comparison")
model_comp = os.path.join(OUTPUT_DIR, "model_comparison.csv")
if os.path.exists(model_comp):
    model_df = pd.read_csv(model_comp)
    st.dataframe(model_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(model_df))
    ax.bar(x, model_df["f1"], color=["#3498db","#e74c3c","#2ecc71"][:len(model_df)], edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(model_df["model"])
    ax.set_title("F1-Score Comparison", fontweight="bold"); ax.set_ylim(0, 1)
    for i, v in enumerate(model_df["f1"]):
        ax.text(i, v+0.02, f"{v:.3f}", ha="center", fontweight="bold")
    st.pyplot(fig); plt.close()
else:
    st.info("Train models first to see comparison results.")

st.divider()
st.caption("Mohammed Badawi – 2105656 | Big Data & Artificial Intelligence | 2026")
