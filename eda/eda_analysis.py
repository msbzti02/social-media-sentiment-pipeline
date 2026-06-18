import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

sns.set_theme(style="darkgrid")
plt.rcParams.update({"figure.dpi": 150, "font.size": 12})
COLORS = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#3498db"}

def load_data():
    combined = os.path.join(RAW_DATA_DIR, "combined_dataset.csv")
    if os.path.exists(combined):
        return pd.read_csv(combined)
    dfs = []
    for f in ["twitter_loaded.csv", "reddit_comments.csv"]:
        p = os.path.join(RAW_DATA_DIR, f)
        if os.path.exists(p): dfs.append(pd.read_csv(p))
    if dfs: return pd.concat(dfs, ignore_index=True)
    raise FileNotFoundError("No data. Run data/load_data.py first.")

def plot_class_distribution(df):
    print("  [1/5] Class distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    counts = df["sentiment"].value_counts()
    colors = [COLORS.get(s, "#95a5a6") for s in counts.index]
    bars = axes[0].bar(counts.index, counts.values, color=colors, edgecolor="white")
    axes[0].set_title("Overall Sentiment Distribution", fontweight="bold")
    axes[0].set_ylabel("Number of Posts")
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+len(df)*0.005,
                     f"{val:,}", ha="center", fontweight="bold")
    sc = df.groupby(["source","sentiment"]).size().unstack(fill_value=0)
    sc.plot(kind="bar", ax=axes[1], color=[COLORS.get(c,"#95a5a6") for c in sc.columns], edgecolor="white")
    axes[1].set_title("By Source", fontweight="bold"); axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    plt.tight_layout(); plt.savefig(os.path.join(FIGURES_DIR,"eda_1_class_distribution.png"), bbox_inches="tight"); plt.close()

def plot_text_length(df):
    print("  [2/5] Text length distribution...")
    df["word_count"] = df["text"].astype(str).apply(lambda x: len(x.split()))
    df["char_count"] = df["text"].astype(str).apply(len)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for s in ["positive","negative","neutral"]:
        sub = df[df["sentiment"]==s]
        if len(sub)>0:
            axes[0].hist(sub["word_count"], bins=50, alpha=0.6, label=s, color=COLORS[s])
            axes[1].hist(sub["char_count"], bins=50, alpha=0.6, label=s, color=COLORS[s])
    axes[0].set_title("Word Count by Sentiment", fontweight="bold"); axes[0].legend()
    axes[1].set_title("Character Count by Sentiment", fontweight="bold"); axes[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIGURES_DIR,"eda_2_text_length.png"), bbox_inches="tight"); plt.close()

def plot_word_clouds(df):
    print("  [3/5] Word clouds...")
    sents = [s for s in ["positive","negative","neutral"] if s in df["sentiment"].values]
    fig, axes = plt.subplots(1, len(sents), figsize=(6*len(sents), 6))
    if len(sents)==1: axes=[axes]
    cmaps = {"positive":"Greens","negative":"Reds","neutral":"Blues"}
    for i,s in enumerate(sents):
        text = " ".join(df[df["sentiment"]==s]["text"].astype(str))
        wc = WordCloud(width=800,height=400,background_color="white",colormap=cmaps.get(s,"viridis"),max_words=100).generate(text)
        axes[i].imshow(wc, interpolation="bilinear"); axes[i].set_title(f"{s.capitalize()}", fontweight="bold"); axes[i].axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(FIGURES_DIR,"eda_3_word_clouds.png"), bbox_inches="tight"); plt.close()

def plot_top_keywords(df, top_n=20):
    print("  [4/5] Top keywords...")
    stops = {"the","a","an","is","it","to","in","for","of","and","i","this","that","you","my","on","at","with","has","have","was","are","be","but","not","so","we","do","no","just","get","im","its","rt","http","https","amp"}
    sents = [s for s in ["positive","negative","neutral"] if s in df["sentiment"].values]
    fig, axes = plt.subplots(1, len(sents), figsize=(6*len(sents), 8))
    if len(sents)==1: axes=[axes]
    for i,s in enumerate(sents):
        words = " ".join(df[df["sentiment"]==s]["text"].astype(str).str.lower()).split()
        words = [w for w in words if w not in stops and len(w)>2]
        common = Counter(words).most_common(top_n)
        if common:
            w,c = zip(*common)
            axes[i].barh(range(len(w)),c, color=COLORS.get(s,"#95a5a6"))
            axes[i].set_yticks(range(len(w))); axes[i].set_yticklabels(w); axes[i].invert_yaxis()
            axes[i].set_title(f"Top {top_n} — {s.capitalize()}", fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(FIGURES_DIR,"eda_4_top_keywords.png"), bbox_inches="tight"); plt.close()

def plot_temporal_trends(df):
    print("  [5/5] Temporal trends...")
    dft = df.copy(); dft["date"] = pd.to_datetime(dft["date"], errors="coerce", format="mixed")
    dft = dft.dropna(subset=["date"])
    if len(dft)==0: print("    [WARN] No valid dates, skipping."); return
    dft["month"] = dft["date"].dt.to_period("M").astype(str)
    trend = dft.groupby(["month","sentiment"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(14,6))
    for col in trend.columns:
        ax.plot(trend.index, trend[col], marker="o", linewidth=2, label=col.capitalize(), color=COLORS.get(col,"#95a5a6"))
    ax.set_title("Sentiment Trends Over Time", fontweight="bold"); ax.legend(title="Sentiment")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR,"eda_5_temporal_trends.png"), bbox_inches="tight"); plt.close()

if __name__ == "__main__":
    print("\n" + "="*60 + "\n  STAGE 3: EDA — Mohammed Badawi\n" + "="*60)
    df = load_data()
    print(f"\n  Total records: {len(df):,}")
    for s in df["sentiment"].unique():
        c = len(df[df["sentiment"]==s])
        print(f"    {s:>10}: {c:,} ({c/len(df)*100:.1f}%)")
    plot_class_distribution(df); plot_text_length(df); plot_word_clouds(df)
    plot_top_keywords(df); plot_temporal_trends(df)
    print(f"\n  STAGE 3 COMPLETE — Figures in {FIGURES_DIR}\n")
