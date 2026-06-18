import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


def load_sentiment140():
    print("=" * 60)
    print("LOADING SENTIMENT140 DATASET")
    print("=" * 60)

    if not os.path.exists(SENTIMENT140_FILE):
        print(f"\n[ERROR] Sentiment140 file not found at:")
        print(f"  {SENTIMENT140_FILE}")
        print(f"\nPlease download it from:")
        print(f"  https://www.kaggle.com/datasets/kazanova/sentiment140")
        print(f"\nExtract 'training.1600000.processed.noemoticon.csv'")
        print(f"and rename it to 'sentiment140.csv' in data/raw/")
        return None

    df = pd.read_csv(
        SENTIMENT140_FILE,
        encoding="latin-1",
        header=None,
        names=SENTIMENT140_COLUMNS
    )

    df["sentiment"] = df["target"].map(LABEL_MAP)
    df["source"] = "twitter"

    print(f"\n  Total records: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\n  Sentiment Distribution:")
    for label, count in df["sentiment"].value_counts().items():
        print(f"    {label:>10}: {count:>10,} ({count/len(df)*100:.1f}%)")
    print(f"\n  Sample tweet: \"{df['text'].iloc[0][:80]}...\"")

    return df


def generate_reddit_comments(n=NUM_REDDIT_COMMENTS):
    print("\n" + "=" * 60)
    print("GENERATING SYNTHETIC REDDIT COMMENTS")
    print("=" * 60)

    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    positive_templates = [
        "This is absolutely amazing! Great work on {}.",
        "I love how {} has improved so much recently.",
        "Finally, some good news about {}! This is wonderful.",
        "The new {} update is fantastic, really impressed.",
        "Couldn't agree more, {} is the best thing this year.",
        "Brilliant analysis of {}. Thanks for sharing!",
        "This {} feature is exactly what we needed. Well done!",
        "So excited about the future of {}! Game changer.",
        "Best {} I've ever seen. Highly recommend to everyone.",
        "The quality of {} keeps getting better and better!",
        "Incredible progress in {}. The team deserves credit.",
        "Just tried {} and I'm blown away by the results.",
        "This is why I love {}. Innovation at its finest!",
        "Happy to see {} getting the recognition it deserves.",
        "Outstanding work on {}. This changes everything!",
    ]

    negative_templates = [
        "This is terrible. {} has really gone downhill.",
        "Disappointed with {}. Expected much better quality.",
        "The new {} is awful, worst update ever released.",
        "Can't believe how bad {} has become. Total disaster.",
        "Waste of time trying to use {}. Completely broken.",
        "Horrible experience with {}. Would not recommend.",
        "The {} situation is getting worse every day.",
        "Frustrated with {}. Nothing works as promised.",
        "Lost all trust in {} after this latest incident.",
        "The decline of {} is really sad to watch.",
        "Terrible decision by {} team. Users are furious.",
        "Avoid {} at all costs. Save yourself the headache.",
        "The {} controversy just shows how broken things are.",
        "Unacceptable quality from {}. We deserve better.",
        "Really angry about what {} has done. Not okay.",
    ]

    neutral_templates = [
        "Interesting perspective on {}. Need more data though.",
        "Has anyone else tried {}? Looking for feedback.",
        "The {} report was released today. Mixed results overall.",
        "Comparing {} vs alternatives. Each has pros and cons.",
        "Updated {} to the latest version. Seems about the same.",
        "Reading about {} in the news. Jury is still out.",
        "The {} debate continues. Both sides make valid points.",
        "Just learned about {}. Need to research more.",
        "Does anyone know if {} is worth looking into?",
        "The {} metrics show some changes but nothing dramatic.",
        "Wondering about the future of {}. Time will tell.",
        "New {} study published. Results are inconclusive.",
        "Thoughts on {}? I'm on the fence about it.",
        "The {} announcement was expected. No big surprises.",
        "Looking at {} from a different angle. Interesting topic.",
    ]

    topics = [
        "AI technology", "the new iPhone", "electric vehicles", "climate policy",
        "remote work", "cryptocurrency", "social media regulation",
        "space exploration", "healthcare reform", "online privacy",
        "machine learning", "renewable energy", "5G networks",
        "quantum computing", "autonomous vehicles", "cybersecurity",
        "virtual reality", "blockchain", "gene therapy", "smart cities",
        "cloud computing", "the gaming industry", "streaming services",
        "digital education", "telemedicine"
    ]

    records = []
    base_date = datetime(2025, 1, 1)

    sentiment_ratios = [0.4, 0.35, 0.25]
    n_pos = int(n * sentiment_ratios[0])
    n_neg = int(n * sentiment_ratios[1])
    n_neu = n - n_pos - n_neg

    for i in range(n):
        if i < n_pos:
            template = random.choice(positive_templates)
            sentiment = "positive"
            score = random.randint(10, 500)
        elif i < n_pos + n_neg:
            template = random.choice(negative_templates)
            sentiment = "negative"
            score = random.randint(-50, 50)
        else:
            template = random.choice(neutral_templates)
            sentiment = "neutral"
            score = random.randint(0, 100)

        topic = random.choice(topics)
        text = template.format(topic)
        timestamp = base_date + timedelta(
            days=random.randint(0, 365),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )

        records.append({
            "text": text,
            "sentiment": sentiment,
            "source": "reddit",
            "user": f"user_{random.randint(10000, 99999)}",
            "subreddit": random.choice(SUBREDDITS),
            "score": score,
            "timestamp": timestamp.isoformat(),
            "date": timestamp.strftime("%a %b %d %H:%M:%S PDT %Y")
        })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"\n  Total records: {len(df):,}")
    print(f"  Subreddits: {df['subreddit'].nunique()}")
    print(f"\n  Sentiment Distribution:")
    for label, count in df["sentiment"].value_counts().items():
        print(f"    {label:>10}: {count:>10,} ({count/len(df)*100:.1f}%)")
    print(f"\n  Sample comment: \"{df['text'].iloc[0][:80]}...\"")

    return df


def save_datasets(twitter_df, reddit_df):
    if twitter_df is not None:
        twitter_path = os.path.join(RAW_DATA_DIR, "twitter_loaded.csv")
        twitter_df.to_csv(twitter_path, index=False)
        print(f"\n  Twitter data saved: {twitter_path}")

    reddit_path = os.path.join(RAW_DATA_DIR, "reddit_comments.csv")
    reddit_df.to_csv(reddit_path, index=False)
    print(f"  Reddit data saved:  {reddit_path}")

    if twitter_df is not None:
        twitter_subset = twitter_df[["text", "sentiment", "source", "user", "date"]].copy()
        reddit_subset = reddit_df[["text", "sentiment", "source", "user", "date"]].copy()
        combined = pd.concat([twitter_subset, reddit_subset], ignore_index=True)
        combined_path = os.path.join(RAW_DATA_DIR, "combined_dataset.csv")
        combined.to_csv(combined_path, index=False)
        print(f"  Combined data saved: {combined_path}")
        print(f"  Combined total: {len(combined):,} records")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  STAGE 2: DATASET LOADING & PREPARATION")
    print("  Mohammed Badawi - 2105656")
    print("=" * 60)

    twitter_df = load_sentiment140()
    reddit_df = generate_reddit_comments()

    print("\n" + "=" * 60)
    print("SAVING DATASETS")
    print("=" * 60)
    save_datasets(twitter_df, reddit_df)

    print("\n" + "=" * 60)
    print("  STAGE 2 COMPLETE")
    print("=" * 60 + "\n")
