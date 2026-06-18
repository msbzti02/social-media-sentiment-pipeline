import os, sys, random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

def generate_twitter_data(n=100000):
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    positive_tweets = [
        "Love this product! Best purchase ever {}",
        "Amazing experience with {} today, highly recommend!",
        "So happy with the new {} update, works perfectly!",
        "Great customer service from {} team, thank you!",
        "The {} conference was absolutely incredible! #inspired",
        "Just tried {} and I'm blown away! Game changer!",
        "Can't stop smiling after seeing {} results! #happy",
        "Wonderful news about {} innovation! Future is bright",
        "The new {} feature is exactly what we needed! #excited",
        "Thrilled to announce our partnership with {} #blessed",
        "{} just made my day! Such a great experience",
        "Finally! {} delivers on their promise. Well done!",
        "So grateful for {} community. You all are amazing!",
        "The {} launch event was spectacular! #innovation",
        "Best {} experience I've ever had. 10/10 would recommend",
    ]

    negative_tweets = [
        "Terrible experience with {} support, waited 3 hours",
        "The new {} update broke everything, so frustrated",
        "Worst {} product I've ever used, total waste of money",
        "Can't believe {} still hasn't fixed this bug #angry",
        "Disappointed with {} quality, used to be so much better",
        "{} customer service is absolutely terrible #fail",
        "The {} app keeps crashing, this is unacceptable",
        "Wasted my whole day dealing with {} issues #frustrated",
        "Avoid {} at all costs, they don't care about users",
        "The {} controversy shows they've lost their way",
        "So angry about {} pricing changes. Ridiculous!",
        "{} really let me down with this latest decision",
        "The decline of {} is sad to watch, they used to be great",
        "Horrible {} experience today, never coming back #done",
        "Fed up with {} constant problems. Switching to competitor",
    ]

    neutral_tweets = [
        "Just read about {} latest announcement. Thoughts?",
        "Interesting article about {} trends this year",
        "Anyone else following the {} situation? #news",
        "{} released new numbers today. Need to analyze",
        "Comparing {} options for my next project",
        "The {} report shows some changes worth noting",
        "Looking into {} alternatives. Anyone have suggestions?",
        "New {} study findings are out. Mixed results overall",
        "{} market showing interesting patterns this quarter",
        "Updated my {} settings today. Seems about the same",
    ]

    brands = ["Apple", "Google", "Tesla", "Amazon", "Microsoft", "Netflix",
              "Samsung", "Meta", "SpaceX", "OpenAI", "Twitter", "Spotify",
              "Uber", "Airbnb", "TikTok", "Discord", "Shopify", "Zoom"]

    records = []
    base_date = datetime(2024, 6, 1)
    n_pos, n_neg = int(n * 0.4), int(n * 0.35)
    n_neu = n - n_pos - n_neg

    for i in range(n):
        brand = random.choice(brands)
        if i < n_pos:
            text = random.choice(positive_tweets).format(brand)
            sentiment = "positive"
            target = 4
        elif i < n_pos + n_neg:
            text = random.choice(negative_tweets).format(brand)
            sentiment = "negative"
            target = 0
        else:
            text = random.choice(neutral_tweets).format(brand)
            sentiment = "neutral"
            target = 2

        ts = base_date + timedelta(days=random.randint(0,365), hours=random.randint(0,23), minutes=random.randint(0,59))
        records.append({
            "target": target,
            "id": random.randint(1000000, 9999999),
            "date": ts.strftime("%a %b %d %H:%M:%S PDT %Y"),
            "flag": "NO_QUERY",
            "user": f"user_{random.randint(1000,99999)}",
            "text": text,
            "sentiment": sentiment,
            "source": "twitter"
        })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return df

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  GENERATING SYNTHETIC TWITTER DATA")
    print("="*60)

    twitter_df = generate_twitter_data(100000)
    print(f"\n  Generated {len(twitter_df):,} tweets")
    for s in twitter_df["sentiment"].value_counts().items():
        print(f"    {s[0]:>10}: {s[1]:,}")

    twitter_df.to_csv(os.path.join(RAW_DATA_DIR, "twitter_loaded.csv"), index=False)
    print(f"\n  Saved to {RAW_DATA_DIR}/twitter_loaded.csv")

    reddit_path = os.path.join(RAW_DATA_DIR, "reddit_comments.csv")
    if os.path.exists(reddit_path):
        reddit_df = pd.read_csv(reddit_path)
        tw_sub = twitter_df[["text","sentiment","source","user","date"]].copy()
        rd_sub = reddit_df[["text","sentiment","source","user","date"]].copy()
        combined = pd.concat([tw_sub, rd_sub], ignore_index=True)
        combined.to_csv(os.path.join(RAW_DATA_DIR, "combined_dataset.csv"), index=False)
        print(f"  Combined dataset: {len(combined):,} records")

    print("\n  DONE\n")
