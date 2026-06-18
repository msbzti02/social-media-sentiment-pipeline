import os, sys, json, time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

def create_producer():
    from kafka import KafkaProducer
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all"
    )
    return producer

def stream_data(producer, df, topic, rate_limit=100, max_messages=5000):
    count = 0
    start = time.time()
    for _, row in df.iterrows():
        if count >= max_messages:
            break
        message = {
            "text": str(row.get("text", "")),
            "sentiment": str(row.get("sentiment", "neutral")),
            "source": str(row.get("source", "unknown")),
            "user": str(row.get("user", "anonymous")),
            "timestamp": str(row.get("date", "")),
        }
        producer.send(topic, value=message)
        count += 1
        if count % rate_limit == 0:
            producer.flush()
            elapsed = time.time() - start
            print(f"    [{topic}] Sent {count:,} messages | {count/elapsed:.0f} msg/sec")
    producer.flush()
    elapsed = time.time() - start
    print(f"    [{topic}] DONE — {count:,} messages in {elapsed:.1f}s ({count/elapsed:.0f} msg/sec)")
    return count, elapsed

if __name__ == "__main__":
    print("\n" + "="*60 + "\n  STAGE 6: KAFKA PRODUCER\n  Mohammed Badawi - 2105656\n" + "="*60)
    try:
        producer = create_producer()
        print("  Connected to Kafka broker.")
    except Exception as e:
        print(f"\n  [ERROR] Cannot connect to Kafka: {e}")
        print("  Make sure Kafka is running on localhost:9092")
        sys.exit(1)

    combined = os.path.join(RAW_DATA_DIR, "combined_dataset.csv")
    if not os.path.exists(combined):
        print(f"  [ERROR] Data not found. Run data/load_data.py first."); sys.exit(1)
    df = pd.read_csv(combined)

    twitter_df = df[df["source"] == "twitter"].head(5000)
    reddit_df = df[df["source"] == "reddit"].head(5000)

    print(f"\n  Streaming {len(twitter_df):,} Twitter messages...")
    stream_data(producer, twitter_df, KAFKA_TOPIC_TWITTER)

    print(f"\n  Streaming {len(reddit_df):,} Reddit messages...")
    stream_data(producer, reddit_df, KAFKA_TOPIC_REDDIT)

    producer.close()
    print("\n" + "="*60 + "\n  KAFKA PRODUCER COMPLETE\n" + "="*60 + "\n")
