"""
kafka_simulation.py  —  Stage 6 (Simulated Kafka Stream)
Mohammed Badawi - 2105656

Demonstrates the Kafka producer→consumer pipeline using Python's built-in
queue module when a real Kafka broker is unavailable.  The output format,
throughput reporting, and MongoDB persistence are identical to the live version.
"""

import os, sys, json, time, queue, threading
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

# ── In-memory broker ──────────────────────────────────────────────────────────
_topics: dict[str, queue.Queue] = {}
_topic_lock = threading.Lock()

def _get_topic(name: str) -> queue.Queue:
    with _topic_lock:
        if name not in _topics:
            _topics[name] = queue.Queue()
        return _topics[name]

# ── Producer ──────────────────────────────────────────────────────────────────
def produce(df: pd.DataFrame, topic: str, rate_limit: int = 500, max_messages: int = 5000) -> tuple[int, float]:
    q = _get_topic(topic)
    count = 0
    start = time.time()
    for _, row in df.iterrows():
        if count >= max_messages:
            break
        msg = {
            "text":      str(row.get("text", "")),
            "sentiment": str(row.get("sentiment", "neutral")),
            "source":    str(row.get("source", "unknown")),
            "user":      str(row.get("user", "anonymous")),
            "timestamp": str(row.get("date", "")),
            "kafka_topic": topic,
        }
        q.put(json.dumps(msg))
        count += 1
        if count % rate_limit == 0:
            elapsed = time.time() - start
            print(f"    [PRODUCER][{topic}] Sent {count:,} msgs | {count/elapsed:,.0f} msg/s")
    elapsed = time.time() - start
    print(f"    [PRODUCER][{topic}] DONE — {count:,} msgs in {elapsed:.2f}s  ({count/elapsed:,.0f} msg/s)")
    return count, elapsed

# ── Consumer ──────────────────────────────────────────────────────────────────
def consume(topics: list[str], max_messages: int = 5000, timeout: float = 5.0) -> int:
    # Optional MongoDB persistence
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        client.server_info()
        coll = client[MONGO_DB]["kafka_ingested"]
        coll.drop()          # fresh run
        use_mongo = True
        print("    [CONSUMER] MongoDB connected — messages will be persisted.")
    except Exception:
        use_mongo = False
        print("    [CONSUMER] MongoDB unavailable — running in memory-only mode.")

    count = 0
    batch = []
    BATCH_SIZE = 200
    deadline = time.time() + timeout

    while count < max_messages and time.time() < deadline:
        for topic in topics:
            q = _get_topic(topic)
            try:
                raw = q.get(timeout=0.05)
                data = json.loads(raw)
                if use_mongo:
                    batch.append(data)
                    if len(batch) >= BATCH_SIZE:
                        coll.insert_many(batch)
                        batch.clear()
                count += 1
                if count % 500 == 0:
                    print(f"    [CONSUMER] Consumed {count:,} messages")
            except queue.Empty:
                pass

    if use_mongo and batch:
        coll.insert_many(batch)

    return count

# ── Runner ────────────────────────────────────────────────────────────────────
def run_simulation(combined_csv: str) -> None:
    df = pd.read_csv(combined_csv)
    twitter_df = df[df["source"] == "twitter"].head(5000)
    reddit_df  = df[df["source"] == "reddit"].head(5000)

    total_msgs = len(twitter_df) + len(reddit_df)
    print(f"\n  Simulating Kafka stream for {total_msgs:,} messages "
          f"({len(twitter_df):,} Twitter + {len(reddit_df):,} Reddit)...\n")

    # ── Producer thread (writes to in-memory queues) ──
    prod_stats = {}
    def producer_thread():
        print(f"  [PRODUCER] Streaming Twitter -> topic: {KAFKA_TOPIC_TWITTER}")
        n, t = produce(twitter_df, KAFKA_TOPIC_TWITTER)
        prod_stats[KAFKA_TOPIC_TWITTER] = (n, t)
        print(f"\n  [PRODUCER] Streaming Reddit  -> topic: {KAFKA_TOPIC_REDDIT}")
        n, t = produce(reddit_df,  KAFKA_TOPIC_REDDIT)
        prod_stats[KAFKA_TOPIC_REDDIT] = (n, t)

    # ── Consumer thread (reads from in-memory queues) ──
    cons_stats = {}
    def consumer_thread():
        time.sleep(0.2)   # slight delay so producer fills the queue first
        topics = [KAFKA_TOPIC_TWITTER, KAFKA_TOPIC_REDDIT]
        print(f"\n  [CONSUMER] Listening on: {topics}")
        n = consume(topics, max_messages=total_msgs, timeout=30.0)
        cons_stats["total"] = n

    t_prod = threading.Thread(target=producer_thread, daemon=True)
    t_cons = threading.Thread(target=consumer_thread, daemon=True)

    t_prod.start(); t_cons.start()
    t_prod.join();  t_cons.join()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n  ── Simulation Summary ──────────────────────────────────────")
    for topic, (n, t) in prod_stats.items():
        print(f"  Topic  : {topic}")
        print(f"    Produced : {n:,} messages in {t:.2f}s  ({n/t:,.0f} msg/s)")
    print(f"  Consumed : {cons_stats.get('total', 0):,} total messages")

    # Kafka offset simulation
    print("\n  ── Simulated Kafka Metadata ────────────────────────────────")
    offset = 0
    for topic, (n, _) in prod_stats.items():
        print(f"  Topic: {topic}  |  Partition: 0  |  Latest Offset: {offset + n - 1}")
        offset += n


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  STAGE 6: KAFKA STREAM SIMULATION")
    print("  Mohammed Badawi - 2105656")
    print("  (Simulated — no external broker required)")
    print("="*60)

    combined = os.path.join(RAW_DATA_DIR, "combined_dataset.csv")
    if not os.path.exists(combined):
        print(f"  [ERROR] Dataset not found: {combined}")
        print("  Run data/load_data.py first.")
        sys.exit(1)

    run_simulation(combined)

    print("\n" + "="*60)
    print("  STAGE 6 COMPLETE")
    print("="*60 + "\n")
