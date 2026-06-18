import os, sys, json
import pandas as pd
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

def get_mongo_client():
    from pymongo import MongoClient
    client = MongoClient(MONGO_URI)
    return client

def setup_database():
    client = get_mongo_client()
    db = client[MONGO_DB]
    existing = db.list_collection_names()
    for coll_name in [MONGO_COLLECTION_RAW, MONGO_COLLECTION_PROCESSED, MONGO_COLLECTION_PREDICTIONS]:
        if coll_name not in existing:
            db.create_collection(coll_name)
            print(f"    Created collection: {coll_name}")
        else:
            print(f"    Collection exists: {coll_name}")
    db[MONGO_COLLECTION_PROCESSED].create_index("sentiment")
    db[MONGO_COLLECTION_PROCESSED].create_index("source")
    db[MONGO_COLLECTION_PROCESSED].create_index("timestamp")
    print("    Indexes created on: sentiment, source, timestamp")
    return db

def insert_raw_data(db, df, batch_size=5000):
    coll = db[MONGO_COLLECTION_RAW]
    coll.drop()
    records = df.to_dict("records")
    total = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        coll.insert_many(batch)
        total += len(batch)
    print(f"    Inserted {total:,} raw documents")

def insert_processed_data(db, df, batch_size=5000):
    coll = db[MONGO_COLLECTION_PROCESSED]
    coll.drop()
    documents = []
    for _, row in df.iterrows():
        doc = {
            "source": row.get("source", "unknown"),
            "text": str(row.get("text", "")),
            "clean_text": str(row.get("clean_text", "")),
            "timestamp": datetime.now().isoformat(),
            "user_id": str(row.get("user", "anonymous")),
            "sentiment": str(row.get("sentiment", "neutral")),
            "score": float(row.get("vader_compound", 0.0)) if "vader_compound" in row.index else 0.0,
            "metadata": {
                "hashtags": [],
                "word_count": int(row.get("word_count", 0)) if "word_count" in row.index else 0,
                "platform_score": 0
            }
        }
        documents.append(doc)
    total = 0
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        coll.insert_many(batch)
        total += len(batch)
    print(f"    Inserted {total:,} processed documents")

def run_aggregation_queries(db):
    coll = db[MONGO_COLLECTION_PROCESSED]
    print("\n  --- Aggregation Results ---")

    pipeline = [{"$group": {"_id": "$sentiment", "count": {"$sum": 1}}}, {"$sort": {"count": -1}}]
    print("\n  Sentiment Distribution:")
    for doc in coll.aggregate(pipeline):
        print(f"    {doc['_id']:>10}: {doc['count']:,}")

    pipeline = [{"$group": {"_id": {"source":"$source","sentiment":"$sentiment"}, "count":{"$sum":1}}}, {"$sort":{"_id.source":1}}]
    print("\n  By Source:")
    for doc in coll.aggregate(pipeline):
        print(f"    {doc['_id']['source']:>8} / {doc['_id']['sentiment']:>10}: {doc['count']:,}")

    pipeline = [{"$group": {"_id":"$sentiment", "avg_score":{"$avg":"$score"}}}, {"$sort":{"avg_score":-1}}]
    print("\n  Average VADER Score by Sentiment:")
    for doc in coll.aggregate(pipeline):
        print(f"    {doc['_id']:>10}: {doc['avg_score']:.4f}")

    total = coll.count_documents({})
    print(f"\n  Total documents in collection: {total:,}")

if __name__ == "__main__":
    print("\n" + "="*60 + "\n  STAGE 5: MONGODB INTEGRATION\n  Mohammed Badawi - 2105656\n" + "="*60)
    try:
        db = setup_database()
    except Exception as e:
        print(f"\n  [ERROR] Cannot connect to MongoDB: {e}")
        print("  Make sure MongoDB is running on localhost:27017")
        print("  You can start it with: mongod --dbpath /data/db")
        sys.exit(1)

    raw_path = os.path.join(RAW_DATA_DIR, "combined_dataset.csv")
    proc_path = os.path.join(PROCESSED_DATA_DIR, "full_processed.csv")

    if os.path.exists(raw_path):
        print("\n  Inserting raw data...")
        raw_df = pd.read_csv(raw_path)
        insert_raw_data(db, raw_df)

    if os.path.exists(proc_path):
        print("\n  Inserting processed data...")
        proc_df = pd.read_csv(proc_path)
        insert_processed_data(db, proc_df)
    else:
        print(f"  [WARN] Processed data not found. Run preprocessing/preprocess.py first.")

    run_aggregation_queries(db)
    print("\n" + "="*60 + "\n  STAGE 5 COMPLETE\n" + "="*60 + "\n")
