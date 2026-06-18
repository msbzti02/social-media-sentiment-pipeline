import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

def create_consumer(topics):
    from kafka import KafkaConsumer
    consumer = KafkaConsumer(
        *topics,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        auto_offset_reset="earliest",
        group_id="sentiment-consumer-group",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        consumer_timeout_ms=10000
    )
    return consumer

def consume_and_store(consumer, max_messages=5000):
    count = 0
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB]
        coll = db["kafka_ingested"]
        has_mongo = True
    except:
        has_mongo = False
        print("    [WARN] MongoDB unavailable, printing messages only.")

    for message in consumer:
        data = message.value
        data["kafka_topic"] = message.topic
        data["kafka_partition"] = message.partition
        data["kafka_offset"] = message.offset
        if has_mongo:
            coll.insert_one(data)
        count += 1
        if count % 500 == 0:
            print(f"    Consumed {count:,} messages from {message.topic}")
        if count >= max_messages:
            break
    print(f"\n    Total consumed: {count:,} messages")
    if has_mongo:
        print(f"    Stored in MongoDB collection: kafka_ingested")

if __name__ == "__main__":
    print("\n" + "="*60 + "\n  STAGE 6: KAFKA CONSUMER\n  Mohammed Badawi - 2105656\n" + "="*60)
    try:
        topics = [KAFKA_TOPIC_TWITTER, KAFKA_TOPIC_REDDIT]
        consumer = create_consumer(topics)
        print(f"  Listening on topics: {topics}")
        consume_and_store(consumer)
        consumer.close()
    except Exception as e:
        print(f"\n  [ERROR] Kafka consumer failed: {e}")
        print("  Make sure Kafka is running and producer has sent messages.")
    print("\n" + "="*60 + "\n  KAFKA CONSUMER COMPLETE\n" + "="*60 + "\n")
