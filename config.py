import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

SENTIMENT140_FILE = os.path.join(RAW_DATA_DIR, "sentiment140.csv")
REDDIT_FILE = os.path.join(RAW_DATA_DIR, "reddit_comments.csv")
SENTIMENT140_COLUMNS = ["target", "id", "date", "flag", "user", "text"]

LABEL_MAP = {0: "negative", 2: "neutral", 4: "positive"}
LABEL_TO_INT = {"negative": 0, "neutral": 1, "positive": 2}

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
RANDOM_SEED = 42

MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB = "sentiment_db"
MONGO_COLLECTION_RAW = "raw_posts"
MONGO_COLLECTION_PROCESSED = "processed_posts"
MONGO_COLLECTION_PREDICTIONS = "predictions"

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC_TWITTER = "twitter-stream"
KAFKA_TOPIC_REDDIT = "reddit-stream"

SPARK_APP_NAME = "SentimentAnalysis"
SPARK_MASTER = "local[*]"

MAX_FEATURES_TFIDF = 50000
MAX_SEQUENCE_LENGTH = 128
EMBEDDING_DIM = 100
LSTM_UNITS = 128
BERT_MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 32
EPOCHS_LSTM = 3
EPOCHS_BERT = 3
LEARNING_RATE = 2e-5

NUM_REDDIT_COMMENTS = 50000
SUBREDDITS = ["technology", "news", "gadgets", "worldnews", "science",
              "politics", "gaming", "movies", "finance", "health"]
