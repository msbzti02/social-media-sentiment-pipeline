import os, sys, re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def tokenize_and_lemmatize(text):
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    try: stops = set(stopwords.words("english"))
    except: nltk.download("stopwords", quiet=True); stops = set(stopwords.words("english"))
    try: lemmatizer = WordNetLemmatizer()
    except: nltk.download("wordnet", quiet=True); lemmatizer = WordNetLemmatizer()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stops and len(t) > 2]
    return " ".join(tokens)

def add_metadata_features(df):
    df["word_count"] = df["clean_text"].apply(lambda x: len(str(x).split()))
    df["char_count"] = df["clean_text"].apply(lambda x: len(str(x)))
    df["has_exclamation"] = df["text"].astype(str).apply(lambda x: int("!" in x))
    df["hashtag_count"] = df["text"].astype(str).apply(lambda x: len(re.findall(r"#\w+", x)))
    return df

def add_vader_scores(df):
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        scores = df["text"].astype(str).apply(lambda x: analyzer.polarity_scores(x))
        df["vader_compound"] = scores.apply(lambda x: x["compound"])
        df["vader_pos"] = scores.apply(lambda x: x["pos"])
        df["vader_neg"] = scores.apply(lambda x: x["neg"])
        df["vader_neu"] = scores.apply(lambda x: x["neu"])
        print("    VADER scores added.")
    except ImportError:
        print("    [WARN] vaderSentiment not installed, skipping VADER features.")
        df["vader_compound"] = 0.0
    return df

def preprocess_pipeline(df):
    print("\n  [1/6] Cleaning text...")
    df["clean_text"] = df["text"].apply(clean_text)

    print("  [2/6] Tokenising & lemmatising...")
    import nltk
    for pkg in ["punkt", "stopwords", "wordnet", "punkt_tab"]:
        try: nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
        except LookupError: nltk.download(pkg, quiet=True)
    df["clean_text"] = df["clean_text"].apply(tokenize_and_lemmatize)

    print("  [3/6] Adding metadata features...")
    df = add_metadata_features(df)

    print("  [4/6] Adding VADER sentiment scores...")
    df = add_vader_scores(df)

    print("  [5/6] Removing empty / short posts (< 3 tokens)...")
    before = len(df)
    df = df[df["clean_text"].apply(lambda x: len(str(x).split()) >= 3)]
    df = df.dropna(subset=["clean_text", "sentiment"])
    print(f"    Removed {before - len(df):,} records ({(before-len(df))/before*100:.1f}%)")

    print("  [6/6] Mapping sentiment labels to integers...")
    df["label"] = df["sentiment"].map(LABEL_TO_INT)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    return df

def split_data(df):
    print(f"\n  Splitting data: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}...")

    train_df, temp_df = train_test_split(
        df, test_size=(1 - TRAIN_RATIO), random_state=RANDOM_SEED, stratify=df["label"])
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_df["label"])

    print(f"    Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    return train_df, val_df, test_df

def build_tfidf(train_df, val_df, test_df):
    print("\n  Building TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=MAX_FEATURES_TFIDF, ngram_range=(1,2))
    X_train = tfidf.fit_transform(train_df["clean_text"])
    X_val = tfidf.transform(val_df["clean_text"])
    X_test = tfidf.transform(test_df["clean_text"])
    print(f"    TF-IDF shape: {X_train.shape}")
    import pickle
    with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
    return X_train, X_val, X_test

if __name__ == "__main__":
    print("\n" + "="*60 + "\n  STAGE 4: DATA PREPROCESSING\n  Mohammed Badawi - 2105656\n" + "="*60)

    combined = os.path.join(RAW_DATA_DIR, "combined_dataset.csv")
    if not os.path.exists(combined):
        print(f"  [ERROR] Dataset not found at {combined}. Run data/load_data.py first.")
        sys.exit(1)

    df = pd.read_csv(combined)
    print(f"\n  Loaded {len(df):,} records")

    df = preprocess_pipeline(df)
    print(f"\n  After preprocessing: {len(df):,} records")

    train_df, val_df, test_df = split_data(df)
    X_train, X_val, X_test = build_tfidf(train_df, val_df, test_df)

    train_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "test.csv"), index=False)
    df.to_csv(os.path.join(PROCESSED_DATA_DIR, "full_processed.csv"), index=False)

    import scipy.sparse as sp
    sp.save_npz(os.path.join(PROCESSED_DATA_DIR, "X_train_tfidf.npz"), X_train)
    sp.save_npz(os.path.join(PROCESSED_DATA_DIR, "X_val_tfidf.npz"), X_val)
    sp.save_npz(os.path.join(PROCESSED_DATA_DIR, "X_test_tfidf.npz"), X_test)

    print(f"\n  All processed data saved to {PROCESSED_DATA_DIR}")
    print("="*60 + "\n  STAGE 4 COMPLETE\n" + "="*60 + "\n")
