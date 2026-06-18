import os, sys, time
import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             accuracy_score, precision_score, recall_score,
                             roc_auc_score, cohen_kappa_score, matthews_corrcoef, log_loss)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

def build_tokenizer(texts, max_words=20000):
    from tensorflow.keras.preprocessing.text import Tokenizer
    tok = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tok.fit_on_texts(texts)
    return tok

def texts_to_sequences(tokenizer, texts, maxlen=MAX_SEQUENCE_LENGTH):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=maxlen, padding="post", truncating="post")

def build_bilstm_model(vocab_size, embedding_dim=EMBEDDING_DIM, max_len=MAX_SEQUENCE_LENGTH, num_classes=3):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, SpatialDropout1D
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        SpatialDropout1D(0.3),
        Bidirectional(LSTM(LSTM_UNITS, return_sequences=True)),
        Bidirectional(LSTM(64)),
        Dropout(0.4),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    print("\n" + "="*60 + "\n  STAGE 8b: BIDIRECTIONAL LSTM\n  Mohammed Badawi - 2105656\n" + "="*60)

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf

    train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "test.csv"))

    max_samples = 100000
    if len(train_df) > max_samples:
        train_df = train_df.sample(max_samples, random_state=RANDOM_SEED)
        val_df = val_df.sample(min(len(val_df), max_samples//8), random_state=RANDOM_SEED)
        test_df = test_df.sample(min(len(test_df), max_samples//8), random_state=RANDOM_SEED)
        print(f"  Using subset: Train={len(train_df):,} Val={len(val_df):,} Test={len(test_df):,}")

    print("\n  Tokenizing texts...")
    tokenizer = build_tokenizer(train_df["clean_text"].astype(str))
    X_train = texts_to_sequences(tokenizer, train_df["clean_text"].astype(str))
    X_val = texts_to_sequences(tokenizer, val_df["clean_text"].astype(str))
    X_test = texts_to_sequences(tokenizer, test_df["clean_text"].astype(str))
    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    vocab_size = min(len(tokenizer.word_index)+1, 20001)
    print(f"  Vocab size: {vocab_size:,}")
    print(f"  Sequence length: {MAX_SEQUENCE_LENGTH}")

    model = build_bilstm_model(vocab_size)
    model.summary()

    print("\n  Training BiLSTM...")
    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS_LSTM, batch_size=BATCH_SIZE,
        verbose=1
    )
    train_time = time.time() - start
    print(f"\n  Training time: {train_time:.1f}s")

    y_proba = model.predict(X_test)
    y_pred = np.argmax(y_proba, axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    logloss = log_loss(y_test, y_proba)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    try:
        roc_auc = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
    except:
        roc_auc = 0.0

    print(f"\n  Test Accuracy:      {acc:.4f}")
    print(f"  Precision (macro):  {precision:.4f}")
    print(f"  Recall (macro):     {recall:.4f}")
    print(f"  F1-Score (macro):   {f1:.4f}")
    print(f"  ROC-AUC (macro):    {roc_auc:.4f}")
    print(f"  Cohen's Kappa:      {kappa:.4f}")
    print(f"  MCC:                {mcc:.4f}")
    print(f"  Log Loss:           {logloss:.4f}")
    print(classification_report(y_test, y_pred, target_names=["negative","neutral","positive"]))

    model.save(os.path.join(MODELS_DIR, "bilstm_model.keras"))
    np.savez(os.path.join(MODELS_DIR, "bilstm_predictions.npz"),
             y_test=y_test, y_pred=y_pred, y_proba=y_proba)
    print(f"  Model + predictions saved to {MODELS_DIR}")

    results = {"model":"BiLSTM","accuracy":acc,"f1_macro":f1,
               "precision_macro":precision,"recall_macro":recall,
               "roc_auc_macro":roc_auc,"cohen_kappa":kappa,"mcc":mcc,
               "log_loss":logloss,"train_time_s":train_time}
    pd.DataFrame([results]).to_csv(os.path.join(OUTPUT_DIR,"bilstm_results.csv"), index=False)
    print("\n" + "="*60 + "\n  STAGE 8b COMPLETE\n" + "="*60 + "\n")
