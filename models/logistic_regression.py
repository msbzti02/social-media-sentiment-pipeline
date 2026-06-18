import os, sys, time, pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             cohen_kappa_score, matthews_corrcoef, log_loss)
from sklearn.preprocessing import label_binarize
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

if __name__ == "__main__":
    print("\n" + "="*60 + "\n  STAGE 8a: LOGISTIC REGRESSION\n  Mohammed Badawi - 2105656\n" + "="*60)

    X_train = sp.load_npz(os.path.join(PROCESSED_DATA_DIR, "X_train_tfidf.npz"))
    X_val = sp.load_npz(os.path.join(PROCESSED_DATA_DIR, "X_val_tfidf.npz"))
    X_test = sp.load_npz(os.path.join(PROCESSED_DATA_DIR, "X_test_tfidf.npz"))
    train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "test.csv"))
    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    print(f"\n  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    print("\n  Training Logistic Regression...")
    start = time.time()
    model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"  Training time: {train_time:.2f}s")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
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
    print(f"\n  Classification Report:")
    target_names = ["negative", "neutral", "positive"]
    print(classification_report(y_test, y_pred, target_names=target_names))
    print(f"  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    with open(os.path.join(MODELS_DIR, "logistic_regression.pkl"), "wb") as f:
        pickle.dump(model, f)
    np.savez(os.path.join(MODELS_DIR, "lr_predictions.npz"),
             y_test=y_test, y_pred=y_pred, y_proba=y_proba)
    print(f"\n  Model + predictions saved to {MODELS_DIR}")

    results = {"model":"LogisticRegression","accuracy":acc,"f1_macro":f1,
               "precision_macro":precision,"recall_macro":recall,
               "roc_auc_macro":roc_auc,"cohen_kappa":kappa,"mcc":mcc,
               "log_loss":logloss,"train_time_s":train_time}
    pd.DataFrame([results]).to_csv(os.path.join(OUTPUT_DIR,"lr_results.csv"), index=False)
    print("\n" + "="*60 + "\n  STAGE 8a COMPLETE\n" + "="*60 + "\n")
