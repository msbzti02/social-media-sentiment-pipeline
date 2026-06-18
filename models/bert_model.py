import os, sys, time
import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             cohen_kappa_score, matthews_corrcoef, log_loss)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

def create_dataset(texts, labels, tokenizer, max_len=MAX_SEQUENCE_LENGTH):
    import torch
    from torch.utils.data import Dataset

    class SentimentDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, idx):
            encoding = self.tokenizer(
                str(self.texts[idx]), max_length=self.max_len,
                padding="max_length", truncation=True, return_tensors="pt"
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long)
            }
    return SentimentDataset(texts, labels, tokenizer, max_len)

if __name__ == "__main__":
    print("\n" + "="*60 + "\n  STAGE 8c: BERT FINE-TUNING\n  Mohammed Badawi - 2105656\n" + "="*60)

    import torch
    from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")

    train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "test.csv"))

    max_samples = 10000
    train_df = train_df.sample(min(len(train_df), max_samples), random_state=RANDOM_SEED)
    val_df = val_df.sample(min(len(val_df), max_samples//5), random_state=RANDOM_SEED)
    test_df = test_df.sample(min(len(test_df), max_samples//5), random_state=RANDOM_SEED)
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    print(f"\n  Loading {BERT_MODEL_NAME}...")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=3)

    train_dataset = create_dataset(train_df["clean_text"].tolist(), train_df["label"].tolist(), tokenizer)
    val_dataset = create_dataset(val_df["clean_text"].tolist(), val_df["label"].tolist(), tokenizer)
    test_dataset = create_dataset(test_df["clean_text"].tolist(), test_df["label"].tolist(), tokenizer)

    training_args = TrainingArguments(
        output_dir=os.path.join(MODELS_DIR, "bert_checkpoints"),
        num_train_epochs=EPOCHS_BERT,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none"
    )

    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=1)
        acc = accuracy_score(eval_pred.label_ids, preds)
        f1 = f1_score(eval_pred.label_ids, preds, average="macro")
        return {"accuracy": acc, "f1_macro": f1}

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    print("\n  Fine-tuning BERT...")
    start = time.time()
    trainer.train()
    train_time = time.time() - start
    print(f"\n  Training time: {train_time:.1f}s")

    preds_output = trainer.predict(test_dataset)
    y_proba_raw = preds_output.predictions
    y_proba = torch.nn.functional.softmax(torch.tensor(y_proba_raw), dim=1).numpy()
    y_pred = np.argmax(y_proba_raw, axis=1)
    y_test = test_df["label"].values
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

    model.save_pretrained(os.path.join(MODELS_DIR, "bert_finetuned"))
    tokenizer.save_pretrained(os.path.join(MODELS_DIR, "bert_finetuned"))
    np.savez(os.path.join(MODELS_DIR, "bert_predictions.npz"),
             y_test=y_test, y_pred=y_pred, y_proba=y_proba)
    print(f"  Model + predictions saved to {MODELS_DIR}")

    results = {"model":"BERT","accuracy":acc,"f1_macro":f1,
               "precision_macro":precision,"recall_macro":recall,
               "roc_auc_macro":roc_auc,"cohen_kappa":kappa,"mcc":mcc,
               "log_loss":logloss,"train_time_s":train_time}
    pd.DataFrame([results]).to_csv(os.path.join(OUTPUT_DIR,"bert_results.csv"), index=False)
    print("\n" + "="*60 + "\n  STAGE 8c COMPLETE\n" + "="*60 + "\n")
