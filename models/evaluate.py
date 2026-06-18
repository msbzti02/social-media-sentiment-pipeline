import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_curve, auc, accuracy_score,
                             f1_score, precision_score, recall_score,
                             roc_auc_score, cohen_kappa_score, matthews_corrcoef,
                             log_loss, classification_report)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

TARGET_NAMES = ["Negative", "Neutral", "Positive"]
MODEL_COLORS = {"LogisticRegression": "#3498db", "BiLSTM": "#e74c3c", "BERT": "#2ecc71"}

# =====================================================================
#  1. LOAD PREDICTIONS
# =====================================================================
def load_predictions():
    """Load saved prediction arrays from all models."""
    results = []
    model_files = {
        "LogisticRegression": "lr_predictions.npz",
        "BiLSTM": "bilstm_predictions.npz",
        "BERT": "bert_predictions.npz"
    }
    for model_name, fname in model_files.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            data = np.load(path)
            results.append({
                "model": model_name,
                "y_test": data["y_test"],
                "y_pred": data["y_pred"],
                "y_proba": data["y_proba"]
            })
            print(f"  Loaded: {fname} ({len(data['y_test']):,} samples)")
        else:
            print(f"  [SKIP] {fname} not found")
    return results

# =====================================================================
#  2. COMPUTE ALL METRICS
# =====================================================================
def compute_all_metrics(res):
    """Compute comprehensive metrics for a single model."""
    y_test, y_pred, y_proba = res["y_test"], res["y_pred"], res["y_proba"]

    metrics = {
        "model": res["model"],
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro"),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted"),
        "recall_macro": recall_score(y_test, y_pred, average="macro"),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted"),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "cohen_kappa": cohen_kappa_score(y_test, y_pred),
        "mcc": matthews_corrcoef(y_test, y_pred),
    }

    # ROC-AUC (One-vs-Rest)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    try:
        metrics["roc_auc_macro"] = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
        metrics["roc_auc_weighted"] = roc_auc_score(y_test_bin, y_proba, average="weighted", multi_class="ovr")
    except:
        metrics["roc_auc_macro"] = 0.0
        metrics["roc_auc_weighted"] = 0.0

    # Log Loss
    try:
        metrics["log_loss"] = log_loss(y_test, y_proba)
    except:
        metrics["log_loss"] = 0.0

    # Per-class metrics
    report = classification_report(y_test, y_pred, target_names=TARGET_NAMES, output_dict=True)
    for cls in TARGET_NAMES:
        metrics[f"precision_{cls}"] = report[cls]["precision"]
        metrics[f"recall_{cls}"] = report[cls]["recall"]
        metrics[f"f1_{cls}"] = report[cls]["f1-score"]
        metrics[f"support_{cls}"] = report[cls]["support"]

    # Load training time from CSV
    csv_map = {"LogisticRegression": "lr_results.csv", "BiLSTM": "bilstm_results.csv", "BERT": "bert_results.csv"}
    csv_path = os.path.join(OUTPUT_DIR, csv_map.get(res["model"], ""))
    if os.path.exists(csv_path):
        r = pd.read_csv(csv_path).iloc[0]
        metrics["train_time_s"] = r.get("train_time_s", 0)
    else:
        metrics["train_time_s"] = 0

    return metrics

# =====================================================================
#  3. CONFUSION MATRICES
# =====================================================================
def plot_confusion_matrices(predictions):
    """Plot confusion matrix heatmaps for all models side by side."""
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    if n == 1: axes = [axes]
    for i, res in enumerate(predictions):
        cm = confusion_matrix(res["y_test"], res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                    xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES)
        acc = accuracy_score(res["y_test"], res["y_pred"])
        f1 = f1_score(res["y_test"], res["y_pred"], average="macro")
        axes[i].set_title(f"{res['model']}\nAcc={acc:.3f} | F1={f1:.3f}", fontweight="bold")
        axes[i].set_xlabel("Predicted"); axes[i].set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "eval_confusion_matrices.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print("  Saved: eval_confusion_matrices.png")

# =====================================================================
#  4. ROC CURVES (One-vs-Rest)
# =====================================================================
def plot_roc_curves(predictions):
    """Plot ROC curves per class for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for cls_idx, cls_name in enumerate(TARGET_NAMES):
        ax = axes[cls_idx]
        for res in predictions:
            y_test_bin = label_binarize(res["y_test"], classes=[0, 1, 2])
            fpr, tpr, _ = roc_curve(y_test_bin[:, cls_idx], res["y_proba"][:, cls_idx])
            roc_auc_val = auc(fpr, tpr)
            color = MODEL_COLORS.get(res["model"], "#95a5a6")
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{res['model']} (AUC={roc_auc_val:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
        ax.set_title(f"ROC Curve — {cls_name}", fontweight="bold")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "eval_roc_curves.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print("  Saved: eval_roc_curves.png")

# =====================================================================
#  5. PER-CLASS METRICS
# =====================================================================
def plot_per_class_metrics(all_metrics):
    """Plot per-class precision, recall, F1 grouped by model."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metric_types = ["precision", "recall", "f1"]
    metric_labels = ["Precision", "Recall", "F1-Score"]
    x = np.arange(len(TARGET_NAMES))
    width = 0.25

    for ax_idx, (mt, ml) in enumerate(zip(metric_types, metric_labels)):
        ax = axes[ax_idx]
        for i, m in enumerate(all_metrics):
            values = [m.get(f"{mt}_{cls}", 0) for cls in TARGET_NAMES]
            color = MODEL_COLORS.get(m["model"], "#95a5a6")
            bars = ax.bar(x + i * width, values, width, label=m["model"],
                          color=color, edgecolor="white")
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", fontsize=8, fontweight="bold")
        ax.set_title(f"Per-Class {ml}", fontweight="bold")
        ax.set_xticks(x + width); ax.set_xticklabels(TARGET_NAMES)
        ax.set_ylim(0, 1.15); ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "eval_per_class_metrics.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print("  Saved: eval_per_class_metrics.png")

# =====================================================================
#  6. COMPREHENSIVE MODEL COMPARISON (6-panel)
# =====================================================================
def plot_model_comparison(all_metrics):
    """Plot 6-panel model comparison: Accuracy, F1, ROC-AUC, Kappa, MCC, Time."""
    models = [m["model"] for m in all_metrics]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    panels = [
        ("accuracy", "Accuracy", True),
        ("f1_macro", "F1-Score (Macro)", True),
        ("roc_auc_macro", "ROC-AUC (Macro)", True),
        ("cohen_kappa", "Cohen's Kappa", True),
        ("mcc", "Matthews Corr. Coeff.", True),
        ("train_time_s", "Training Time (s)", False),
    ]

    for idx, (key, title, is_score) in enumerate(panels):
        ax = axes[idx // 3][idx % 3]
        values = [m.get(key, 0) for m in all_metrics]
        colors = [MODEL_COLORS.get(name, "#95a5a6") for name in models]
        bars = ax.bar(models, values, color=colors, edgecolor="white")
        ax.set_title(title, fontweight="bold")
        if is_score:
            ax.set_ylim(0, 1.15)
        for bar, val in zip(bars, values):
            label = f"{val:.4f}" if is_score else f"{val:.1f}s"
            y_pos = bar.get_height() + (0.02 if is_score else max(values) * 0.02)
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos, label,
                    ha="center", fontweight="bold", fontsize=10)

    plt.suptitle("Comprehensive Model Comparison", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "eval_model_comparison.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print("  Saved: eval_model_comparison.png")

# =====================================================================
#  7. LOG LOSS COMPARISON
# =====================================================================
def plot_log_loss(all_metrics):
    """Plot log loss comparison (lower is better)."""
    models = [m["model"] for m in all_metrics]
    values = [m.get("log_loss", 0) for m in all_metrics]
    colors = [MODEL_COLORS.get(name, "#95a5a6") for name in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, values, color=colors, edgecolor="white")
    ax.set_title("Log Loss Comparison (Lower is Better)", fontweight="bold")
    ax.set_ylabel("Log Loss")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                f"{val:.4f}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "eval_log_loss.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print("  Saved: eval_log_loss.png")

# =====================================================================
#  8. NORMALIZED CONFUSION MATRICES (percentage)
# =====================================================================
def plot_normalized_confusion_matrices(predictions):
    """Plot percentage-based confusion matrices."""
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    if n == 1: axes = [axes]
    for i, res in enumerate(predictions):
        cm = confusion_matrix(res["y_test"], res["y_pred"], normalize="true")
        sns.heatmap(cm, annot=True, fmt=".2%", cmap="Purples", ax=axes[i],
                    xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES)
        axes[i].set_title(f"{res['model']} (Normalized)", fontweight="bold")
        axes[i].set_xlabel("Predicted"); axes[i].set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "eval_confusion_normalized.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print("  Saved: eval_confusion_normalized.png")

# =====================================================================
#  9. METRICS RADAR CHART
# =====================================================================
def plot_radar_chart(all_metrics):
    """Plot radar/spider chart comparing models across key metrics."""
    categories = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Kappa", "MCC"]
    keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_macro", "cohen_kappa", "mcc"]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    for m in all_metrics:
        values = [m.get(k, 0) for k in keys]
        values += values[:1]
        color = MODEL_COLORS.get(m["model"], "#95a5a6")
        ax.plot(angles, values, "o-", linewidth=2, label=m["model"], color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Performance Radar Chart", fontweight="bold", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "eval_radar_chart.png"), bbox_inches="tight", dpi=150)
    plt.close()
    print("  Saved: eval_radar_chart.png")

# =====================================================================
#  10. SUMMARY TABLE PRINT
# =====================================================================
def print_summary_table(all_metrics):
    """Print a comprehensive metrics summary table to console."""
    print("\n  " + "=" * 90)
    header = f"  {'Metric':<30}"
    for m in all_metrics:
        header += f"{m['model']:>18}"
    print(header)
    print("  " + "-" * 90)

    rows = [
        ("Accuracy", "accuracy"),
        ("Precision (Macro)", "precision_macro"),
        ("Precision (Weighted)", "precision_weighted"),
        ("Recall (Macro)", "recall_macro"),
        ("Recall (Weighted)", "recall_weighted"),
        ("F1-Score (Macro)", "f1_macro"),
        ("F1-Score (Weighted)", "f1_weighted"),
        ("ROC-AUC (Macro)", "roc_auc_macro"),
        ("ROC-AUC (Weighted)", "roc_auc_weighted"),
        ("Cohen's Kappa", "cohen_kappa"),
        ("Matthews Corr. Coeff.", "mcc"),
        ("Log Loss", "log_loss"),
        ("Training Time (s)", "train_time_s"),
    ]

    for label, key in rows:
        line = f"  {label:<30}"
        for m in all_metrics:
            val = m.get(key, 0)
            if key == "train_time_s":
                line += f"{val:>18.1f}"
            else:
                line += f"{val:>18.4f}"
        print(line)

    print("  " + "=" * 90)

# =====================================================================
#  MAIN
# =====================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60 + "\n  STAGE 8d: COMPREHENSIVE MODEL EVALUATION\n  Mohammed Badawi - 2105656\n" + "=" * 60)

    predictions = load_predictions()
    if not predictions:
        print("  [ERROR] No prediction files found. Train models first.")
        sys.exit(1)

    # Compute all metrics
    print("\n  Computing comprehensive metrics...")
    all_metrics = [compute_all_metrics(res) for res in predictions]

    # Print summary table
    print_summary_table(all_metrics)

    # Generate all visualizations
    print("\n  Generating evaluation visualizations...")
    plot_confusion_matrices(predictions)
    plot_normalized_confusion_matrices(predictions)
    plot_roc_curves(predictions)
    plot_per_class_metrics(all_metrics)
    plot_model_comparison(all_metrics)
    plot_log_loss(all_metrics)
    plot_radar_chart(all_metrics)

    # Save comprehensive CSV
    summary_df = pd.DataFrame(all_metrics)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
    print(f"\n  Summary saved: {OUTPUT_DIR}/model_comparison.csv")

    # Save per-class detailed results
    per_class_rows = []
    for m in all_metrics:
        for cls in TARGET_NAMES:
            per_class_rows.append({
                "model": m["model"], "class": cls,
                "precision": m.get(f"precision_{cls}", 0),
                "recall": m.get(f"recall_{cls}", 0),
                "f1": m.get(f"f1_{cls}", 0),
                "support": m.get(f"support_{cls}", 0)
            })
    pd.DataFrame(per_class_rows).to_csv(os.path.join(OUTPUT_DIR, "per_class_metrics.csv"), index=False)
    print(f"  Per-class saved: {OUTPUT_DIR}/per_class_metrics.csv")

    print(f"\n  Total figures generated: 7")
    print(f"  Total metrics computed: {len(all_metrics[0]) - 1} per model")
    print("\n" + "=" * 60 + "\n  EVALUATION COMPLETE\n" + "=" * 60 + "\n")
