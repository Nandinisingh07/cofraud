"""
scripts/evaluate.py
Evaluates the trained model on the held-out (never-seen) temporal test
set. Computes and reports precision/recall at production thresholds (0.26 REVIEW,
0.60 HOLD). Saves metrics to data/evaluation_results.json.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

MODEL_PATH = Path("data/model.pkl")
TEST_FEATURES_PATH = Path("data/test_features.csv")
OUT_PATH = Path("data/evaluation_results.json")

REVIEW_THRESHOLD = 0.26  # match app/db.py RISK_THRESHOLD (REVIEW tier and above)
HOLD_THRESHOLD = 0.60    # match app/db.py HOLD threshold


def get_metrics_at_threshold(y_test, y_scores, t: float):
    y_pred = (y_scores >= t).astype(int)
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(t),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def evaluate():
    bundle = joblib.load(MODEL_PATH)
    model, features = bundle["model"], bundle["features"]

    test_df = pd.read_csv(TEST_FEATURES_PATH)
    X_test = test_df[features]
    y_test = test_df["class"]

    y_scores = model.predict_proba(X_test)[:, 1]

    review_metrics = get_metrics_at_threshold(y_test, y_scores, REVIEW_THRESHOLD)
    hold_metrics = get_metrics_at_threshold(y_test, y_scores, HOLD_THRESHOLD)

    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_test, y_scores)

    step = max(len(pr_thresholds) // 200, 1)
    pr_curve = [
        {"threshold": float(t), "precision": float(p), "recall": float(r)}
        for t, p, r in zip(
            pr_thresholds[::step],
            pr_precision[::step],
            pr_recall[::step],
        )
    ]

    results = {
        "threshold_used": REVIEW_THRESHOLD,
        "threshold_label": f"at REVIEW threshold ({REVIEW_THRESHOLD})",
        "precision": review_metrics["precision"],
        "recall": review_metrics["recall"],
        "f1_score": review_metrics["f1_score"],
        "confusion_matrix": review_metrics["confusion_matrix"],
        "review_metrics": review_metrics,
        "hold_metrics": hold_metrics,
        "test_set_size": int(len(test_df)),
        "test_fraud_count": int(y_test.sum()),
        "pr_curve": pr_curve,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"REVIEW Threshold ({REVIEW_THRESHOLD}): Precision: {review_metrics['precision']:.4f}  Recall: {review_metrics['recall']:.4f}  F1: {review_metrics['f1_score']:.4f}")
    print(f"  Confusion matrix -> {review_metrics['confusion_matrix']}")
    print(f"HOLD Threshold ({HOLD_THRESHOLD}):   Precision: {hold_metrics['precision']:.4f}  Recall: {hold_metrics['recall']:.4f}  F1: {hold_metrics['f1_score']:.4f}")
    print(f"  Confusion matrix -> {hold_metrics['confusion_matrix']}")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    evaluate()

