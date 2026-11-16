"""Step 7: Model evaluation - metrics, confusion matrix, ROC curve."""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, classification_report,
    f1_score, precision_score, recall_score, roc_auc_score,
)

from config import FIG_DIR, MODEL_DIR, REPORT_DIR
from utils import get_split


def main():
    model = joblib.load(MODEL_DIR / "best_model.joblib")
    _, X_test, _, y_test = get_split()

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    print({k: round(v, 4) for k, v in metrics.items()})
    print("\n", classification_report(y_test, pred, target_names=["On time", "Delayed"]))

    # Confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["On time", "Delayed"], cmap="Blues"
    )
    plt.title("Confusion matrix (test set)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_confusion_matrix.png", dpi=150)
    plt.close()

    # ROC curve with AUC
    RocCurveDisplay.from_predictions(y_test, proba, name="Tuned model")
    plt.plot([0, 1], [0, 1], "k--", label="Random guess")
    plt.title(f"ROC curve (AUC = {metrics['roc_auc']:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "08_roc_curve.png", dpi=150)
    plt.close()

    with open(REPORT_DIR / "final_metrics.json", "w") as f:
        json.dump({k: round(v, 4) for k, v in metrics.items()}, f, indent=2)

    best = json.load(open(REPORT_DIR / "best_params.json"))["model"]
    print(f"Summary: best model = {best}, test accuracy = {metrics['accuracy']:.1%}, "
          f"AUC = {metrics['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
