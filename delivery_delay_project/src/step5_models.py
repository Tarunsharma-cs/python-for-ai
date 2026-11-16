"""Step 5: Model development - train and compare four classifiers."""
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from config import RANDOM_STATE, REPORT_DIR
from utils import build_preprocessor, get_split

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}


def main():
    X_train, X_test, y_train, y_test = get_split()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for name, clf in MODELS.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", clf)])

        # Cross-validation on the TRAINING set only (used later to pick the best model)
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc").mean()

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        rows.append(
            {
                "model": name,
                "cv_auc": round(cv_auc, 4),
                "test_accuracy": round(accuracy_score(y_test, pred), 4),
                "test_precision": round(precision_score(y_test, pred), 4),
                "test_recall": round(recall_score(y_test, pred), 4),
                "test_f1": round(f1_score(y_test, pred), 4),
                "test_auc": round(roc_auc_score(y_test, proba), 4),
            }
        )

    results = pd.DataFrame(rows).sort_values("cv_auc", ascending=False)
    results.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    print(results.to_string(index=False))
    print(f"\nBest model by cross-validated AUC: {results.iloc[0]['model']}")


if __name__ == "__main__":
    main()
