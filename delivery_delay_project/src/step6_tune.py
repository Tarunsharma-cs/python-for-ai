"""Step 6: Hyperparameter tuning with GridSearchCV."""
import json

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from config import MODEL_DIR, RANDOM_STATE, REPORT_DIR
from utils import build_preprocessor, get_split

SEARCH_SPACE = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        {"model__C": [0.01, 0.1, 1, 10, 100]},
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {"model__max_depth": [3, 5, 7, 10, None], "model__min_samples_leaf": [1, 5, 10, 20]},
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [5, 10, 15, None],
            "model__min_samples_leaf": [1, 3, 5],
        },
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3, 4],
        },
    ),
}


def main():
    # Pick the best model from Step 5 using cross-validated AUC (not the test set)
    comparison = pd.read_csv(REPORT_DIR / "model_comparison.csv")
    best_name = comparison.sort_values("cv_auc", ascending=False).iloc[0]["model"]
    print("Tuning:", best_name)

    X_train, X_test, y_train, y_test = get_split()
    clf, grid = SEARCH_SPACE[best_name]
    pipe = Pipeline([("prep", build_preprocessor()), ("model", clf)])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    search.fit(X_train, y_train)

    print("Best parameters:", search.best_params_)
    print("Best cross-validated AUC:", round(search.best_score_, 4))

    joblib.dump(search.best_estimator_, MODEL_DIR / "best_model.joblib")
    with open(REPORT_DIR / "best_params.json", "w") as f:
        json.dump(
            {"model": best_name, "params": search.best_params_, "cv_auc": search.best_score_},
            f, indent=2, default=str,
        )
    print(f"Saved tuned model to {MODEL_DIR / 'best_model.joblib'}")


if __name__ == "__main__":
    main()
