"""Step 4: Train-test split (80% / 20%)."""
from config import CLEAN_DATA, TARGET
from utils import get_split


def main():
    X_train, X_test, y_train, y_test = get_split()

    print("Features used:", list(X_train.columns))
    print("X_train:", X_train.shape, "| X_test:", X_test.shape)
    print(f"Delayed rate - train: {y_train.mean():.3f} | test: {y_test.mean():.3f}")

    out = CLEAN_DATA.parent
    X_train.assign(**{TARGET: y_train}).to_csv(out / "train.csv", index=False)
    X_test.assign(**{TARGET: y_test}).to_csv(out / "test.csv", index=False)
    print(f"Saved train.csv and test.csv to {out}")


if __name__ == "__main__":
    main()
