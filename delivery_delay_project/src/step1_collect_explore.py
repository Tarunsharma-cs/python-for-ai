"""Step 1: Data collection - load the dataset and explore it."""
import pandas as pd

from config import RAW_DATA, TARGET


def main():
    df = pd.read_csv(RAW_DATA)

    print("Shape (rows, columns):", df.shape)
    print("\nFirst 5 rows:\n", df.head().to_string())
    print("\nData types and non-null counts:")
    df.info()
    print("\nNumeric summary:\n", df.describe().round(2).to_string())
    print("\nMissing values per column:\n", df.isna().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
    print(f"\nTarget balance ('{TARGET}'):\n", df[TARGET].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    main()
