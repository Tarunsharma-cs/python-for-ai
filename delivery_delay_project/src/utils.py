"""Helper functions: load data, split it, build the preprocessing pipeline."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    CATEGORICAL_FEATURES, CLEAN_DATA, NUMERIC_FEATURES,
    RANDOM_STATE, TARGET, TEST_SIZE,
)


def load_clean() -> pd.DataFrame:
    return pd.read_csv(CLEAN_DATA)


def get_xy(df: pd.DataFrame):
    """Separate features (X) from the target (y)."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    return X, y


def get_split():
    """80/20 train-test split, stratified so both sets keep the same delayed rate."""
    X, y = get_xy(load_clean())
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric columns and one-hot encode categorical columns."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
