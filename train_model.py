"""
train_model.py
--------------
Re-creates the pipeline from your notebook (frauddetection_2.ipynb)
and saves the trained XGBoost model to  model.joblib

Usage:
    python train_model.py path/to/PS_20174392719_1491204439457_log.csv.zip
"""
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "PS_20174392719_1491204439457_log.csv.zip"
MODEL_PATH = "model.joblib"
THRESHOLD = 0.4  # chosen in the notebook

FEATURE_COLUMNS = [
    "step",
    "amount",
    "origin_transaction_count",
    "origin_avg_amount",
    "origin_max_amount",
    "amount_vs_origin_avg",
    "destination_unique_origins",
    "destination_transaction_count",
    "origin_amount_ratio",
    "origin_remaining_ratio",
    "destination_amount_ratio",
    "destination_balance_ratio",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # ---- sort by time -------------------------------------------------
    df = df.sort_values("step", kind="stable").reset_index(drop=True)

    # ---- origin account history ---------------------------------------
    origin = df.groupby("nameOrig", sort=False)
    df["origin_transaction_count"] = origin.cumcount()

    origin_previous_sum = origin["amount"].cumsum() - df["amount"]
    df["origin_avg_amount"] = origin_previous_sum / df["origin_transaction_count"]

    df["origin_max_amount"] = (
        origin["amount"].cummax().groupby(df["nameOrig"], sort=False).shift()
    )

    # ---- destination account history ----------------------------------
    destination = df.groupby("nameDest", sort=False)
    df["destination_transaction_count"] = destination.cumcount()

    dest_origin = pd.factorize(
        pd.MultiIndex.from_arrays([df["nameDest"], df["nameOrig"]])
    )[0]
    pair_occurrence = pd.Series(dest_origin).groupby(dest_origin, sort=False).cumcount()
    new_origin = (pair_occurrence == 0).astype("int8")
    df["destination_unique_origins"] = (
        new_origin.groupby(df["nameDest"], sort=False).cumsum() - new_origin
    )

    # ---- ratio features -----------------------------------------------
    df["origin_amount_ratio"] = np.where(df["oldbalanceOrg"] > 0, df["amount"] / df["oldbalanceOrg"], 0)
    df["origin_remaining_ratio"] = np.where(df["oldbalanceOrg"] > 0, df["newbalanceOrig"] / df["oldbalanceOrg"], 0)
    df["destination_amount_ratio"] = np.where(df["oldbalanceDest"] > 0, df["amount"] / df["oldbalanceDest"], 0)
    df["destination_balance_ratio"] = np.where(df["oldbalanceDest"] > 0, df["newbalanceDest"] / df["oldbalanceDest"], 0)
    df["amount_vs_origin_avg"] = np.where(df["origin_avg_amount"] > 0, df["amount"] / df["origin_avg_amount"], 0)

    # ---- one-hot encode transaction type ------------------------------
    df = pd.get_dummies(df, columns=["type"])

    out = df[FEATURE_COLUMNS + ["isFraud"]].copy()
    out["origin_avg_amount"] = out["origin_avg_amount"].fillna(0)
    out["origin_max_amount"] = out["origin_max_amount"].fillna(0)

    # NOTE: the notebook casts every feature to int. We keep that so the
    # model is trained exactly like in the notebook (app.py does the same).
    return out.astype(int)


def main():
    print("Loading data ...")
    df = pd.read_csv(DATA_PATH, compression="zip" if DATA_PATH.endswith(".zip") else None)
    print("Raw shape:", df.shape)

    data = build_features(df)
    X = data[FEATURE_COLUMNS]
    y = data["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    base = XGBClassifier(tree_method="hist", eval_metric="logloss", random_state=42)
    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "min_child_weight": [1, 3, 5],
    }
    search = RandomizedSearchCV(
        base, param_distributions=param_grid, n_iter=5,
        scoring="recall", n_jobs=-1, cv=2, random_state=42,
    )
    print("Tuning XGBoost ...")
    search.fit(X_train, y_train)
    model = search.best_estimator_
    print("Best params:", search.best_params_)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    print(f"\nEvaluation at threshold {THRESHOLD}")
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))

    joblib.dump({"model": model, "features": FEATURE_COLUMNS, "threshold": THRESHOLD}, MODEL_PATH)
    print(f"Saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
