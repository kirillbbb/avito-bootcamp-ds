
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from metric import precision_at_recall, recall_at_fpr

SEED = 42
FINAL_ITERATIONS = 400
VALIDATION_ITERATIONS = 250


def make_model(iterations=FINAL_ITERATIONS):
    return CatBoostClassifier(
        iterations=iterations,
        depth=6,
        learning_rate=0.04,
        l2_leaf_reg=6.0,
        loss_function="Logloss",
        random_seed=SEED,
        thread_count=4,
        verbose=False,
        allow_writing_files=False,
    )


def load_matrix(path, cookie_ids):
    features = pd.read_pickle(path).set_index("cookie_id").reindex(cookie_ids)
    matrix = features.to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Non-finite value found in {path}")
    return np.ascontiguousarray(matrix)


def validate_cutoff(train, features, cutoff, iterations):
    valid = train["window_start_ts"] >= pd.Timestamp(cutoff)
    if not valid.any() or valid.all():
        raise ValueError(f"Cutoff {cutoff} does not split the training data")
    X = features.set_index("cookie_id").reindex(train.cookie_id).to_numpy(np.float32)
    y = train.target.to_numpy(dtype=int)
    model = make_model(iterations)
    model.fit(X[~valid], y[~valid])
    score = model.predict_proba(X[valid])[:, 1]
    return (
        int(valid.sum()),
        int(y[valid].sum()),
        precision_at_recall(y[valid], score),
        recall_at_fpr(y[valid], score),
    )


def validate(train_path, features_path):
    train = pd.read_csv(train_path, parse_dates=["window_start_ts"])
    features = pd.read_pickle(features_path)
    print("Chronological validation:")
    results = []
    for cutoff in ("2026-04-17", "2026-04-18", "2026-04-19"):
        n, positives, precision, recall = validate_cutoff(
            train, features, cutoff, VALIDATION_ITERATIONS
        )
        results.append(precision)
        print(
            f"cutoff={cutoff} | n={n} | positives={positives} "
            f"| P@R>=0.70={precision:.6f} | Recall@1%FPR={recall:.6f}"
        )
    print(f"Mean P@R>=0.70: {np.mean(results):.6f}")


def validate_inputs(train, test):
    if train.cookie_id.duplicated().any():
        raise ValueError("Duplicate cookie_id in train.csv")
    if test.cookie_id.duplicated().any():
        raise ValueError("Duplicate cookie_id in test.csv")
    if set(train.cookie_id) & set(test.cookie_id):
        raise ValueError("train.csv and test.csv contain overlapping cookie_id values")
    if not train.target.isin([0, 1]).all():
        raise ValueError("target must contain only 0 and 1")


def train_and_submit(train_path, test_path, train_features_path, test_features_path, output_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    validate_inputs(train, test)
    X_train = load_matrix(train_features_path, train.cookie_id)
    X_test = load_matrix(test_features_path, test.cookie_id)
    y = train.target.to_numpy(dtype=int)
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Train and test feature matrices have different widths")
    model = make_model()
    model.fit(X_train, y)
    score = np.clip(model.predict_proba(X_test)[:, 1], 0.0, 1.0)
    submission = pd.DataFrame({"cookie_id": test.cookie_id, "score": score})
    if len(submission) != len(test):
        raise ValueError("Submission row count does not match test.csv")
    if not submission.cookie_id.is_unique:
        raise ValueError("Submission contains duplicate cookie_id")
    if not submission.score.between(0.0, 1.0).all():
        raise ValueError("Submission score is outside [0, 1]")
    if set(submission.cookie_id) != set(test.cookie_id):
        raise ValueError("Submission cookie_id set differs from test.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Saved {len(submission)} predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train and submit the bot detector")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--test-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    validate_inputs(train, test)
    if args.validate:
        validate(args.train, args.train_features)
        return
    train_and_submit(
        args.train, args.test, args.train_features, args.test_features, args.output
    )


if __name__ == "__main__":
    main()
