from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import pandas as pd

from solution_features import build_features

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(
        args.data_dir / "train.csv",
        parse_dates=["cookie_created_at", "window_start_ts", "window_end_ts"],
    )
    test = pd.read_csv(
        args.data_dir / "test.csv",
        parse_dates=["cookie_created_at", "window_start_ts", "window_end_ts"],
    )
    events = pd.read_csv(args.data_dir / "events.csv.gz", parse_dates=["event_ts"])

    print("Building train features...", flush=True)
    train_features = build_features(events, train)
    print("Building test features...", flush=True)
    train_columns = [c for c in train_features.columns if c != "cookie_id"]
    test_features = build_features(events, test, feature_columns=train_columns)

    assert train_features.cookie_id.is_unique and len(train_features) == len(train)
    assert test_features.cookie_id.is_unique and len(test_features) == len(test)

    train_features.to_pickle(args.out_dir / "train_features.pkl")
    test_features.to_pickle(args.out_dir / "test_features.pkl")
    print("Feature files saved.", flush=True)


if __name__ == "__main__":
    main()
