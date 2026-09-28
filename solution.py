from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end bot detection solution")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run chronological validation instead of creating a submission",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    feature_dir = root / ".features"
    prepare = root / "prepare_features.py"
    trainer = root / "train_models.py"

    run([
        sys.executable, str(prepare),
        "--data-dir", str(args.data_dir),
        "--out-dir", str(feature_dir),
    ])

    command = [
        sys.executable, str(trainer),
        "--train", str(args.data_dir / "train.csv"),
        "--test", str(args.data_dir / "test.csv"),
        "--train-features", str(feature_dir / "train_features.pkl"),
        "--test-features", str(feature_dir / "test_features.pkl"),
        "--output", str(args.output),
    ]
    if args.validate:
        command.append("--validate")
    run(command)


if __name__ == "__main__":
    main()
