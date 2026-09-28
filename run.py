from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("$", " ".join(map(str, command)), flush=True)
    subprocess.run(command, check=True)


def venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set up the environment and run the Avito DS solution."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run chronological validation instead of creating a submission.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    data_dir = (root / args.data_dir).resolve()
    venv_dir = root / ".venv"
    requirements = root / "requirements.txt"
    solution = root / "solution.py"

    if sys.version_info[:2] != (3, 13):
        raise SystemExit(
            f"Python 3.13 is required, but {sys.version.split()[0]} is active."
        )

    required_files = [
        data_dir / "train.csv",
        data_dir / "test.csv",
        data_dir / "events.csv.gz",
    ]
    missing = [path.name for path in required_files if not path.exists()]
    if missing:
        raise SystemExit(
            f"Missing data files in {data_dir}: {', '.join(missing)}. "
            "See data/README.md."
        )

    python = venv_python(venv_dir)

    if not python.exists():
        print("Creating virtual environment...", flush=True)
        run([sys.executable, "-m", "venv", str(venv_dir)])

    if not python.exists():
        raise SystemExit(f"Virtual environment Python was not found: {python}")

    print("Installing/updating dependencies...", flush=True)
    run([str(python), "-m", "pip", "install", "-r", str(requirements)])

    command = [
        str(python),
        str(solution),
        "--data-dir",
        str(data_dir),
        "--output",
        str((root / args.output).resolve()),
    ]
    if args.validate:
        command.append("--validate")

    run(command)


if __name__ == "__main__":
    main()
