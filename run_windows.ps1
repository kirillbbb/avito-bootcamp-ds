# Run from the project directory in PowerShell.
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# Put train.csv, test.csv and events.csv.gz into data/ first.
python solution.py --data-dir data --output submission.csv
