"""Paths and constants shared by every pipeline step."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
METRICS = REPORTS / "metrics"  # small CSVs the DOCX report is built from (committed)
ARTIFACTS = ROOT / "artifacts"

TRAIN_CSV = DATA / "train_test.csv"
VALID_CSV = DATA / "validation.csv"
TEMPLATE_CSV = DATA / "validation_predictions_template.csv"
DECEMBER_CSV = DATA / "december_chart_inputs.csv"
PREDICTIONS_CSV = ROOT / "validation_predictions.csv"

SEED = 42
TARGET = "posted_rate"
EQUIPMENT = ["Dry Van", "Reefer", "Flatbed"]

# Chart styling (validated categorical palette, first three slots; recessive ink/grid).
COLORS = {"Dry Van": "#2a78d6", "Reefer": "#eb6834", "Flatbed": "#1baf7a"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
