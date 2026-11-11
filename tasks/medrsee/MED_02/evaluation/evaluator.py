"""Validation bridge to the shared organizer scoring and authorization snapshot."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]/"benchmarks"))
from medrsee.entry import run_evaluation
