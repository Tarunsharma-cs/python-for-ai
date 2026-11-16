"""Run the whole pipeline (Steps 1-7) in order:  python src/run_all.py"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
STEPS = [
    "step1_collect_explore.py",
    "step2_clean.py",
    "step3_visualize.py",
    "step4_split.py",
    "step5_models.py",
    "step6_tune.py",
    "step7_evaluate.py",
]

for step in STEPS:
    print(f"\n{'=' * 60}\nRunning {step}\n{'=' * 60}")
    subprocess.run([sys.executable, str(SRC / step)], check=True, cwd=SRC)

print("\nDone. Launch the web app with:  streamlit run app/app.py")
