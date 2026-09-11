"""
run_all.py — Runs the full pipeline end-to-end, reproducing all results
reported in the paper. Run from the repository root:

    python run_all.py

Outputs are written to results/.
"""
import subprocess
import sys

STEPS = [
    ("Loading dataset", "src/data.py"),
    ("Running classifiers (SVM + Naive Bayes)", "src/classifiers.py"),
    ("Deriving neutrosophic T/I/F and distance/entropy measures", "src/neutrosophic.py"),
    ("Running fuzzy c-means baseline", "src/fuzzy_baseline.py"),
    ("Evaluating: classification metrics + statistical validation", "src/evaluate.py"),
]

if __name__ == "__main__":
    for description, script in STEPS:
        print(f"\n{'=' * 60}\n{description}\n{'=' * 60}")
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"\nStep failed: {script}. Aborting.")
            sys.exit(1)

    print(f"\n{'=' * 60}\nAll steps completed. See results/evaluation_report.json\n{'=' * 60}")
