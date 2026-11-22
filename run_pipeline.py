import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
commands = [
    [sys.executable, "-m", "rag.ingest"],
    [sys.executable, "-m", "rag.index"],
    [sys.executable, "-m", "evaluation.evaluate_retrieval"],
]
for command in commands:
    print("\\n>>>", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
print("\\nPipeline completed. Review reports/retrieval_results.json.")
