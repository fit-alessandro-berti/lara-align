"""Run the unchanged reference training protocol with explicit CPU threading.

Arguments after this script are passed to scripts/train_model.py. Checkpoints
stay in the local output directories named on that command line.
"""
import json
import os
from pathlib import Path
import platform
import runpy
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")), platform.processor())
print(json.dumps({"python": platform.python_version(), "torch": torch.__version__,
                  "cpu": cpu, "intra_threads": 1, "inter_threads": 1,
                  "argv": sys.argv[1:], "pid": os.getpid()}), flush=True)
sys.argv[0] = str(ROOT / "scripts/train_model.py")
runpy.run_path(sys.argv[0], run_name="__main__")
