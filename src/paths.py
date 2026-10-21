"""Dataset selection. UAIS_SET=real (default) uses the real-image benchmark under
data/real, runs/real and results/real; UAIS_SET=synthetic uses the earlier fully
synthetic stimulus set under data/, runs/ and results/."""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SET = os.environ.get("UAIS_SET", "real")
if SET == "synthetic":
    DATA, RUNS, RESULTS = (os.path.join(ROOT, d) for d in ("data", "runs", "results"))
else:
    DATA, RUNS, RESULTS = (os.path.join(ROOT, d, SET) for d in ("data", "runs", "results"))
