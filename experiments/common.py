import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FIG = ROOT / "results" / "figures"
RES = ROOT / "results"
FIG.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID = "#1d2433", "#6b7385", "#e6e2da"
GOLD, TEAL, PINK, BLUE = "#b8862f", "#2a9d84", "#d0507a", "#3d6fb4"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11,
    "legend.frameon": False,
})


def save_json(name: str, obj) -> None:
    (RES / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False))
