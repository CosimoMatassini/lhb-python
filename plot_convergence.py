import re
import sys
from pathlib import Path

import matplot2tikz
import matplotlib.pyplot as plt

directory = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
pattern = re.compile(r"^(hb|lhb)_(.+)_iter_(\d+)_results\.txt$")
results = {}

for path in directory.glob("*_iter_*_results.txt"):
    match = pattern.match(path.name)
    if not match:
        continue
    method, name, iteration = match.groups()
    values = {
        key.strip(): value.strip()
        for key, value in (
            line.split(":", 1) for line in path.read_text().splitlines() if ":" in line
        )
    }
    results.setdefault(name, {}).setdefault(method, []).append(
        (int(values["DoFs"]), float(values["MSE"]), int(iteration))
    )

if not results:
    raise SystemExit(f"No result files found in {directory}")

for name, methods in results.items():
    fig, ax = plt.subplots()
    for method, marker in (("hb", "o"), ("lhb", "s")):
        data = sorted(methods.get(method, []))
        if not data:
            continue
        dofs, errors, _ = zip(*data)
        ax.loglog(dofs, errors, marker=marker, label=method.upper(), linewidth=2)

    ax.grid(True, which="major", linestyle=":", alpha=0.5)
    ax.legend()

    fig.tight_layout()

    matplot2tikz.save(directory / f"convergence_{name}.tex")

    output = directory / f"convergence_{name}.png"
    fig.savefig(output, dpi=500, bbox_inches="tight")
    plt.close(fig)
    print(output)
