#!/usr/bin/env python3
"""Generate the archived within-label shuffle result figure."""

from pathlib import Path
import csv

import matplotlib.pyplot as plt


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "docs" / "within_label_results.csv"
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    labels = [f"S{row['shuffle_seed']}\nM{row['model_seed']}" for row in rows]
    deltas = [100 * float(row["macro_f1_delta"]) for row in rows]
    lower = [value - 100 * float(row["ci95_low"]) for value, row in zip(deltas, rows)]
    upper = [100 * float(row["ci95_high"]) - value for value, row in zip(deltas, rows)]
    mean_delta = sum(deltas) / len(deltas)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)
    fig.patch.set_facecolor("#f7f8fa")
    ax.set_facecolor("#f7f8fa")

    colors = ["#386cb0"] * 3 + ["#7fc97f"] * 3 + ["#f0027f"] * 3
    x = list(range(len(rows)))
    ax.errorbar(x, deltas, yerr=[lower, upper], fmt="none", ecolor="#667085", capsize=5)
    ax.scatter(x, deltas, c=colors, s=75, zorder=3)
    ax.axhline(0, color="#293241", linewidth=1.2)
    ax.axhline(mean_delta, color="#d97706", linewidth=1.5, linestyle="--")
    ax.text(8.45, mean_delta + 0.08, f"mean {mean_delta:+.2f} pp", color="#9a5704", ha="right")

    ax.set_xticks(x, labels)
    ax.set_ylabel("Macro-F1 change (percentage points)")
    ax.set_ylim(-1.2, 5.4)
    ax.grid(axis="y", color="#d9dee7", linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#aab2bf")
    ax.tick_params(axis="y", length=0)

    fig.text(0.09, 0.94, "Within-label rationale shuffling did not reduce Macro-F1", weight="bold", fontsize=20)
    fig.text(
        0.09,
        0.902,
        "Nine fixed-checkpoint evaluations; points show change from clean with paired-bootstrap 95% intervals",
        color="#5c677d",
        fontsize=11,
    )
    fig.text(
        0.09,
        0.025,
        "S = rationale shuffle seed; M = model/split seed. Internal EARAM-style experiment, not an official reproduction.",
        color="#5c677d",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.04, 0.07, 0.98, 0.85))
    output = root / "docs" / "within_label_shuffle_results.png"
    fig.savefig(output, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
