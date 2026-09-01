#!/usr/bin/env python3
"""Generate the presentation figure for the internal rationale stress test."""

from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    labels = ["Wrong verdict\n(50% fields)", "No rationale", "Shuffled rationale\n(3 × 3 runs)"]
    changes = [0.14, -0.86, -1.39]
    colors = ["#8492a6", "#f0a35e", "#d85b5b"]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)
    fig.patch.set_facecolor("#f7f8fa")
    ax.set_facecolor("#f7f8fa")

    bars = ax.bar(labels, changes, color=colors, width=0.62)
    ax.axhline(0, color="#293241", linewidth=1.2)
    ax.grid(axis="y", color="#d9dee7", linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    ax.set_ylim(-1.75, 0.45)
    ax.set_ylabel("Macro-F1 change (percentage points)")
    fig.text(
        0.09,
        0.94,
        "Misaligned rationales hurt more than missing rationales",
        weight="bold",
        fontsize=20,
    )
    fig.text(
        0.09,
        0.905,
        "Internal low-memory EARAM-style experiment; clean mean Macro-F1 = 0.9193",
        color="#5c677d",
        fontsize=12,
    )

    for bar, value in zip(bars, changes):
        vertical = 0.05 if value >= 0 else -0.07
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + vertical,
            f"{value:+.2f} pp",
            ha="center",
            va="bottom" if value >= 0 else "top",
            weight="bold",
            color="#293241",
        )

    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#aab2bf")
    ax.tick_params(axis="y", length=0)
    fig.text(
        0.09,
        0.025,
        "Shuffled rationale averages 3 model seeds × 3 independent permutations. Results are not an official EARAM reproduction.",
        color="#5c677d",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.04, 0.07, 0.98, 0.85))

    output = Path(__file__).resolve().parents[1] / "docs" / "rationale_reliability_results.png"
    fig.savefig(output, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
