"""Plot recall differences from the archived measurements; no benchmark is run."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "implementation-audit"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def export(directory, rows, figure):
    with (directory / "recall_difference.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for extension in ("png", "pdf", "svg"):
        figure.savefig(directory / f"recall_difference.{extension}",
                       dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_performance_differences():
    data = json.loads((ROOT / "summary.json").read_text())
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert manifest["independent_builds_per_variant"] == 1
    assert manifest["build_threads"] == 16
    points = {
        (row["variant"], row["threads"], row["k"], row["ef"]): row
        for row in data["curves"]
    }
    rows = []
    for (variant, threads, k, ef), current in sorted(points.items()):
        if variant != "current" or threads != 1:
            continue
        upstream = points["upstream", threads, k, ef]
        for name, point in (("current", current), ("upstream", upstream)):
            assert point["recall"] == points[name, 16, k, ef]["recall"]
        rows.append({
            "k": k,
            "ef": ef,
            "upstream_recall": upstream["recall"],
            "current_recall": current["recall"],
            "recall_delta_pp": 100 * (current["recall"] - upstream["recall"]),
        })
    assert len(rows) == 33
    bound = max(abs(row["recall_delta_pp"]) for row in rows)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharey=True)
    for ax, k in zip(axes, (1, 10, 100)):
        selected = [row for row in rows if row["k"] == k]
        ax.axhline(0, color="#555555", lw=1)
        ax.plot([row["ef"] for row in selected],
                [row["recall_delta_pp"] for row in selected],
                color="#087f8c", marker="o", markersize=4, lw=1.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks([row["ef"] for row in selected
                       if row["ef"] & (row["ef"] - 1) == 0])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.set_ylim(-bound * 1.3, bound * 1.3)
        ax.set_title(f"Recall@{k}")
        ax.set_xlabel("Search queue (ef)")
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Current - original recall (percentage points)")
    fig.suptitle("Full SIFT1M: recall difference at matching k / ef",
                 fontsize=17, fontweight="bold", y=0.99)
    fig.text(0.5, 0.91, f"1,000,000 base vectors · 10,000 queries · "
             f"maximum absolute difference: {bound:.4f} percentage points",
             ha="center", fontsize=10)
    fig.text(0.5, 0.015,
             "Historical run: one independent build per variant, 16 physical build cores, "
             "one NUMA socket, no interleave.\n"
             "Recall is identical for 1 and 16 query threads within each variant; "
             "positive values favor current HVS. No uncertainty estimate.",
             ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.10, 1, 0.89))
    export(ROOT, rows, fig)


def plot_audit_differences():
    cases = json.loads((AUDIT / "audit-results.json").read_text())
    rows = []
    for case in cases:
        assert case["threads"] == 64
        assert case["same_index_different_ids"] == 0
        assert all(case["training_fields_equal"].values())
        for query in case["queries"]:
            expected = 100 * (query["current_recall"] - query["upstream_recall"])
            assert abs(expected - query["recall_delta_pp"]) < 1e-10
        maximum = max(abs(query["recall_delta_pp"]) for query in case["queries"])
        rows.append({
            "case": case["case"],
            "independent_build_max_abs_recall_delta_pp": maximum,
            "same_index_max_abs_recall_delta_pp": 0.0,
            "same_index_compared_ids": case["same_index_compared_ids"],
            "same_index_different_ids": case["same_index_different_ids"],
        })
    assert sum(case["query_configurations"] for case in cases) == 112
    assert sum(row["same_index_compared_ids"] for row in rows) == 974544
    fig, ax = plt.subplots(figsize=(11, 6.8))
    positions = list(range(len(rows)))
    values = [row["independent_build_max_abs_recall_delta_pp"] for row in rows]
    ax.barh(positions, values, color="#cb6139", height=0.6,
            label="Independent parallel builds")
    ax.scatter([0] * len(rows), positions, color="#087f8c", s=45, marker="D",
               zorder=3, label="Same index, original vs current query")
    for position, value in zip(positions, values):
        ax.text(value + 0.06, position, f"{value:.4f}", va="center", fontsize=10)
    ax.set_yticks(positions, [row["case"] for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(-0.12, max(values) + 0.8)
    ax.set_xlabel("Maximum absolute recall difference across k / ef (percentage points)")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    fig.suptitle("Implementation audit: fixed index vs independent builds",
                 fontsize=17, fontweight="bold", y=0.98)
    fig.text(0.5, 0.92,
             "10,000 base vectors per case · all 64 OpenMP threads · NUMA interleave",
             ha="center", fontsize=11)
    fig.text(0.5, 0.02,
             "Same index: 0 / 974,544 ordered IDs differ across 112 configurations.\n"
             "Independent builds: one build per variant and case; all training fields match, "
             "but graphs differ.\nCorrectness audit only; maxima are not average differences "
             "or statistical bounds.", ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.11, 1, 0.90))
    export(AUDIT, rows, fig)


if __name__ == "__main__":
    plot_performance_differences()
    plot_audit_differences()
    print("Saved both recall-difference plots (PNG/PDF/SVG) and their CSV data")
