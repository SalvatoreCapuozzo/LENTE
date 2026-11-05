import os
import json
import glob
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Metrics tracked at each epoch (train_/val_ prefixes) and in the test_results block.
METRICS = ["loss", "iou", "iou_bg", "miou", "f1", "precision", "recall"]

# Consistent colors for the three splits across all plots.
COLORS = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}


def parse_history(history_path):
    """
    Reads a HISTORY_*.json file and separates the curves into Train, Validation and Test.

    Returns:
        epochs : list of epoch indices
        train  : {metric: [values per epoch]}
        val    : {metric: [values per epoch]}
        test   : {metric: scalar value}  (empty if not present)
    """
    with open(history_path, "r") as f:
        records = json.load(f)

    epochs = []
    train = {m: [] for m in METRICS}
    val = {m: [] for m in METRICS}
    test = {}

    for rec in records:
        if "test_results" in rec:
            tr = rec["test_results"]
            for m in METRICS:
                if f"test_{m}" in tr:
                    test[m] = tr[f"test_{m}"]
            continue

        epochs.append(rec.get("epoch", len(epochs) + 1))
        for m in METRICS:
            train[m].append(rec.get(f"train_{m}"))
            val[m].append(rec.get(f"val_{m}"))

    return epochs, train, val, test


def plot_metric(ax, epochs, train, val, test, metric):
    """Draws the Train/Val curve and the Test reference line for a metric on an axis."""
    if any(v is not None for v in train[metric]):
        ax.plot(epochs, train[metric], color=COLORS["train"],
                label="Train", linewidth=2)
    if any(v is not None for v in val[metric]):
        ax.plot(epochs, val[metric], color=COLORS["val"],
                label="Validation", linewidth=2)
    if metric in test:
        ax.axhline(test[metric], color=COLORS["test"], linestyle="--",
                   linewidth=1.8, label=f"Test ({test[metric]:.4f})")

    ax.set_title(metric.upper(), fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)


def plot_history(history_path, out_root="history_plots"):
    """Generates one plot per metric + a summary figure for a single history."""
    name = os.path.splitext(os.path.basename(history_path))[0]
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)

    epochs, train, val, test = parse_history(history_path)
    print(f"[{name}] {len(epochs)} epochs, test {'present' if test else 'absent'}")

    # A dedicated plot for each metric (Train vs Validation vs Test).
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_metric(ax, epochs, train, val, test, metric)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{metric}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

    # Summary figure with all metrics in a grid.
    n = len(METRICS)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))
    axes = axes.flatten()
    for ax, metric in zip(axes, METRICS):
        plot_metric(ax, epochs, train, val, test, metric)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(name, fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    summary_path = os.path.join(out_dir, "summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  -> saved {n + 1} plots in {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot training histories (Train/Validation/Test).")
    parser.add_argument(
        "history", nargs="*",
        help="HISTORY_*.json file(s) to plot. If omitted, uses histories/*.json")
    parser.add_argument(
        "--out", default="history_plots",
        help="Output root folder (default: history_plots)")
    args = parser.parse_args()

    paths = args.history or sorted(glob.glob(os.path.join("histories", "*.json")))
    if not paths:
        print("No history file found.")
        return

    for path in paths:
        plot_history(path, args.out)

    print("Plot generation complete!")


if __name__ == "__main__":
    main()
