"""
plot_test_results.py
=====================
Bar chart comparativo dei risultati sul test set
per i modelli SAM2-Tiny su Landslide4Sense.

Uso:
    python plot_test_results.py

Output:
    test_comparison.png
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================
# CONFIGURAZIONE
# ============================================================

BASE_DIR = "./"

MODELS = {
    "Adapter":        "history_sam2tiny_adapter_h5.json",
    "Baseline":       "history_sam2tiny_baseline_l4s.json",
    "LoRA r8":        "history_sam2tiny_lora_r8_l4s_alldataset.json",
    "UnifiedStream":  "history_sam2tiny_unifiedstream_alldata.json",
}

COLORS = {
    "Adapter":        "#E63946",
    "Baseline":       "#457B9D",
    "LoRA r8":        "#2A9D8F",
    "UnifiedStream":  "#F4A261",
}

METRICS = [
    ("test_iou",       "IoU Landslide"),
    ("test_miou",      "mIoU"),
    ("test_f1",        "F1"),
    ("test_precision", "Precision"),
    ("test_recall",    "Recall"),
    ("test_loss",      "Loss"),
]

OUTPUT_FILE = "test_comparison.png"


# ============================================================
# CARICAMENTO
# ============================================================

def load_test_results(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    for entry in data:
        if "test_results" in entry:
            return entry["test_results"]
    # Fallback: cerca chiavi test_* direttamente nell'ultimo elemento
    last = data[-1]
    if any(k.startswith("test_") for k in last):
        return last
    return None


def load_all():
    results = {}
    for name, filename in MODELS.items():
        path = os.path.join(BASE_DIR, filename)
        if not os.path.exists(path):
            print(f"  ⚠ File non trovato: {path}")
            continue
        try:
            test = load_test_results(path)
            if test is None:
                print(f"  ⚠ Nessun test_results in: {filename}")
                continue
            results[name] = test
            iou = test.get("test_iou", float("nan"))
            print(f"  ✓ {name}: IoU={iou:.4f}")
        except Exception as e:
            print(f"  ✗ Errore {name}: {e}")
    return results


# ============================================================
# PLOT
# ============================================================

def make_test_dashboard(results):
    model_names = list(results.keys())
    n_metrics   = len(METRICS)
    n_cols      = 3
    n_rows      = (n_metrics + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor("#0D1117")
    fig.suptitle(
        "SAM2-Tiny — Risultati Test Set · Landslide4Sense",
        fontsize=17, fontweight="bold", color="#F0F6FC", y=0.98,
    )

    gs = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        hspace=0.55, wspace=0.35,
        top=0.91, bottom=0.12, left=0.06, right=0.97,
    )

    x = np.arange(len(model_names))
    bar_w = 0.55

    for idx, (metric_key, metric_label) in enumerate(METRICS):
        row, col = divmod(idx, n_cols)
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor("#161B22")
        ax.tick_params(colors="#8B949E", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363D")
        ax.grid(True, axis="y", color="#21262D", linewidth=0.8, linestyle="--")
        ax.set_title(metric_label, color="#F0F6FC", fontsize=11,
                     fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=20, ha="right",
                           fontsize=8, color="#C9D1D9")

        values = [results[n].get(metric_key, np.nan) for n in model_names]
        bar_colors = [COLORS[n] for n in model_names]

        bars = ax.bar(x, values, width=bar_w, color=bar_colors,
                      alpha=0.85, edgecolor="#0D1117", linewidth=0.8)

        # Valore sopra ogni barra
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (max(v for v in values if not np.isnan(v)) * 0.02),
                    f"{val:.4f}",
                    ha="center", va="bottom",
                    fontsize=7.5, color="#F0F6FC", fontweight="bold",
                )

        # Evidenzia il best (max per tutto tranne loss, min per loss)
        valid = [(i, v) for i, v in enumerate(values) if not np.isnan(v)]
        if valid:
            if "loss" in metric_key:
                best_i = min(valid, key=lambda t: t[1])[0]
            else:
                best_i = max(valid, key=lambda t: t[1])[0]
            bars[best_i].set_edgecolor("white")
            bars[best_i].set_linewidth(2.0)
            bars[best_i].set_alpha(1.0)

        # Scala y con margine
        valid_vals = [v for v in values if not np.isnan(v)]
        if valid_vals:
            ymax = max(valid_vals) * 1.18
            ymin = min(valid_vals) * 0.85 if min(valid_vals) > 0 else 0
            ax.set_ylim(ymin, ymax)

    # Legenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS[n], label=n, alpha=0.85)
        for n in model_names if n in COLORS
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center", ncol=len(legend_elements),
        fontsize=9, framealpha=0.0,
        labelcolor="#C9D1D9",
        bbox_to_anchor=(0.5, 0.01),
    )

    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n✓ Plot salvato: {OUTPUT_FILE}")
    plt.close()


# ============================================================
# TABELLA RIASSUNTIVA
# ============================================================

def print_summary(results):
    cols = [k for k, _ in METRICS]
    labels = [l for _, l in METRICS]
    print("\n" + "="*80)
    header = f"  {'Modello':<18}" + "".join(f"{l:>12}" for l in labels)
    print(header)
    print("="*80)
    for name, test in results.items():
        row = f"  {name:<18}"
        for key in cols:
            val = test.get(key, float("nan"))
            row += f"{val:>12.4f}"
        print(row)
    print("="*80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("Caricamento test results...")
    results = load_all()
    if not results:
        print("Nessun risultato trovato.")
    else:
        print_summary(results)
        print("\nGenerazione plot...")
        make_test_dashboard(results)