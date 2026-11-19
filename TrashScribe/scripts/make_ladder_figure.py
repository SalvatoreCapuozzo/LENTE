"""Figure of the paper's central comparison: the 34 M edge model against SAM 3 (848 M, zero-shot) on each
environment, as the edge model receives more data from the site.

    python scripts/make_ladder_figure.py        # -> paper/figures/ladder.pdf (+ .png preview)

Bars (one-hue ramp, light to dark = more site data): leave-one-dataset-out model (no site data, E3), fine-tuned
on SAM 3 labels of the site's unlabelled images (E9), fine-tuned with 10 labelled site images (E10), site in
training (E4). Dashed orange segment: SAM 3 zero-shot with the same half-image rule (16 prompts). Palette
validated with the dataviz skill's validator (ordinal ramp + orange vs blue, all checks pass). CPU only.
"""
import _bootstrap  # noqa: F401
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from trashscribe.engine import _p  # noqa: E402

RUNS = Path(_p("runs"))
DATASETS = [("taco", "TACO\nstreet"), ("zerowaste", "ZeroWaste-f\nconveyor"), ("uavvaste", "UAVVaste\naerial"),
            ("mju", "MJU-Waste\nindoor"), ("bepli", "BePLi v2\nbeach")]
BARS = [("e3_lodo_{d}", "no site data", "#86b6ef"),
        ("e9_sam3_{d}", "+ SAM 3 labels on site images", "#3987e5"),
        ("e10_k10_{d}", "+ 10 labelled site images", "#1c5cab"),
        ("e4_heldout_classes", "site in training", "#0d366b")]
SAM3 = "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def iou(run, d, key="seg"):
    f = RUNS / run / "test_metrics.json"
    if not f.exists():
        return None
    m = json.loads(f.read_text())
    v = ((m.get(key) or {}).get(d) or {}).get("iou")
    return None if v is None else 100 * v


def bar(ax, x, w, h, color):
    """Thin bar; the white edge leaves a gap between neighbouring bars."""
    ax.add_patch(Rectangle((x, 0), w, h, fc=color, ec="white", lw=0.8))


def main():
    plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED,
                         "xtick.color": INK, "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(5.16, 2.35))
    n, gw = len(BARS), 0.8
    w = gw / n
    for gi, (d, _) in enumerate(DATASETS):
        x0 = gi - gw / 2
        for bi, (tpl, _, c) in enumerate(BARS):
            v = iou(tpl.format(d=d), d)
            if v is not None:
                bar(ax, x0 + bi * w + 0.01, w - 0.02, v, c)
        s = iou("e4_heldout_classes__zeroshot_sam3obj_area", d, "where_sam3_area")
        if s is not None:
            ax.plot([x0 - 0.04, x0 + gw + 0.04], [s, s], color=SAM3, lw=1.6, ls=(0, (4, 2)), solid_capstyle="butt",
                    zorder=5)
            ax.text(x0 + gw + 0.05, s, f"{s:.0f}", color=MUTED, fontsize=6, va="center", ha="left")
    ax.set_xlim(-0.6, len(DATASETS) - 0.35)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(len(DATASETS)), [lab for _, lab in DATASETS])
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylabel("waste IoU (%)", color=MUTED)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    sam = [plt.Line2D([], [], color=SAM3, lw=1.6, ls=(0, (4, 2)), label="zero-shot, no site data")]
    l1 = fig.legend(handles=sam, loc="upper left", frameon=False, fontsize=6.5, bbox_to_anchor=(0.06, 1.0),
                    handlelength=2.2, title="SAM 3 (848 M param.)", title_fontsize=6.5, alignment="left")
    ts = [Rectangle((0, 0), 1, 1, fc=c, label=lab) for _, lab, c in BARS]
    fig.legend(handles=ts, loc="upper left", ncol=2, frameon=False, fontsize=6.5, bbox_to_anchor=(0.36, 1.0),
               handlelength=1.4, columnspacing=1.2, title="TrashScribe (34 M param.) with more data from the site",
               title_fontsize=6.5, alignment="left")
    fig.add_artist(l1)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.17, top=0.74)
    out = Path(_p("paper/figures/ladder"))
    fig.savefig(out.with_suffix(".pdf"), dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print("wrote", out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
