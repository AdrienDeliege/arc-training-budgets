#!/usr/bin/env python3
"""Generate the paper's ARC anatomy, protocol, and result figures."""

from __future__ import annotations

import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, Rectangle


HERE = Path(__file__).resolve().parents[1]
FIGURES = HERE / "figures"
TASK_PATH = HERE / "data" / "arc-agi-1" / "data" / "evaluation" / "a680ac02.json"

ARC_COLORS = [
    "#111111",  # black
    "#0074D9",  # blue
    "#FF4136",  # red
    "#2ECC40",  # green
    "#FFDC00",  # yellow
    "#AAAAAA",  # gray
    "#F012BE",  # magenta
    "#FF851B",  # orange
    "#7FDBFF",  # light blue
    "#870C25",  # maroon
]
ARC_CMAP = ListedColormap(ARC_COLORS)

VARC_PASS2 = np.array(
    [
        [26.12, 34.38, 16.13, 35.50],
        [34.88, 39.37, 32.37, 39.25],
        [34.25, 40.00, 33.12, 36.75],
        [28.75, 35.75, 28.63, 35.13],
        [29.75, 32.37, 30.13, 33.75],
    ]
)
TRM_PASS2 = np.array(
    [
        [2.38, 10.37, 1.87, 19.63],
        [28.00, 33.00, 29.63, 33.37],
        [29.00, 33.37, 31.37, 37.50],
        [28.00, 33.75, 28.75, 34.63],
        [31.87, 35.13, 32.25, 35.75],
    ]
)

PRETRAIN_LABELS = ["None", "51 / 40M", "51 / 130M", "1001 / 40M", "1001 / 130M"]
TTT_LABELS = ["51 / Low", "51 / High", "1001 / Low", "1001 / High"]
MODEL_COLORS = {"VARC": "#2A788E", "TRM": "#2A788E"}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.8,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def draw_grid(fig: plt.Figure, rect: tuple[float, float, float, float], grid: list[list[int]]) -> None:
    """Draw an ARC grid inside a normalized figure-coordinate rectangle."""
    x, y, width, height = rect
    rows, cols = len(grid), len(grid[0])
    cell = min(width / cols, height / rows)
    actual_width, actual_height = cols * cell, rows * cell
    x += (width - actual_width) / 2
    y += (height - actual_height) / 2
    ax = fig.add_axes([x, y, actual_width, actual_height])
    ax.imshow(np.asarray(grid), cmap=ARC_CMAP, vmin=-0.5, vmax=9.5, interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="#D7D7D7", linewidth=0.20)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_color("#4A4A4A")
        spine.set_linewidth(0.55)


def add_arrow(fig: plt.Figure, start: tuple[float, float], end: tuple[float, float], **kwargs) -> None:
    style = {"arrowstyle": "-|>", "mutation_scale": 8, "linewidth": 0.8, "color": "#444444"}
    style.update(kwargs)
    fig.add_artist(FancyArrowPatch(start, end, transform=fig.transFigure, **style))


def add_stage(
    fig: plt.Figure,
    rect: tuple[float, float, float, float],
    title: str,
    detail: str,
    facecolor: str = "#F3F3F3",
    detail_color: str = "#222222",
) -> None:
    x, y, width, height = rect
    fig.add_artist(
        Rectangle(
            (x, y),
            width,
            height,
            transform=fig.transFigure,
            facecolor=facecolor,
            edgecolor="#6A6A6A",
            linewidth=0.65,
        )
    )
    fig.text(
        x + width / 2,
        y + height * 0.67,
        title,
        ha="center",
        va="center",
        fontsize=6.5,
        weight="bold",
        linespacing=0.95,
    )
    fig.text(
        x + width / 2,
        y + height * 0.24,
        detail,
        ha="center",
        va="center",
        fontsize=5.9,
        linespacing=1.05,
        color=detail_color,
    )


def make_arc_anatomy_figure() -> None:
    data = json.loads(TASK_PATH.read_text())
    query = data["test"][0]

    fig = plt.figure(figsize=(4.80, 2.05), facecolor="white")
    examples = [*data["train"], query]
    headings = ["Demonstration 1", "Demonstration 2", "Demonstration 3", "Evaluation query"]
    centers = [0.135, 0.375, 0.615, 0.855]
    for index, (example, heading, center) in enumerate(zip(examples, headings, centers)):
        fig.text(center, 0.955, heading, fontsize=6.4, ha="center", va="top")
        draw_grid(fig, (center - 0.108, 0.520, 0.216, 0.370), example["input"])
        fig.text(center, 0.495, "input", ha="center", va="top", fontsize=5.9)
        add_arrow(fig, (center, 0.455), (center, 0.375), mutation_scale=9, linewidth=0.9)
        draw_grid(fig, (center - 0.090, 0.135, 0.180, 0.225), example["output"])
        output_label = "output" if index < 3 else "hidden target (scoring only)"
        fig.text(center, 0.105, output_label, ha="center", va="top", fontsize=5.6)

    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"arc_task_anatomy.{suffix}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def make_controlled_protocol_figure() -> None:
    fig = plt.figure(figsize=(4.80, 2.55), facecolor="white")

    # Pretraining row.
    fig.text(0.135, 0.760, "Pretraining\n(done once)", ha="right", va="center", fontsize=6.6, weight="bold")
    add_stage(fig, (0.150, 0.670, 0.165, 0.180), "Training tasks", "1302 demos",facecolor="#a7f2b6")
    add_arrow(fig, (0.318, 0.760), (0.345, 0.760))
    add_stage(fig,(0.350, 0.670, 0.165, 0.180),"Task variants","$V=51$ or $1001$",facecolor="#f2cfa7",detail_color="#f03405",)
    add_arrow(fig, (0.518, 0.760), (0.545, 0.760))
    add_stage(fig, (0.550, 0.670, 0.155, 0.180), "Train model", "$B=40M$ or $130M$",facecolor="#f2cfa7",detail_color="#f03405",)
    add_arrow(fig, (0.708, 0.760), (0.735, 0.760))
    add_stage(fig, (0.740, 0.670, 0.235, 0.180), "Shared checkpoint", "used budget config. $(V,B)$", facecolor="#a7f2b6")

    # TTT row.
    fig.text(0.135, 0.440, "TTT\n(per task)", ha="right", va="center", fontsize=6.6, weight="bold")
    add_stage(fig, (0.150, 0.350, 0.165, 0.180), "One new task", "$3$ demos",facecolor="#a7f2b6")
    add_arrow(fig, (0.318, 0.440), (0.345, 0.440))
    add_stage(fig, (0.350, 0.350, 0.165, 0.180), "Task variants", "$V=51$ or $1001$",facecolor="#f2cfa7",detail_color="#f03405",)
    add_arrow(fig, (0.518, 0.440), (0.545, 0.440))
    add_stage(fig, (0.550, 0.350, 0.155, 0.180), "Perform TTT", "$B=15k$ or $300k$",facecolor="#f2cfa7",detail_color="#f03405",)
    add_arrow(fig, (0.708, 0.440), (0.735, 0.440))
    add_stage(fig, (0.740, 0.350, 0.235, 0.180), "Predict and regroup", "for input query", facecolor="#a7f2b6")

    add_arrow(
        fig,
        (0.858, 0.665),
        (0.670, 0.535),
        mutation_scale=9,
        linewidth=0.95,
        color="#7eb4ed",
    )
    fig.text(
        0.780,
        0.595,
        "initialize TTT",
        ha="center",
        va="center",
        fontsize=6.0,
        color="#2473c7",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4},
    )

#    fig.text(
#        0.165,
#        0.225,
#        "Scratch runs omit the pretraining row and initialize TTT randomly.",
#        ha="left",
#        va="center",
#        fontsize=6.5,
#    )
#    fig.text(
#        0.165,
#        0.135,
#        "After scoring, the model is reset before adapting to the next evaluation task.",
#        ha="left",
#        va="center",
#        fontsize=6.5,
#    )
#    fig.text(
#        0.165,
#        0.045,
#        "Budgets count optimization samples; prediction views and candidates are separate.",
#        ha="left",
#        va="center",
#        fontsize=6.5,
#    )

    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"controlled_training_protocol.{suffix}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def make_heatmaps() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(4.80, 2.75), sharey=True)
    plt.subplots_adjust(left=0.205, right=0.875, bottom=0.245, top=0.855, wspace=0.12)

    image = None
    color_map = mpl.colormaps["viridis"].copy()
    color_map.set_under("#3F3F3F")
    normalizer = mpl.colors.Normalize(vmin=26, vmax=40)
    for ax, values, title in zip(axes, (VARC_PASS2, TRM_PASS2), ("VARC", "TRM")):
        image = ax.imshow(values, cmap=color_map, norm=normalizer, aspect="auto")
        ax.set_title(title, weight="bold", pad=5)
        ax.set_xticks(np.arange(4), ["51\n15k", "51\n300k", "1001\n15k", "1001\n300k"])
        ax.set_yticks(np.arange(5), PRETRAIN_LABELS)
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.3)
        ax.tick_params(which="minor", bottom=False, left=False)
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                rgba = color_map(normalizer(max(value, 26)))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                color = "white" if value < 26 or luminance < 0.48 else "#111111"
                display_value = Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
                ax.text(col, row, f"{display_value:.1f}", ha="center", va="center", fontsize=6.7, color=color)

    axes[0].set_ylabel(r"$\mathbf{Pretraining}$ variants / sample budget")
    fig.text(0.540, 0.10, r"$\mathbf{TTT}$ variants / sample budget", ha="center", fontsize=7.3)
    cbar_ax = fig.add_axes([0.910, 0.245, 0.025, 0.610])
    cbar = fig.colorbar(
        image,
        cax=cbar_ax,
        orientation="vertical",
        ticks=[26, 30, 34, 38, 40],
        extend="min",
        extendrect=True,
        extendfrac=0.08,
    )
    cbar.set_label("pass@2 (%)", labelpad=4)
    cbar.ax.tick_params(length=2, pad=2)

    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"protocol_heatmaps.{suffix}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def protocol_differences(values: np.ndarray) -> list[np.ndarray]:
    """Return paired pass@2 changes while holding all other grid factors fixed."""
    ttt_presentations = np.concatenate((values[:, 1] - values[:, 0], values[:, 3] - values[:, 2]))
    ttt_variants = np.concatenate((values[:, 2] - values[:, 0], values[:, 3] - values[:, 1]))
    pretrain_presentations = np.concatenate((values[2] - values[1], values[4] - values[3]))
    pretrain_variants = np.concatenate((values[3] - values[1], values[4] - values[2]))
    return [ttt_presentations, ttt_variants, pretrain_presentations, pretrain_variants]


def make_factor_effects() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(4.80, 3.20), sharex=True, sharey=True)
    plt.subplots_adjust(left=0.255, right=0.985, bottom=0.175, top=0.845, wspace=0.10)
    improvement_colors = mpl.colors.LinearSegmentedColormap.from_list(
        "improvement",
        [(0.0, "#B3261E"), (0.5, "#8A6D1D"), (1.0, "#1B7F3A")],
    )

    labels = [
        "TTT samples $B$\n15k $\\rightarrow$ 300k",
        "TTT variants $V$\n51 $\\rightarrow$ 1001",
        "Pretrain samples $B$\n40M $\\rightarrow$ 130M",
        "Pretrain variants $V$\n51 $\\rightarrow$ 1001",
    ]
    y_positions = np.arange(len(labels))[::-1]

    for ax, values, title in zip(axes, (VARC_PASS2, TRM_PASS2), ("VARC", "TRM")):
        color = MODEL_COLORS[title]
        for index, (y_pos, differences) in enumerate(zip(y_positions, protocol_differences(values))):
            if index % 2:
                ax.axhspan(y_pos - 0.43, y_pos + 0.43, color="#F4F4F4", zorder=0)

            jitter = np.linspace(-0.14, 0.14, len(differences))
            ax.hlines(
                y_pos,
                differences.min(),
                differences.max(),
                color="#A5A5A5",
                linewidth=0.8,
                zorder=1,
            )
            ax.scatter(
                differences,
                y_pos + jitter,
                s=16,
                color=color,
                alpha=0.78,
                edgecolor="white",
                linewidth=0.45,
                zorder=2,
            )
            median = float(np.median(differences))
            ax.scatter(
                median,
                y_pos,
                marker="D",
                s=30,
                color="#202020",
                edgecolor="white",
                linewidth=0.55,
                zorder=3,
            )

            positive = int(np.sum(differences > 0))
            ties = int(np.sum(np.isclose(differences, 0)))
            count_label = f"{positive}/{len(differences)} improve"
            count_color = improvement_colors(positive / len(differences))
            #if ties:
            #    count_label += f"; {ties} tie"
            ax.text(
                22.0,
                y_pos - 0.29,
                count_label,
                ha="right",
                va="center",
                fontsize=5.7,
                color=count_color,
                zorder=4,
            )

        ax.axvline(0, color="#303030", linewidth=0.8, zorder=1)
        ax.set_xlim(-11, 22.5)
        ax.set_ylim(-0.55, 3.55)
        ax.set_xticks([-10, -5, 0, 5, 10, 15, 20])
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.5, zorder=0)
        ax.set_title(title, weight="bold", pad=5)
        for spine in ax.spines.values():
            spine.set_color("#777777")
            spine.set_linewidth(0.6)

    axes[0].set_yticks(y_positions, labels)
    axes[0].tick_params(axis="y", length=0, pad=4)
    axes[1].tick_params(axis="y", length=0)
    fig.text(0.62, 0.065, "Change in pass@2 (percentage points)", ha="center", fontsize=7.3)
    #fig.text(
    #    0.62,
    #    0.960,
    #    "Dots: matched comparisons   |   diamond: median   |   positive favors the setting at right",
    #    ha="center",
    #    va="top",
    #    fontsize=6.4,
    #)

    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"protocol_factor_effects.{suffix}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    make_arc_anatomy_figure()
    make_controlled_protocol_figure()
    make_heatmaps()
    make_factor_effects()


if __name__ == "__main__":
    main()
