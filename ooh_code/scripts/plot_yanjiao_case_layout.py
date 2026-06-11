"""Plot the Beijing Yanjiao case-study layout from generated instance files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INSTANCE_DIR = ROOT / "Environments" / "OOH" / "Beijing_Yanjiao"
DEFAULT_PREFIX = "yanjiao_400_0"
DEFAULT_OUT = ROOT / "Experiments" / "analysis" / "yanjiao_case_layout_seed0_400.png"


def read_coords(path: Path) -> dict[str, np.ndarray]:
    depot = []
    homes = []
    meeting_points = []

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("NODE_COORD_SECTION"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            name, lon_s, lat_s = parts[:3]
            lon, lat = float(lon_s), float(lat_s)
            if name.startswith("DEPOT"):
                depot.append((lon, lat))
            elif "_HOME" in name:
                homes.append((lon, lat))
            elif "_MP_" in name:
                meeting_points.append((lon, lat))

    return {
        "depot": np.asarray(depot, dtype=float),
        "homes": np.asarray(homes, dtype=float),
        "meeting_points": np.asarray(meeting_points, dtype=float),
    }


def expand_bounds(points: np.ndarray, pad_frac: float = 0.08) -> tuple[float, float, float, float]:
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)
    dx = max(xmax - xmin, 1e-6)
    dy = max(ymax - ymin, 1e-6)
    return xmin - dx * pad_frac, xmax + dx * pad_frac, ymin - dy * pad_frac, ymax + dy * pad_frac


def add_scale_bar(ax: plt.Axes, lon: float, lat: float, km: float = 5.0) -> None:
    # Longitude degrees per km at Beijing latitude.
    deg = km / (111.32 * np.cos(np.deg2rad(lat)))
    ax.plot([lon, lon + deg], [lat, lat], color="#333333", lw=2.2, solid_capstyle="butt")
    tick = 0.002
    ax.plot([lon, lon], [lat - tick, lat + tick], color="#333333", lw=1.4)
    ax.plot([lon + deg, lon + deg], [lat - tick, lat + tick], color="#333333", lw=1.4)
    ax.text(lon + deg / 2, lat - 0.006, f"{km:g} km", ha="center", va="top", fontsize=8, color="#333333")


def style_geo_axes(ax: plt.Axes) -> None:
    ax.grid(True, color="#d9dde4", lw=0.7, alpha=0.8)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.tick_params(axis="both", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#9aa3af")
        spine.set_linewidth(0.8)


def plot_layout(prefix: str, output: Path, dpi: int) -> None:
    coord_path = INSTANCE_DIR / f"{prefix}_coords_latlon.txt"
    metadata_path = INSTANCE_DIR / f"{prefix}_metadata.json"
    if not coord_path.exists():
        raise FileNotFoundError(coord_path)

    coords = read_coords(coord_path)
    depot = coords["depot"]
    homes = coords["homes"]
    mps = coords["meeting_points"]
    if depot.size == 0 or homes.size == 0 or mps.size == 0:
        raise ValueError(f"Incomplete coordinate data in {coord_path}")

    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    all_points = np.vstack([depot, homes, mps])
    yanjiao_points = np.vstack([homes, mps])
    x0, x1, y0, y1 = expand_bounds(all_points, 0.05)
    zx0, zx1, zy0, zy1 = expand_bounds(yanjiao_points, 0.10)

    fig = plt.figure(figsize=(12.6, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.22, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    home_color = "#2563eb"
    mp_color = "#f97316"
    depot_color = "#111827"
    corridor_color = "#64748b"

    # Full corridor view.
    ax0.set_title("Yanjiao-Guomao Corridor", fontsize=13, fontweight="bold", pad=10)
    ax0.plot(
        [depot[0, 0], np.median(homes[:, 0])],
        [depot[0, 1], np.median(homes[:, 1])],
        color=corridor_color,
        lw=2.0,
        alpha=0.45,
        zorder=1,
    )
    ax0.scatter(homes[:, 0], homes[:, 1], s=14, color=home_color, alpha=0.42, edgecolors="none", zorder=2)
    ax0.scatter(
        mps[:, 0],
        mps[:, 1],
        s=44,
        marker="s",
        color=mp_color,
        alpha=0.78,
        edgecolors="white",
        linewidths=0.45,
        zorder=3,
    )
    ax0.scatter(depot[:, 0], depot[:, 1], s=140, marker="*", color=depot_color, edgecolors="white", linewidths=0.7, zorder=4)
    ax0.annotate("Guomao destination", xy=depot[0], xytext=(depot[0, 0] + 0.025, depot[0, 1] - 0.018),
                 arrowprops=dict(arrowstyle="-", color=depot_color, lw=0.8), fontsize=9, color=depot_color)
    ax0.text(np.median(homes[:, 0]), homes[:, 1].max() + 0.006, "Yanjiao origin area", ha="center", fontsize=9, color="#1f2937")
    ax0.set_xlim(x0, x1)
    ax0.set_ylim(y0, y1)
    style_geo_axes(ax0)
    add_scale_bar(ax0, x0 + (x1 - x0) * 0.08, y0 + (y1 - y0) * 0.10, km=10)

    # Yanjiao zoom view.
    ax1.set_title("Origin-Side Service Area", fontsize=13, fontweight="bold", pad=10)
    ax1.hexbin(homes[:, 0], homes[:, 1], gridsize=24, cmap="Blues", mincnt=1, alpha=0.24, linewidths=0, zorder=0)
    ax1.scatter(homes[:, 0], homes[:, 1], s=15, color=home_color, alpha=0.50, edgecolors="none", zorder=2)
    ax1.scatter(
        mps[:, 0],
        mps[:, 1],
        s=48,
        marker="s",
        color=mp_color,
        alpha=0.86,
        edgecolors="white",
        linewidths=0.55,
        zorder=3,
    )
    ax1.set_xlim(zx0, zx1)
    ax1.set_ylim(zy0, zy1)
    style_geo_axes(ax1)
    add_scale_bar(ax1, zx0 + (zx1 - zx0) * 0.06, zy0 + (zy1 - zy0) * 0.09, km=5)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", label=f"Home locations ({len(homes)})",
               markerfacecolor=home_color, markeredgecolor="none", markersize=7, alpha=0.65),
        Line2D([0], [0], marker="s", color="none", label=f"Candidate meeting points ({len(mps)})",
               markerfacecolor=mp_color, markeredgecolor="white", markersize=8, alpha=0.9),
        Line2D([0], [0], marker="*", color="none", label="Guomao destination",
               markerfacecolor=depot_color, markeredgecolor="white", markersize=11),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))

    scenario = metadata.get("scenario", "Yanjiao-Guomao many-to-one commuter DRT")
    seed = metadata.get("seed", "?")
    fig.suptitle(f"{scenario} | seed={seed}", fontsize=15, fontweight="bold")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    pdf_output = output.with_suffix(".pdf")
    fig.savefig(pdf_output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}")
    print(f"Wrote {pdf_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Beijing Yanjiao case-study layout")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="Instance prefix, e.g. yanjiao_400_0")
    parser.add_argument("--output", default=str(DEFAULT_OUT), help="Output PNG path")
    parser.add_argument("--dpi", default=220, type=int)
    args = parser.parse_args()

    plot_layout(args.prefix, Path(args.output), args.dpi)


if __name__ == "__main__":
    main()
