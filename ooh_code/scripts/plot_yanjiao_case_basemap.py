"""Publication-style map for the Beijing Yanjiao case study.

The script uses the generated Yanjiao instance coordinates and a light
web-tile basemap. It intentionally keeps dependencies small: matplotlib,
numpy, requests, and Pillow.
"""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageEnhance
import requests


ROOT = Path(__file__).resolve().parents[1]
INSTANCE_DIR = ROOT / "Environments" / "OOH" / "Beijing_Yanjiao"
DEFAULT_PREFIX = "yanjiao_400_0"
DEFAULT_OUT = ROOT / "Experiments" / "analysis" / "yanjiao_case_map_scientific_seed0_400.png"
TILE_CACHE = ROOT / "Experiments" / "analysis" / "_tile_cache"
R_MAJOR = 6378137.0
TILE_SIZE = 256


def lonlat_to_mercator(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = R_MAJOR * np.deg2rad(lon)
    lat = np.clip(lat, -85.05112878, 85.05112878)
    y = R_MAJOR * np.log(np.tan(np.pi / 4.0 + np.deg2rad(lat) / 2.0))
    return x, y


def mercator_to_lonlat(x: float, y: float) -> tuple[float, float]:
    lon = math.degrees(x / R_MAJOR)
    lat = math.degrees(2.0 * math.atan(math.exp(y / R_MAJOR)) - math.pi / 2.0)
    return lon, lat


def lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 2**z
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tile_bounds_mercator(x: int, y: int, z: int) -> tuple[float, float, float, float]:
    n = 2**z
    lon_w = x / n * 360.0 - 180.0
    lon_e = (x + 1) / n * 360.0 - 180.0
    lat_n = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_s = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    xw, yn = lonlat_to_mercator(np.array([lon_w]), np.array([lat_n]))
    xe, ys = lonlat_to_mercator(np.array([lon_e]), np.array([lat_s]))
    return float(xw[0]), float(xe[0]), float(ys[0]), float(yn[0])


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


def expand_bounds_xy(points_xy: np.ndarray, pad_frac: float) -> tuple[float, float, float, float]:
    xmin, ymin = points_xy.min(axis=0)
    xmax, ymax = points_xy.max(axis=0)
    dx = max(xmax - xmin, 1.0)
    dy = max(ymax - ymin, 1.0)
    return xmin - dx * pad_frac, xmax + dx * pad_frac, ymin - dy * pad_frac, ymax + dy * pad_frac


def fetch_tile(z: int, x: int, y: int, timeout: int = 20) -> Image.Image:
    cache_path = TILE_CACHE / str(z) / str(x) / f"{y}.png"
    if cache_path.exists():
        return Image.open(cache_path).convert("RGB")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    subdomain = "abcd"[(x + y) % 4]
    url = f"https://{subdomain}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png"
    headers = {"User-Agent": "DRT-Yanjiao-case-study-map/1.0"}
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    cache_path.write_bytes(response.content)
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def basemap_for_bounds(
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zoom: int,
    alpha: float = 0.78,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    lon_w, lat_s = mercator_to_lonlat(xlim[0], ylim[0])
    lon_e, lat_n = mercator_to_lonlat(xlim[1], ylim[1])
    tx0, ty1 = lonlat_to_tile(lon_w, lat_s, zoom)
    tx1, ty0 = lonlat_to_tile(lon_e, lat_n, zoom)
    tx0, tx1 = sorted((tx0, tx1))
    ty0, ty1 = sorted((ty0, ty1))

    mosaic = Image.new("RGB", ((tx1 - tx0 + 1) * TILE_SIZE, (ty1 - ty0 + 1) * TILE_SIZE), "white")
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            tile = fetch_tile(zoom, tx, ty)
            mosaic.paste(tile, ((tx - tx0) * TILE_SIZE, (ty - ty0) * TILE_SIZE))

    # Make the basemap quieter for overlaid research symbols.
    mosaic = ImageEnhance.Color(mosaic).enhance(0.55)
    mosaic = ImageEnhance.Contrast(mosaic).enhance(0.92)
    arr = np.asarray(mosaic).astype(float) / 255.0
    arr = arr * alpha + np.ones_like(arr) * (1.0 - alpha)

    left, _, _, top = tile_bounds_mercator(tx0, ty0, zoom)
    _, right, bottom, _ = tile_bounds_mercator(tx1, ty1, zoom)
    return arr, (left, right, bottom, top)


def add_north_arrow(ax: plt.Axes, x: float = 0.93, y: float = 0.16) -> None:
    ax.annotate(
        "N",
        xy=(x, y + 0.08),
        xytext=(x, y),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#111827",
        arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#111827"),
        zorder=20,
    )


def add_scale_bar_m(ax: plt.Axes, length_km: float, loc: tuple[float, float] = (0.06, 0.08)) -> None:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    length = length_km * 1000.0
    x = xmin + (xmax - xmin) * loc[0]
    y = ymin + (ymax - ymin) * loc[1]
    ax.plot([x, x + length], [y, y], color="#111827", lw=2.8, solid_capstyle="butt", zorder=20)
    tick = (ymax - ymin) * 0.012
    ax.plot([x, x], [y - tick, y + tick], color="#111827", lw=1.4, zorder=20)
    ax.plot([x + length, x + length], [y - tick, y + tick], color="#111827", lw=1.4, zorder=20)
    txt = ax.text(
        x + length / 2,
        y - (ymax - ymin) * 0.035,
        f"{length_km:g} km",
        ha="center",
        va="top",
        fontsize=9,
        color="#111827",
        zorder=20,
    )
    txt.set_path_effects([pe.withStroke(linewidth=3, foreground="white", alpha=0.85)])


def style_map_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=12.5, fontweight="bold", loc="left", pad=9, color="#111827")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#4b5563")
        spine.set_linewidth(0.9)


def draw_density_contours(ax: plt.Axes, homes_xy: np.ndarray) -> None:
    x, y = homes_xy[:, 0], homes_xy[:, 1]
    counts, xedges, yedges = np.histogram2d(x, y, bins=34)
    if counts.max() <= 0:
        return
    xs = (xedges[:-1] + xedges[1:]) / 2
    ys = (yedges[:-1] + yedges[1:]) / 2
    levels = np.linspace(max(2, counts.max() * 0.18), counts.max(), 5)
    ax.contourf(xs, ys, counts.T, levels=levels, cmap="Blues", alpha=0.18, zorder=3)
    ax.contour(xs, ys, counts.T, levels=levels, colors="#1d4ed8", linewidths=0.45, alpha=0.35, zorder=4)


def plot(prefix: str, output: Path, dpi: int, sample_homes: int) -> None:
    coord_path = INSTANCE_DIR / f"{prefix}_coords_latlon.txt"
    metadata_path = INSTANCE_DIR / f"{prefix}_metadata.json"
    coords = read_coords(coord_path)
    depot_ll = coords["depot"]
    homes_ll = coords["homes"]
    mps_ll = coords["meeting_points"]
    if depot_ll.size == 0 or homes_ll.size == 0 or mps_ll.size == 0:
        raise ValueError(f"Incomplete coordinate data in {coord_path}")

    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    hx, hy = lonlat_to_mercator(homes_ll[:, 0], homes_ll[:, 1])
    mx, my = lonlat_to_mercator(mps_ll[:, 0], mps_ll[:, 1])
    dx, dy = lonlat_to_mercator(depot_ll[:, 0], depot_ll[:, 1])
    homes_xy = np.column_stack([hx, hy])
    mps_xy = np.column_stack([mx, my])
    depot_xy = np.column_stack([dx, dy])
    yanjiao_xy = np.vstack([homes_xy, mps_xy])
    full_xy = np.vstack([depot_xy, yanjiao_xy])

    main_xlim = expand_bounds_xy(yanjiao_xy, 0.16)[:2]
    main_bounds = expand_bounds_xy(yanjiao_xy, 0.16)
    full_bounds = expand_bounds_xy(full_xy, 0.08)

    rng = np.random.default_rng(7)
    if sample_homes > 0 and sample_homes < len(homes_xy):
        sample_idx = np.sort(rng.choice(len(homes_xy), size=sample_homes, replace=False))
        homes_plot = homes_xy[sample_idx]
    else:
        homes_plot = homes_xy

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(8.6, 8.0), constrained_layout=False)
    ax = fig.add_axes([0.055, 0.075, 0.89, 0.84])

    bg, bg_extent = basemap_for_bounds((main_bounds[0], main_bounds[1]), (main_bounds[2], main_bounds[3]), zoom=12)
    ax.imshow(bg, extent=bg_extent, origin="upper", zorder=0)
    ax.set_xlim(main_bounds[0], main_bounds[1])
    ax.set_ylim(main_bounds[2], main_bounds[3])
    style_map_axis(ax, "Yanjiao origin-side DRT case")

    draw_density_contours(ax, homes_xy)
    ax.scatter(
        homes_plot[:, 0],
        homes_plot[:, 1],
        s=15,
        c="#2563eb",
        alpha=0.48,
        linewidths=0,
        zorder=6,
        label="Sampled home locations",
    )
    ax.scatter(
        mps_xy[:, 0],
        mps_xy[:, 1],
        s=46,
        marker="s",
        c="#f97316",
        edgecolors="white",
        linewidths=0.65,
        alpha=0.92,
        zorder=8,
        label="Candidate meeting points",
    )

    add_scale_bar_m(ax, 5, (0.055, 0.075))
    add_north_arrow(ax, 0.94, 0.11)

    # Inset corridor map.
    inset = fig.add_axes([0.565, 0.665, 0.345, 0.235])
    inset_bg, inset_extent = basemap_for_bounds((full_bounds[0], full_bounds[1]), (full_bounds[2], full_bounds[3]), zoom=10)
    inset.imshow(inset_bg, extent=inset_extent, origin="upper", zorder=0)
    inset.set_xlim(full_bounds[0], full_bounds[1])
    inset.set_ylim(full_bounds[2], full_bounds[3])
    style_map_axis(inset, "Guomao-Yanjiao corridor")
    inset.plot(
        [depot_xy[0, 0], np.median(homes_xy[:, 0])],
        [depot_xy[0, 1], np.median(homes_xy[:, 1])],
        color="#475569",
        lw=2.0,
        alpha=0.70,
        zorder=5,
    )
    inset.scatter(homes_xy[:, 0], homes_xy[:, 1], s=6, c="#2563eb", alpha=0.35, linewidths=0, zorder=6)
    inset.scatter(mps_xy[:, 0], mps_xy[:, 1], s=14, marker="s", c="#f97316", alpha=0.75, linewidths=0, zorder=7)
    inset.scatter(depot_xy[:, 0], depot_xy[:, 1], s=95, marker="*", c="#111827", edgecolors="white", linewidths=0.5, zorder=9)
    inset.text(
        depot_xy[0, 0] + 2500,
        depot_xy[0, 1] - 1100,
        "Guomao",
        fontsize=7.5,
        color="#111827",
        path_effects=[pe.withStroke(linewidth=2, foreground="white", alpha=0.9)],
        zorder=10,
    )
    inset.text(
        np.median(homes_xy[:, 0]) - 4500,
        homes_xy[:, 1].max() + 900,
        "Yanjiao",
        fontsize=7.5,
        color="#111827",
        path_effects=[pe.withStroke(linewidth=2, foreground="white", alpha=0.9)],
        zorder=10,
    )

    # Show inset extent on main map.
    rect = Rectangle(
        (main_bounds[0], main_bounds[2]),
        main_bounds[1] - main_bounds[0],
        main_bounds[3] - main_bounds[2],
        fill=False,
        edgecolor="#111827",
        linewidth=1.1,
        alpha=0.70,
        zorder=8,
    )
    inset.add_patch(rect)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2563eb", markeredgecolor="none",
               alpha=0.60, markersize=7, label=f"Home locations ({len(homes_xy)})"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#f97316", markeredgecolor="white",
               markersize=8, label=f"Candidate meeting points ({len(mps_xy)})"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#111827", markeredgecolor="white",
               markersize=11, label="Guomao destination"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(0.99, 0.02),
        frameon=True,
        facecolor="white",
        edgecolor="#d1d5db",
        framealpha=0.92,
        fontsize=9,
    )

    seed = metadata.get("seed", "?")
    fig.text(0.055, 0.955, "Beijing Yanjiao-Guomao commuter DRT case study", fontsize=15, fontweight="bold", color="#111827")
    fig.text(
        0.055,
        0.928,
        f"Instance {prefix}: {len(homes_xy)} passengers, {len(mps_xy)} bus-stop meeting points, data seed {seed}",
        fontsize=9.5,
        color="#4b5563",
    )
    fig.text(
        0.945,
        0.035,
        "Basemap: CARTO Positron / OpenStreetMap contributors",
        ha="right",
        fontsize=7.5,
        color="#6b7280",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Wrote {output}")
    print(f"Wrote {output.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a publication-style Yanjiao case map")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--dpi", default=320, type=int)
    parser.add_argument("--sample_homes", default=220, type=int, help="Number of home points to show over the density layer")
    args = parser.parse_args()
    plot(args.prefix, Path(args.output), args.dpi, args.sample_homes)


if __name__ == "__main__":
    main()
