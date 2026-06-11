"""Clean academic map for the Beijing Yanjiao case study.

This figure emphasizes the demand-relevant origin-side service area. Candidate
meeting points are styled by how many homes include them in the k-nearest
adjacency matrix. Low-coverage stops are hidden because they are not meaningful
customer choices in the generated demand sample.
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
DEFAULT_OUTPUT = ROOT / "Experiments" / "analysis" / "yanjiao_case_map_academic_seed0_400.png"
TILE_CACHE = ROOT / "Experiments" / "analysis" / "_tile_cache"
TILE_SIZE = 256
R_MAJOR = 6378137.0


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


def fetch_tile(z: int, x: int, y: int, timeout: int = 20) -> Image.Image:
    cache_path = TILE_CACHE / str(z) / str(x) / f"{y}.png"
    if cache_path.exists():
        return Image.open(cache_path).convert("RGB")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    subdomain = "abcd"[(x + y) % 4]
    url = f"https://{subdomain}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png"
    response = requests.get(url, headers={"User-Agent": "DRT-Yanjiao-academic-map/1.0"}, timeout=timeout)
    response.raise_for_status()
    cache_path.write_bytes(response.content)
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def basemap_for_bounds(xlim: tuple[float, float], ylim: tuple[float, float], zoom: int, alpha: float) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    lon_w, lat_s = mercator_to_lonlat(xlim[0], ylim[0])
    lon_e, lat_n = mercator_to_lonlat(xlim[1], ylim[1])
    tx0, ty1 = lonlat_to_tile(lon_w, lat_s, zoom)
    tx1, ty0 = lonlat_to_tile(lon_e, lat_n, zoom)
    tx0, tx1 = sorted((tx0, tx1))
    ty0, ty1 = sorted((ty0, ty1))

    mosaic = Image.new("RGB", ((tx1 - tx0 + 1) * TILE_SIZE, (ty1 - ty0 + 1) * TILE_SIZE), "white")
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            mosaic.paste(fetch_tile(zoom, tx, ty), ((tx - tx0) * TILE_SIZE, (ty - ty0) * TILE_SIZE))

    mosaic = ImageEnhance.Color(mosaic).enhance(0.18)
    mosaic = ImageEnhance.Contrast(mosaic).enhance(0.82)
    mosaic = ImageEnhance.Brightness(mosaic).enhance(1.06)
    arr = np.asarray(mosaic).astype(float) / 255.0
    arr = arr * alpha + np.ones_like(arr) * (1.0 - alpha)

    left, _, _, top = tile_bounds_mercator(tx0, ty0, zoom)
    _, right, bottom, _ = tile_bounds_mercator(tx1, ty1, zoom)
    return arr, (left, right, bottom, top)


def read_coords(path: Path) -> dict[str, np.ndarray]:
    depot, homes, meeting_points = [], [], []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("NODE_COORD_SECTION"):
                continue
            name, lon_s, lat_s = line.split()[:3]
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


def expand(points: np.ndarray, pad: float) -> tuple[float, float, float, float]:
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)
    dx = max(xmax - xmin, 1.0)
    dy = max(ymax - ymin, 1.0)
    return xmin - dx * pad, xmax + dx * pad, ymin - dy * pad, ymax + dy * pad


def add_scale_bar(ax: plt.Axes, length_km: float, loc: tuple[float, float]) -> None:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    length = length_km * 1000.0
    x = xmin + (xmax - xmin) * loc[0]
    y = ymin + (ymax - ymin) * loc[1]
    ax.plot([x, x + length], [y, y], color="#1f2937", lw=2.2, solid_capstyle="butt", zorder=30)
    tick = (ymax - ymin) * 0.010
    ax.plot([x, x], [y - tick, y + tick], color="#1f2937", lw=1.1, zorder=30)
    ax.plot([x + length, x + length], [y - tick, y + tick], color="#1f2937", lw=1.1, zorder=30)
    text = ax.text(x + length / 2, y - (ymax - ymin) * 0.030, f"{length_km:g} km", ha="center", va="top", fontsize=8.5, color="#1f2937", zorder=31)
    text.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white", alpha=0.9)])


def add_north_arrow(ax: plt.Axes) -> None:
    ax.annotate(
        "N",
        xy=(0.945, 0.185),
        xytext=(0.945, 0.105),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#1f2937",
        arrowprops=dict(arrowstyle="-|>", lw=1.35, color="#1f2937"),
        zorder=30,
    )


def draw_home_density(ax: plt.Axes, homes_xy: np.ndarray) -> None:
    x, y = homes_xy[:, 0], homes_xy[:, 1]
    counts, xedges, yedges = np.histogram2d(x, y, bins=38)
    xs = (xedges[:-1] + xedges[1:]) / 2.0
    ys = (yedges[:-1] + yedges[1:]) / 2.0
    if counts.max() <= 0:
        return
    levels = np.linspace(max(2.0, counts.max() * 0.15), counts.max(), 6)
    ax.contourf(xs, ys, counts.T, levels=levels, cmap="Blues", alpha=0.16, zorder=4)
    ax.contour(xs, ys, counts.T, levels=levels, colors="#2563eb", linewidths=0.45, alpha=0.30, zorder=5)


def style_axis(ax: plt.Axes, title: str | None = None) -> None:
    if title:
        ax.set_title(title, fontsize=11.5, fontweight="bold", loc="left", pad=7, color="#111827")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#9ca3af")
        spine.set_linewidth(0.75)


def plot(prefix: str, output: Path, dpi: int) -> None:
    coord_path = INSTANCE_DIR / f"{prefix}_coords_latlon.txt"
    adj_path = INSTANCE_DIR / f"{prefix}_adjacency10.npy"
    meta_path = INSTANCE_DIR / f"{prefix}_metadata.json"

    coords = read_coords(coord_path)
    depot_ll = coords["depot"]
    homes_ll = coords["homes"]
    mps_ll = coords["meeting_points"]
    adj = np.load(adj_path)
    mp_use = adj[1:, :].sum(axis=0)

    hx, hy = lonlat_to_mercator(homes_ll[:, 0], homes_ll[:, 1])
    mx, my = lonlat_to_mercator(mps_ll[:, 0], mps_ll[:, 1])
    dx, dy = lonlat_to_mercator(depot_ll[:, 0], depot_ll[:, 1])
    homes_xy = np.column_stack([hx, hy])
    mps_xy = np.column_stack([mx, my])
    depot_xy = np.column_stack([dx, dy])
    yanjiao_xy = np.vstack([homes_xy, mps_xy])
    full_xy = np.vstack([depot_xy, yanjiao_xy])

    # Use an origin-side view with enough margin to show sparse eastern stops.
    xb0, xb1, yb0, yb1 = expand(yanjiao_xy, 0.13)
    fx0, fx1, fy0, fy1 = expand(full_xy, 0.08)

    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    seed = metadata.get("seed", "?")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11.5,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig = plt.figure(figsize=(8.15, 7.95))
    ax = fig.add_axes([0.055, 0.080, 0.89, 0.815])
    bg, bg_extent = basemap_for_bounds((xb0, xb1), (yb0, yb1), zoom=12, alpha=0.64)
    ax.imshow(bg, extent=bg_extent, origin="upper", zorder=0)
    ax.set_xlim(xb0, xb1)
    ax.set_ylim(yb0, yb1)
    style_axis(ax)

    draw_home_density(ax, homes_xy)
    ax.scatter(homes_xy[:, 0], homes_xy[:, 1], s=9, color="#2f80ed", alpha=0.28, linewidths=0, zorder=6)

    low = mp_use < 5
    moderate = (mp_use >= 5) & (mp_use < 25)
    high = mp_use >= 25
    sizes = 24 + 0.58 * np.clip(mp_use, 0, 120)

    ax.scatter(mps_xy[moderate, 0], mps_xy[moderate, 1], s=sizes[moderate], marker="s", facecolors="#f6ad55", edgecolors="white", linewidths=0.60, alpha=0.88, zorder=8)
    ax.scatter(mps_xy[high, 0], mps_xy[high, 1], s=sizes[high], marker="s", facecolors="#d95f02", edgecolors="white", linewidths=0.70, alpha=0.94, zorder=9)

    # Label the most demand-relevant meeting-point cluster and the sparse east side.
    top_idx = np.argsort(mp_use)[-3:]
    ax.scatter(mps_xy[top_idx, 0], mps_xy[top_idx, 1], s=sizes[top_idx] + 32, marker="s", facecolors="none", edgecolors="#7c2d12", linewidths=1.15, zorder=10)

    add_scale_bar(ax, 5, (0.055, 0.070))
    add_north_arrow(ax)

    inset = fig.add_axes([0.585, 0.662, 0.320, 0.218])
    ibg, ibg_extent = basemap_for_bounds((fx0, fx1), (fy0, fy1), zoom=10, alpha=0.58)
    inset.imshow(ibg, extent=ibg_extent, origin="upper", zorder=0)
    inset.set_xlim(fx0, fx1)
    inset.set_ylim(fy0, fy1)
    style_axis(inset, "Commuting corridor")
    inset.plot(
        [depot_xy[0, 0], np.median(homes_xy[:, 0])],
        [depot_xy[0, 1], np.median(homes_xy[:, 1])],
        color="#334155",
        lw=1.75,
        alpha=0.78,
        zorder=4,
    )
    inset.scatter(homes_xy[:, 0], homes_xy[:, 1], s=4, color="#2f80ed", alpha=0.30, linewidths=0, zorder=5)
    inset.scatter(mps_xy[:, 0], mps_xy[:, 1], s=10, marker="s", color="#d95f02", alpha=0.60, linewidths=0, zorder=6)
    inset.scatter(depot_xy[:, 0], depot_xy[:, 1], s=85, marker="*", color="#111827", edgecolors="white", linewidths=0.55, zorder=8)
    rect = Rectangle((xb0, yb0), xb1 - xb0, yb1 - yb0, fill=False, edgecolor="#111827", linewidth=0.95, alpha=0.68, zorder=9)
    inset.add_patch(rect)
    for text, xy, offset in [
        ("Guomao", depot_xy[0], (2600, -1100)),
        ("Yanjiao", (np.median(homes_xy[:, 0]), homes_xy[:, 1].max()), (-4300, 1000)),
    ]:
        t = inset.text(xy[0] + offset[0], xy[1] + offset[1], text, fontsize=7.5, color="#111827", zorder=10)
        t.set_path_effects([pe.withStroke(linewidth=2.3, foreground="white", alpha=0.92)])

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2f80ed", markeredgecolor="none", alpha=0.45, markersize=6, label=f"Home locations ({len(homes_xy)})"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#d95f02", markeredgecolor="white", markersize=7.5, label="Meeting point: high coverage"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#f6ad55", markeredgecolor="white", markersize=7.0, label="Meeting point: moderate coverage"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.030),
        frameon=True,
        facecolor="white",
        edgecolor="#e5e7eb",
        framealpha=0.94,
        fontsize=8.4,
        handletextpad=0.5,
        borderpad=0.65,
    )

    fig.text(0.055, 0.952, "Beijing Yanjiao origin-side service area", fontsize=14.2, fontweight="bold", color="#111827")
    fig.text(
        0.055,
        0.925,
        f"Instance {prefix}: {len(homes_xy)} sampled homes and {int((~low).sum())} demand-relevant bus-stop meeting points; low-coverage stops hidden",
        fontsize=8.8,
        color="#4b5563",
    )
    fig.text(0.945, 0.030, "Basemap: CARTO Positron / OpenStreetMap contributors", ha="right", fontsize=7.2, color="#6b7280")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Wrote {output}")
    print(f"Wrote {output.with_suffix('.pdf')}")
    print(f"Meeting-point coverage counts: low={int(low.sum())}, moderate={int(moderate.sum())}, high={int(high.sum())}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw a clean academic Yanjiao case-study map")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dpi", type=int, default=360)
    args = parser.parse_args()
    plot(args.prefix, Path(args.output), args.dpi)


if __name__ == "__main__":
    main()
