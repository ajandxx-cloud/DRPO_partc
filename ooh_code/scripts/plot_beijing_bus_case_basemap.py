"""Publication-style basemap for the Beijing bus case study.

Plots 170 bus-stop meeting points and 240 random home/customer points
with a CARTO Positron light tile basemap and Mercator projection.

Usage:
    python scripts/plot_beijing_bus_case_basemap.py
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image, ImageEnhance
import requests


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT.parent / "related_work" / "bus_stops_and_random_points_Beijingcase.txt"
DEFAULT_OUT = ROOT / "Experiments" / "analysis" / "beijing_bus_case_map"
TILE_CACHE = ROOT / "Experiments" / "analysis" / "_tile_cache"
R_MAJOR = 6378137.0
TILE_SIZE = 256


# ── Mercator / tile helpers (from plot_yanjiao_case_basemap.py) ──────────

def lonlat_to_mercator(lon: np.ndarray, lat: np.ndarray):
    x = R_MAJOR * np.deg2rad(lon)
    lat = np.clip(lat, -85.05112878, 85.05112878)
    y = R_MAJOR * np.log(np.tan(np.pi / 4.0 + np.deg2rad(lat) / 2.0))
    return x, y


def mercator_to_lonlat(x: float, y: float):
    lon = math.degrees(x / R_MAJOR)
    lat = math.degrees(2.0 * math.atan(math.exp(y / R_MAJOR)) - math.pi / 2.0)
    return lon, lat


def lonlat_to_tile(lon: float, lat: float, z: int):
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tile_bounds_mercator(x: int, y: int, z: int):
    n = 2 ** z
    lon_w = x / n * 360.0 - 180.0
    lon_e = (x + 1) / n * 360.0 - 180.0
    lat_n = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_s = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    xw, yn = lonlat_to_mercator(np.array([lon_w]), np.array([lat_n]))
    xe, ys = lonlat_to_mercator(np.array([lon_e]), np.array([lat_s]))
    return float(xw[0]), float(xe[0]), float(ys[0]), float(yn[0])


def expand_bounds_xy(points_xy: np.ndarray, pad_frac: float):
    xmin, ymin = points_xy.min(axis=0)
    xmax, ymax = points_xy.max(axis=0)
    dx = max(xmax - xmin, 1.0)
    dy = max(ymax - ymin, 1.0)
    return xmin - dx * pad_frac, xmax + dx * pad_frac, ymin - dy * pad_frac, ymax + dy * pad_frac


def fetch_tile(z: int, x: int, y: int, timeout: int = 30) -> Image.Image:
    cache_path = TILE_CACHE / str(z) / str(x) / f"{y}.png"
    if cache_path.exists():
        return Image.open(cache_path).convert("RGB")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    subdomain = "abcd"[(x + y) % 4]
    urls = [
        f"https://{subdomain}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        f"https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    ]
    headers = {"User-Agent": "DRT-Beijing-bus-case-map/1.0"}
    for url in urls:
        for attempt in range(3):
            try:
                response = requests.get(url, headers=headers, timeout=timeout)
                response.raise_for_status()
                cache_path.write_bytes(response.content)
                return Image.open(io.BytesIO(response.content)).convert("RGB")
            except Exception:
                if attempt < 2:
                    import time; time.sleep(1)
                continue
    # Fallback: return a blank tile
    return Image.new("RGB", (TILE_SIZE, TILE_SIZE), (240, 240, 240))


def basemap_for_bounds(xlim, ylim, zoom, alpha=0.78):
    lon_w, lat_s = mercator_to_lonlat(xlim[0], ylim[0])
    lon_e, lat_n = mercator_to_lonlat(xlim[1], ylim[1])
    tx0, ty1 = lonlat_to_tile(lon_w, lat_s, zoom)
    tx1, ty0 = lonlat_to_tile(lon_e, lat_n, zoom)
    tx0, tx1 = sorted((tx0, tx1))
    ty0, ty1 = sorted((ty0, ty1))

    mosaic = Image.new("RGB", ((tx1 - tx0 + 1) * TILE_SIZE, (ty1 - ty0 + 1) * TILE_SIZE), "white")
    total = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
    done = 0
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            done += 1
            print(f"\r  Fetching tiles: {done}/{total}", end="", flush=True)
            tile = fetch_tile(zoom, tx, ty)
            mosaic.paste(tile, ((tx - tx0) * TILE_SIZE, (ty - ty0) * TILE_SIZE))
    print()

    mosaic = ImageEnhance.Color(mosaic).enhance(0.55)
    mosaic = ImageEnhance.Contrast(mosaic).enhance(0.92)
    arr = np.asarray(mosaic).astype(float) / 255.0
    arr = arr * alpha + np.ones_like(arr) * (1.0 - alpha)

    left, _, _, top = tile_bounds_mercator(tx0, ty0, zoom)
    _, right, bottom, _ = tile_bounds_mercator(tx1, ty1, zoom)
    return arr, (left, right, bottom, top)


# ── Data reader ───────────────────────────────────────────────────────────

def read_beijing_bus_data(path: Path):
    """Read bus stops + random home points from the Beijing case text file."""
    bus_stops = []
    random_points = []
    section = None

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("NODE_COORD_SECTION"):
                if "Bus Stops" in line:
                    section = "bus"
                elif "Random Points" in line:
                    section = "home"
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            lon, lat = float(parts[1]), float(parts[2])
            if section == "bus":
                bus_stops.append((lon, lat))
            elif section == "home":
                random_points.append((lon, lat))

    return {
        "meeting_points": np.asarray(bus_stops, dtype=float),
        "homes": np.asarray(random_points, dtype=float),
    }


# ── Decorations ───────────────────────────────────────────────────────────

def add_north_arrow(ax, x=0.93, y=0.16):
    ax.annotate(
        "N", xy=(x, y + 0.08), xytext=(x, y),
        xycoords="axes fraction", ha="center", va="center",
        fontsize=10, fontweight="bold", color="#111827",
        arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#111827"),
        zorder=20,
    )


def add_scale_bar_m(ax, length_km, loc=(0.06, 0.08)):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    length = length_km * 1000.0
    x = xmin + (xmax - xmin) * loc[0]
    y = ymin + (ymax - ymin) * loc[1]
    ax.plot([x, x + length], [y, y], color="#111827", lw=2.8, solid_capstyle="butt", zorder=20)
    tick = (ymax - ymin) * 0.012
    ax.plot([x, x], [y - tick, y + tick], color="#111827", lw=1.4, zorder=20)
    ax.plot([x + length, x + length], [y - tick, y + tick], color="#111827", lw=1.4, zorder=20)
    txt = ax.text(x + length / 2, y - (ymax - ymin) * 0.035,
                  f"{length_km:g} km", ha="center", va="top", fontsize=9, color="#111827", zorder=20)
    txt.set_path_effects([pe.withStroke(linewidth=3, foreground="white", alpha=0.85)])


def style_map_axis(ax, title):
    ax.set_title(title, fontsize=12.5, fontweight="bold", loc="left", pad=9, color="#111827")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#4b5563")
        spine.set_linewidth(0.9)


# ── Main plot ─────────────────────────────────────────────────────────────

def plot_map(output: Path, dpi: int = 300):
    coords = read_beijing_bus_data(DATA_FILE)
    homes_ll = coords["homes"]
    mps_ll = coords["meeting_points"]
    print(f"  Home points: {len(homes_ll)}")
    print(f"  Meeting points (bus stops): {len(mps_ll)}")

    hx, hy = lonlat_to_mercator(homes_ll[:, 0], homes_ll[:, 1])
    mx, my = lonlat_to_mercator(mps_ll[:, 0], mps_ll[:, 1])
    homes_xy = np.column_stack([hx, hy])
    mps_xy = np.column_stack([mx, my])
    all_xy = np.vstack([homes_xy, mps_xy])

    bounds = expand_bounds_xy(all_xy, 0.12)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_axes([0.05, 0.06, 0.90, 0.84])

    print("Downloading basemap tiles...")
    bg, bg_extent = basemap_for_bounds((bounds[0], bounds[1]), (bounds[2], bounds[3]), zoom=12)
    ax.imshow(bg, extent=bg_extent, origin="upper", zorder=0)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    style_map_axis(ax, "Beijing bus case study")

    # Density contours for homes
    counts, xedges, yedges = np.histogram2d(homes_xy[:, 0], homes_xy[:, 1], bins=30)
    if counts.max() > 0:
        xs = (xedges[:-1] + xedges[1:]) / 2
        ys = (yedges[:-1] + yedges[1:]) / 2
        levels = np.linspace(max(2, counts.max() * 0.18), counts.max(), 5)
        ax.contourf(xs, ys, counts.T, levels=levels, cmap="Blues", alpha=0.15, zorder=3)
        ax.contour(xs, ys, counts.T, levels=levels, colors="#1d4ed8", linewidths=0.4, alpha=0.3, zorder=4)

    ax.scatter(
        homes_xy[:, 0], homes_xy[:, 1],
        s=14, c="#2563eb", alpha=0.45, linewidths=0, zorder=6,
        label=f"Home locations ({len(homes_xy)})",
    )
    ax.scatter(
        mps_xy[:, 0], mps_xy[:, 1],
        s=42, marker="s", c="#f97316", edgecolors="white", linewidths=0.6,
        alpha=0.90, zorder=8,
        label=f"Bus-stop meeting points ({len(mps_xy)})",
    )

    add_scale_bar_m(ax, 5, (0.055, 0.075))
    add_north_arrow(ax, 0.94, 0.11)

    legend = ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#2563eb",
                   markeredgecolor="none", alpha=0.60, markersize=7,
                   label=f"Home locations ({len(homes_xy)})"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor="#f97316",
                   markeredgecolor="white", markersize=8,
                   label=f"Bus-stop meeting points ({len(mps_xy)})"),
        ],
        loc="lower right", bbox_to_anchor=(0.99, 0.02),
        frameon=True, facecolor="white", edgecolor="#d1d5db",
        framealpha=0.92, fontsize=9,
    )

    fig.text(0.05, 0.955, "Beijing Bus Case — Meeting Points & Home Locations",
             fontsize=14, fontweight="bold", color="#111827")
    fig.text(0.05, 0.928,
             f"170 bus stops + 240 random customer points  |  Lon {homes_ll[:,0].min():.2f}–{mps_ll[:,0].max():.2f}, Lat {homes_ll[:,1].min():.2f}–{mps_ll[:,1].max():.2f}",
             fontsize=9.5, color="#4b5563")
    fig.text(0.95, 0.035, "Basemap: CARTO Positron / OpenStreetMap contributors",
             ha="right", fontsize=7.5, color="#6b7280")

    output.parent.mkdir(parents=True, exist_ok=True)
    png_path = output.with_suffix(".png")
    pdf_path = output.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"\nSaved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    plot_map(DEFAULT_OUT)
