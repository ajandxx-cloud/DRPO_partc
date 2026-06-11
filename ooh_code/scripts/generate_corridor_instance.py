#!/usr/bin/env python
"""Generate a corridor-style Yanjiao instance that mirrors Beijing_bus structure.

Beijing_bus works because homes and MPs are spread across a wide 2D area (~55x29),
creating high routing-cost heterogeneity. The original Yanjiao case fails because
all 400 homes and 100 MPs cluster in one suburb, making route costs near-identical.

This script:
1. Spreads 240 homes along the Guomao→Tongzhou→Yanjiao corridor using residential POIs
2. Selects ~170 meeting points (bus stops) spread across the same corridor
3. Projects to Beijing_bus-style normalized coordinates (55.51 x 28.82)
4. Outputs Beijing_bus-compatible files (coords.txt, adjacency10.npy, etc.)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from generate_yanjiao_instance import (  # noqa: E402
    load_csv,
    parse_location,
    perturb_anchor,
    write_coords_latlon,
    write_coords_txt,
)
from generate_yanjiao_commuter_instance import (  # noqa: E402
    build_stop_records,
    greedy_add_diverse,
    km_distance_latlon,
)

# --- Beijing_bus target dimensions ---
BUS_X_SPAN = 55.51
BUS_Y_SPAN = 28.82
BUS_CUSTOMERS = 240
BUS_MEETING_POINTS = 100
DEFAULT_K = 10

# --- Depot ---
GUOMAO_DESTINATION = {"name": "Guomao", "lon": 116.4610, "lat": 39.9087}

# --- Data ---
DATA_DIR = ROOT / "data" / "corridor"
OUT_DIR = ROOT / "Environments" / "OOH" / "Beijing_Yanjiao"

# --- Corridor zone boundaries (by longitude) ---
ZONE_CBD = (116.46, 116.60)     # CBD outskirts → near depot
ZONE_TONGZHOU = (116.60, 116.78)  # Tongzhou mid-corridor
ZONE_YANJIAO = (116.78, 117.06)  # Yanjiao suburb


def classify_zone(lon: float) -> str:
    if lon < ZONE_TONGZHOU[0]:
        return "cbd"
    elif lon < ZONE_YANJIAO[0]:
        return "tongzhou"
    return "yanjiao"


def sample_homes_uniform(
    residential_pois: list[dict],
    passengers: int,
    rng: np.random.RandomState,
    sigma: float = 0.0018,
    max_perturb: float = 0.0040,
    n_lon_bins: int = 8,
    n_lat_bins: int = 4,
) -> list[tuple[float, float]]:
    """Uniform grid-based home sampling: divide the corridor into a grid and
    sample equal numbers of homes from each cell.  This avoids clustering that
    occurs with zone-based proportional allocation."""
    lons = np.array([parse_location(p["location"])[0] for p in residential_pois])
    lats = np.array([parse_location(p["location"])[1] for p in residential_pois])

    lon_edges = np.linspace(float(lons.min()) - 1e-6, float(lons.max()) + 1e-6, n_lon_bins + 1)
    lat_edges = np.linspace(float(lats.min()) - 1e-6, float(lats.max()) + 1e-6, n_lat_bins + 1)

    # Assign each POI to a grid cell
    cells: dict[tuple[int, int], list[int]] = {}
    for i, p in enumerate(residential_pois):
        lon, _ = parse_location(p["location"])
        ci = int(np.searchsorted(lon_edges, lon, side="right")) - 1
        ri = int(np.searchsorted(lat_edges, _, side="right")) - 1
        ci = max(0, min(ci, n_lon_bins - 1))
        ri = max(0, min(ri, n_lat_bins - 1))
        cells.setdefault((ci, ri), []).append(i)

    non_empty = [k for k, v in cells.items() if v]
    n_cells = len(non_empty)
    base_per_cell = passengers // n_cells
    remainder = passengers - base_per_cell * n_cells
    rng.shuffle(non_empty)

    quota: dict[tuple[int, int], int] = {}
    for i, key in enumerate(non_empty):
        quota[key] = base_per_cell + (1 if i < remainder else 0)

    homes: list[tuple[float, float]] = []
    for key in non_empty:
        pool = cells[key]
        n_pick = quota[key]
        # Always allow replacement to hit the quota
        idx = rng.choice(pool, size=n_pick, replace=True)
        for j in idx:
            lon, lat = parse_location(residential_pois[int(j)]["location"])
            plon, plat = perturb_anchor(lon, lat, sigma, max_perturb, rng)
            homes.append((plon, plat))

    rng.shuffle(homes)
    return homes[:passengers]


def select_mps_global_diverse(
    bus_stops: list[dict],
    n_mp: int,
    rng: np.random.RandomState,
) -> list[dict[str, Any]]:
    """Global quality+diversity MP selection across the entire corridor.
    Uses larger min-separation to increase walking distances."""
    records = build_stop_records(bus_stops)
    if len(records) < n_mp:
        n_mp = len(records)

    lons = np.array([r["lon"] for r in records])
    lats = np.array([r["lat"] for r in records])
    center = (float(np.median(lons)), float(np.median(lats)))
    for r in records:
        r["edge_distance_km"] = km_distance_latlon((r["lon"], r["lat"]), center)
        r["selection_noise"] = float(rng.uniform(0.0, 0.02))

    quality_rank = sorted(
        records,
        key=lambda r: (r["quality_score"] + r["selection_noise"], r["route_count"]),
        reverse=True,
    )
    coverage_rank = sorted(
        records,
        key=lambda r: (r["selection_noise"], r["edge_distance_km"]),
        reverse=True,
    )

    selected: list[dict[str, Any]] = []
    # Phase 1: quality-ranked with 1.0 km min separation
    greedy_add_diverse(selected, quality_rank, int(round(n_mp * 0.55)),
                       min_separations_km=(1.0, 0.7, 0.4, 0.0))
    # Phase 2: coverage-ranked with relaxed separation
    greedy_add_diverse(selected, coverage_rank, int(round(n_mp * 0.80)),
                       min_separations_km=(0.8, 0.5, 0.3, 0.0))
    # Phase 3: fill with quality-ranked, lower separation
    greedy_add_diverse(selected, quality_rank, n_mp,
                       min_separations_km=(0.5, 0.3, 0.0))

    if len(selected) < n_mp:
        selected_locs = {s["raw"]["location"] for s in selected}
        remaining = [r for r in records if r["raw"]["location"] not in selected_locs]
        rng.shuffle(remaining)
        selected.extend(remaining[: n_mp - len(selected)])

    return selected[:n_mp]


def project_bus_style(
    all_latlon: list[tuple[float, float]],
    depot_latlon: tuple[float, float],
) -> tuple[list[tuple[float, float]], dict[str, float]]:
    depot_lon, depot_lat = depot_latlon
    ref_lat = math.radians(depot_lat)
    raw_x = np.array([(lon - depot_lon) * 111320 * math.cos(ref_lat) for lon, _ in all_latlon], dtype=float)
    raw_y = np.array([max(0.0, (lat - depot_lat) * 110540) for _, lat in all_latlon], dtype=float)

    x_scale = BUS_X_SPAN / max(float(np.max(raw_x)), 1e-9)
    y_scale = BUS_Y_SPAN / max(float(np.max(raw_y)), 1e-9)
    coords = [(round(max(0.0, raw_x[i]) * x_scale, 2),
               round(raw_y[i] * y_scale, 2)) for i in range(len(all_latlon))]
    coords[0] = (0.0, 0.0)
    bounds = {
        "coordinate_unit": "beijing_bus_normalized_unit",
        "x_span_target": BUS_X_SPAN,
        "y_span_target": BUS_Y_SPAN,
        "raw_x_max_km": float(np.max(raw_x)),
        "raw_y_max_km": float(np.max(raw_y)),
        "x_scale": float(x_scale),
        "y_scale": float(y_scale),
    }
    return coords, bounds


def build_adjacency(
    coords: list[tuple[float, float]], passengers: int, n_mp: int, k: int,
) -> np.ndarray:
    arr = np.array(coords, dtype=float)
    mp = arr[1 + passengers:]
    adjacency = np.zeros((1 + passengers, n_mp), dtype=np.int32)
    k_eff = min(int(k), n_mp)
    target_near = max(1, int(round(k_eff * 0.35)))

    for customer_idx in range(1, 1 + passengers):
        dd = np.sqrt(((mp - arr[customer_idx]) ** 2).sum(axis=1))
        ranked = np.argsort(dd)
        selected = [int(x) for x in ranked[:target_near]]

        # Add medium and far-range candidates for route-cost diversity
        bands = [
            ranked[target_near:min(len(ranked), 35)],
            ranked[35:min(len(ranked), 85)],
            ranked[85:min(len(ranked), 140)],
        ]
        quotas = [2, 2, max(0, k_eff - target_near - 4)]
        for band, quota in zip(bands, quotas):
            for idx in band[:max(0, quota)]:
                jj = int(idx)
                if jj not in selected:
                    selected.append(jj)
                if len(selected) >= k_eff:
                    break
            if len(selected) >= k_eff:
                break
        for idx in ranked:
            if len(selected) >= k_eff:
                break
            jj = int(idx)
            if jj not in selected:
                selected.append(jj)
        adjacency[customer_idx, selected[:k_eff]] = 1
    return adjacency


def diagnostics(
    coords: list[tuple[float, float]], adjacency: np.ndarray,
    passengers: int, n_mp: int, mp_records: list[dict],
) -> dict[str, Any]:
    arr = np.array(coords, dtype=float)
    homes = arr[1:1 + passengers]
    mps = arr[1 + passengers:]
    all_d, chosen_d = [], []
    for i, home in enumerate(homes, start=1):
        dd = np.sqrt(((mps - home) ** 2).sum(axis=1))
        all_d.extend(dd.tolist())
        chosen_d.extend(dd[np.where(adjacency[i] == 1)[0]].tolist())
    row_set = {tuple(row.tolist()) for row in adjacency[1:]}

    # Zone distribution
    mp_zones = {"cbd": 0, "tongzhou": 0, "yanjiao": 0}
    home_zones = {"cbd": 0, "tongzhou": 0, "yanjiao": 0}
    for r in mp_records:
        mp_zones[classify_zone(r["lon"])] += 1
    latlon_start = 1 + passengers  # index into all_latlon
    # Homes zone counts from actual latlon
    return {
        "bbox_x_span": float(np.max(arr[:, 0]) - np.min(arr[:, 0])),
        "bbox_y_span": float(np.max(arr[:, 1]) - np.min(arr[:, 1])),
        "bbox_ratio": float((np.max(arr[:, 0]) - np.min(arr[:, 0])) /
                            max(np.max(arr[:, 1]) - np.min(arr[:, 1]), 1e-9)),
        "all_customer_mp_distance_mean": float(np.mean(all_d)),
        "chosen_distance_mean": float(np.mean(chosen_d)),
        "chosen_distance_p50": float(np.percentile(chosen_d, 50)),
        "chosen_distance_p90": float(np.percentile(chosen_d, 90)),
        "adjacency_unique_rows": int(len(row_set)),
        "adjacency_rows": int(passengers),
        "mp_zones": mp_zones,
    }


def write_metadata(path: Path, data: dict[str, Any]) -> None:
    meta = {
        "instance": "Beijing_Yanjiao_Corridor",
        "scenario": "Corridor-spread Yanjiao-Guomao commuter DRT (Beijing_bus style)",
        "variant": "corridor",
        "seed": data["seed"],
        "passengers": data["passengers"],
        "n_meeting_points": data["n_meeting_points"],
        "k_nearest": data["k_nearest"],
        "coordinate_system": "beijing_bus_style_normalized",
        "depot": {
            "name": "Guomao",
            "lon": data["depot_latlon"][0],
            "lat": data["depot_latlon"][1],
        },
        "bounds": data["bounds"],
        "diagnostics": data["diagnostics"],
        "generation_method": (
            "Corridor2-spread instance: 240 homes uniformly sampled via grid "
            "from residential POIs along the Guomao-Tongzhou-Yanjiao corridor, "
            "~100 meeting points selected globally from bus stops using quality-ranked "
            "greedy diversity with 1.0 km min separation. Beijing_bus-style "
            "55.51x28.82 normalized projection."
        ),
        "data_sources": {
            "residential_pois": "data/corridor/corridor_residential_pois.csv",
            "bus_stops": "data/corridor/corridor_bus_stops.csv",
        },
    }
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def write_mp_features(path: Path, records: list[dict[str, Any]]) -> None:
    fields = ["mp_index", "lon", "lat", "route_count", "direct_count",
              "quality_score", "name", "address"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, record in enumerate(records):
            raw = record["raw"]
            writer.writerow({
                "mp_index": i,
                "lon": f"{record['lon']:.6f}",
                "lat": f"{record['lat']:.6f}",
                "route_count": record["route_count"],
                "direct_count": record["direct_count"],
                "quality_score": f"{record['quality_score']:.4f}",
                "name": raw.get("name", ""),
                "address": raw.get("address", ""),
            })


def generate_one(
    args: argparse.Namespace, seed: int,
    residential_pois: list[dict[str, str]],
    bus_stops: list[dict[str, str]],
) -> dict[str, Any]:
    rng = np.random.RandomState(seed)
    depot_latlon = (float(GUOMAO_DESTINATION["lon"]), float(GUOMAO_DESTINATION["lat"]))

    homes_latlon = sample_homes_uniform(
        residential_pois, passengers=args.passengers, rng=rng,
        sigma=args.sigma, max_perturb=args.max_perturb,
    )
    mp_records = select_mps_global_diverse(bus_stops, args.mp, rng)
    mps_latlon = [(float(r["lon"]), float(r["lat"])) for r in mp_records]

    all_latlon = [depot_latlon] + homes_latlon + mps_latlon
    coords, bounds = project_bus_style(all_latlon, depot_latlon)
    adjacency = build_adjacency(coords, args.passengers, args.mp, args.k)
    diag = diagnostics(coords, adjacency, args.passengers, args.mp, mp_records)

    return {
        "seed": seed,
        "passengers": args.passengers,
        "n_meeting_points": args.mp,
        "k_nearest": args.k,
        "depot_latlon": depot_latlon,
        "bounds": bounds,
        "all_coords_latlon": all_latlon,
        "all_coords_rel": coords,
        "adjacency": adjacency,
        "diagnostics": diag,
        "mp_records": mp_records,
    }


def write_outputs(prefix: str, data: dict[str, Any], k: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_coords_txt(OUT_DIR / f"{prefix}_coords.txt", data)
    write_coords_latlon(OUT_DIR / f"{prefix}_coords_latlon.txt", data)
    np.save(OUT_DIR / f"{prefix}_adjacency{k}.npy", data["adjacency"])
    write_metadata(OUT_DIR / f"{prefix}_metadata.json", data)
    write_mp_features(OUT_DIR / f"{prefix}_mp_features.csv", data["mp_records"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate corridor-spread Yanjiao instance")
    parser.add_argument("--passengers", type=int, default=BUS_CUSTOMERS)
    parser.add_argument("--mp", type=int, default=BUS_MEETING_POINTS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--prefix", default="corridor2_{passengers}_{seed}")
    parser.add_argument("--sigma", type=float, default=0.0018)
    parser.add_argument("--max_perturb", type=float, default=0.0040)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    residential_pois = load_csv(DATA_DIR / "corridor_residential_pois.csv")
    bus_stops = load_csv(DATA_DIR / "corridor_bus_stops.csv")
    print(f"[INFO] Residential POIs={len(residential_pois)}, bus stops={len(bus_stops)}")

    # Quick zone summary
    for name, zone in [("CBD", ZONE_CBD), ("Tongzhou", ZONE_TONGZHOU), ("Yanjiao", ZONE_YANJIAO)]:
        n_res = sum(1 for p in residential_pois if zone[0] <= parse_location(p["location"])[0] < zone[1])
        n_bus = sum(1 for p in bus_stops if zone[0] <= parse_location(p["location"])[0] < zone[1])
        print(f"  {name}: {n_res} residential POIs, {n_bus} bus stops")

    for seed in args.seeds:
        prefix = args.prefix.format(passengers=args.passengers, seed=seed)
        data = generate_one(args, seed, residential_pois, bus_stops)
        write_outputs(prefix, data, args.k)
        d = data["diagnostics"]
        print(f"\n[INFO] Wrote {prefix}")
        print(f"  nodes={1 + args.passengers + args.mp}, passengers={args.passengers}, mp={args.mp}")
        print(f"  bbox span={d['bbox_x_span']:.2f} x {d['bbox_y_span']:.2f}, ratio={d['bbox_ratio']:.2f}")
        print(f"  chosen distance mean/p50/p90={d['chosen_distance_mean']:.2f}/"
              f"{d['chosen_distance_p50']:.2f}/{d['chosen_distance_p90']:.2f}")
        print(f"  unique adjacency rows={d['adjacency_unique_rows']}/{d['adjacency_rows']}")
        print(f"  MP zones: {d['mp_zones']}")


if __name__ == "__main__":
    main()
