#!/usr/bin/env python
"""Generate normalized commuter-aware Yanjiao instances.

The runtime coordinates mimic the legacy Beijing_bus span so CNN/grid inputs
stay comparable, while routing times and candidate sets are computed from true
local-kilometer geometry. The passenger choice model intentionally stays with
the manuscript's original three alternatives/utilities: outside option, home
pickup, and meeting-point pickup with distance/price effects from the runtime
MNL model. No choice-utility sidecar is written.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_yanjiao_instance import (  # noqa: E402
    DATA_DIR,
    OUT_DIR,
    DEFAULT_K,
    GUOMAO_DESTINATION,
    load_csv,
    write_coords_latlon,
    write_coords_txt,
)
from generate_yanjiao_commuter_instance import (  # noqa: E402
    build_duration_matrix,
    calibrate_with_amap,
    sample_home_latlons,
    select_commuter_meeting_points,
    write_duration_matrix,
    write_mp_features,
)


BUS_X_SPAN = 55.51
BUS_Y_SPAN = 28.82


def depot_metric_points(
    latlons: list[tuple[float, float]],
    depot_latlon: tuple[float, float],
) -> list[tuple[float, float]]:
    """Project lon/lat to local kilometers with Guomao as the origin."""
    depot_lon, depot_lat = depot_latlon
    ref_lat = math.radians(depot_lat)
    points = []
    for lon, lat in latlons:
        x_km = (lon - depot_lon) * 111.320 * math.cos(ref_lat)
        y_km = (lat - depot_lat) * 110.540
        points.append((float(x_km), float(y_km)))
    return points


def normalize_like_beijing_bus(
    metric_points: list[tuple[float, float]],
    x_span: float,
    y_span: float,
) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    """Map true commuter geometry to non-negative Beijing_bus-like coordinates."""
    raw_x = np.array([max(0.0, p[0]) for p in metric_points], dtype=float)
    raw_y = np.array([max(0.0, p[1]) for p in metric_points], dtype=float)
    x_scale = x_span / max(float(raw_x.max()), 1e-9)
    y_scale = y_span / max(float(raw_y.max()), 1e-9)
    coords = [
        (round(float(raw_x[i] * x_scale), 2), round(float(raw_y[i] * y_scale), 2))
        for i in range(len(metric_points))
    ]
    coords[0] = (0.0, 0.0)
    bounds = {
        "coordinate_unit": "beijing_bus_normalized_unit",
        "x_span_target": float(x_span),
        "y_span_target": float(y_span),
        "raw_x_max_km": float(raw_x.max()),
        "raw_y_max_km": float(raw_y.max()),
        "x_scale": float(x_scale),
        "y_scale": float(y_scale),
        "negative_x_clamped": int(sum(1 for x, _ in metric_points if x < 0.0)),
        "negative_y_clamped": int(sum(1 for _, y in metric_points if y < 0.0)),
    }
    return coords, bounds


def route_distance_diagnostics(
    metric_points: list[tuple[float, float]],
    adjacency: np.ndarray,
    passengers: int,
    n_mp: int,
) -> dict[str, Any]:
    homes = np.asarray(metric_points[1:1 + passengers], dtype=float)
    mps = np.asarray(metric_points[1 + passengers:1 + passengers + n_mp], dtype=float)
    chosen = []
    nearest = []
    all_d = []
    for row_idx, home in enumerate(homes, start=1):
        dd = np.sqrt(((mps - home) ** 2).sum(axis=1))
        all_d.extend(dd.tolist())
        nearest.append(float(dd.min()))
        selected = np.where(adjacency[row_idx] == 1)[0]
        chosen.extend(dd[selected].tolist())
    return {
        "all_home_mp_km_mean": float(np.mean(all_d)),
        "nearest_home_mp_km_mean": float(np.mean(nearest)),
        "chosen_home_mp_km_mean": float(np.mean(chosen)),
        "chosen_home_mp_km_p50": float(np.percentile(chosen, 50)),
        "chosen_home_mp_km_p90": float(np.percentile(chosen, 90)),
        "chosen_home_mp_km_max": float(np.max(chosen)),
        "adjacency_unique_rows": int(len({tuple(row.tolist()) for row in adjacency[1:]})),
    }


def build_nearest_adjacency(
    metric_points: list[tuple[float, float]],
    passengers: int,
    n_mp: int,
    k: int,
) -> np.ndarray:
    """Build candidate meeting-point sets from true home-MP distance only."""
    arr = np.asarray(metric_points, dtype=float)
    homes = arr[1:1 + passengers]
    mps = arr[1 + passengers:1 + passengers + n_mp]
    adjacency = np.zeros((1 + passengers, n_mp), dtype=np.int32)
    k_eff = min(int(k), n_mp)
    for i, home in enumerate(homes, start=1):
        dd = np.sqrt(((mps - home) ** 2).sum(axis=1))
        adjacency[i, np.argsort(dd)[:k_eff]] = 1
    return adjacency


def build_walk_distance_matrix(
    metric_points: list[tuple[float, float]],
    passengers: int,
    n_mp: int,
) -> np.ndarray:
    """Square matrix whose home-MP entries are true walking-distance proxies in km."""
    n_total = 1 + passengers + n_mp
    matrix = np.full((n_total, n_total), np.nan, dtype=np.float32)
    arr = np.asarray(metric_points, dtype=float)
    homes = arr[1:1 + passengers]
    mps = arr[1 + passengers:1 + passengers + n_mp]
    for i, home in enumerate(homes, start=1):
        dd = np.sqrt(((mps - home) ** 2).sum(axis=1))
        matrix[i, 1 + passengers:1 + passengers + n_mp] = dd.astype(np.float32)
    return matrix


def write_metadata(path: Path, data: dict[str, Any]) -> None:
    meta = {
        "instance": "Beijing_Yanjiao",
        "scenario": "Yanjiao-Guomao normalized commuter DRT",
        "variant": "commuter_normalized",
        "seed": data["seed"],
        "passengers": data["passengers"],
        "n_meeting_points": data["n_meeting_points"],
        "k_nearest": data["k_nearest"],
        "coordinate_system": "beijing_bus_style_normalized",
        "projection": "normalized_commuter",
        "depot": {
            "name": "Guomao",
            "lon": data["depot_latlon"][0],
            "lat": data["depot_latlon"][1],
        },
        "bounds": data["bounds"],
        "true_metric_geometry": data["metric_diag"],
        "duration_matrix": data["duration_diag"],
        "choice_model": {
            "choice_utility_sidecar": None,
            "walking_distance_sidecar": f"{data['prefix']}_walk_distance.npy",
            "description": (
                "No *_choice_utility.npy is generated. Runtime choice utility is "
                "strictly composed of predicted in-vehicle travel time, home-to-"
                "meeting-point distance, price adjustment, and the outside-option "
                "utility."
            ),
        },
        "amap_calibration": data["amap_calibration"],
        "generation_method": (
            "Normalized Beijing_bus-style coordinates for model input; "
            "true local-km geometry for duration matrix and nearest meeting-point "
            "candidate adjacency; lower MP density and k to increase realistic "
            "choice pressure without adding extra choice-utility covariates."
        ),
        "data_sources": {
            "residential_pois": "data/yanjiao/residential_pois.csv",
            "bus_stops": "data/yanjiao/bus_stops.csv",
        },
    }
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_one(
    args: argparse.Namespace,
    seed: int,
    residential_pois: list[dict[str, str]],
    bus_stops: list[dict[str, str]],
) -> dict[str, Any]:
    rng = np.random.RandomState(seed)
    depot_latlon = (float(GUOMAO_DESTINATION["lon"]), float(GUOMAO_DESTINATION["lat"]))

    homes_latlon = sample_home_latlons(
        residential_pois,
        passengers=args.passengers,
        rng=rng,
        home_edge_share=args.home_edge_share,
        sigma=args.sigma,
        max_perturb=args.max_perturb,
    )
    mp_records = select_commuter_meeting_points(bus_stops, args.mp, rng)
    mps_latlon = [(float(r["lon"]), float(r["lat"])) for r in mp_records]
    all_latlon = [depot_latlon] + homes_latlon + mps_latlon

    metric_coords = depot_metric_points(all_latlon, depot_latlon)
    normalized_coords, bounds = normalize_like_beijing_bus(metric_coords, args.x_span, args.y_span)

    adjacency = build_nearest_adjacency(metric_coords, args.passengers, args.mp, args.k)
    walk_distance_matrix = build_walk_distance_matrix(metric_coords, args.passengers, args.mp)

    calibration = calibrate_with_amap(all_latlon, metric_coords, depot_latlon, args)
    duration_matrix = build_duration_matrix(
        metric_coords,
        scale=float(calibration.get("scale", 1.0)),
        east_west_kmh=args.east_west_kmh,
        north_south_kmh=args.north_south_kmh,
        local_kmh=args.local_kmh,
        fixed_sec=args.fixed_sec,
    )

    service_times = np.zeros(len(all_latlon), dtype=np.int32)
    metric_diag = route_distance_diagnostics(metric_coords, adjacency, args.passengers, args.mp)
    duration_diag = {
        "max_duration_s": int(np.max(duration_matrix)),
        "home_to_depot_mean_s": float(np.mean(duration_matrix[1:1 + args.passengers, 0])),
        "mp_to_depot_mean_s": float(np.mean(duration_matrix[1 + args.passengers:, 0])),
        "home_mp_selected_mean_s": float(np.mean([
            duration_matrix[1 + i, 1 + args.passengers + j]
            for i in range(args.passengers)
            for j in np.where(adjacency[1 + i] == 1)[0]
        ])),
    }

    return {
        "seed": seed,
        "passengers": args.passengers,
        "n_meeting_points": args.mp,
        "k_nearest": args.k,
        "depot_latlon": depot_latlon,
        "bounds": bounds,
        "all_coords_latlon": all_latlon,
        "all_coords_rel": normalized_coords,
        "home_locations_latlon": homes_latlon,
        "home_locations_rel": normalized_coords[1:1 + args.passengers],
        "mp_locations_latlon": mps_latlon,
        "mp_locations_rel": normalized_coords[1 + args.passengers:],
        "metric_coords": metric_coords,
        "adjacency": adjacency,
        "walk_distance_matrix": walk_distance_matrix,
        "duration_matrix": duration_matrix,
        "service_times": service_times,
        "mp_records": mp_records,
        "metric_diag": metric_diag,
        "duration_diag": duration_diag,
        "amap_calibration": calibration,
    }


def write_outputs(prefix: str, data: dict[str, Any], k: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data["prefix"] = prefix
    write_coords_txt(OUT_DIR / f"{prefix}_coords.txt", data)
    write_coords_latlon(OUT_DIR / f"{prefix}_coords_latlon.txt", data)
    write_duration_matrix(OUT_DIR / f"{prefix}_duration_matrix.txt", data["duration_matrix"])
    np.save(OUT_DIR / f"{prefix}_adjacency{k}.npy", data["adjacency"])
    np.save(OUT_DIR / f"{prefix}_walk_distance.npy", data["walk_distance_matrix"])
    for stale_choice_file in (
        OUT_DIR / f"{prefix}_choice_utility.npy",
        OUT_DIR / f"{prefix}_choice_utility.txt",
    ):
        if stale_choice_file.exists():
            stale_choice_file.unlink()
    np.savetxt(OUT_DIR / f"{prefix}_service_times.txt", data["service_times"], fmt="%d")
    write_metadata(OUT_DIR / f"{prefix}_metadata.json", data)
    write_mp_features(OUT_DIR / f"{prefix}_mp_features.csv", data["mp_records"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate normalized commuter-aware Yanjiao data")
    parser.add_argument("--passengers", type=int, default=400)
    parser.add_argument("--mp", type=int, default=80)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--prefix", default="yanjiao_commuter_norm_{passengers}_{seed}")
    parser.add_argument("--home_edge_share", type=float, default=0.35)
    parser.add_argument("--sigma", type=float, default=0.0015)
    parser.add_argument("--max_perturb", type=float, default=0.0040)
    parser.add_argument("--x_span", type=float, default=BUS_X_SPAN)
    parser.add_argument("--y_span", type=float, default=BUS_Y_SPAN)
    parser.add_argument("--east_west_kmh", type=float, default=34.0)
    parser.add_argument("--north_south_kmh", type=float, default=16.0)
    parser.add_argument("--local_kmh", type=float, default=18.0)
    parser.add_argument("--fixed_sec", type=float, default=75.0)
    parser.add_argument("--use_amap", action="store_true")
    parser.add_argument("--amap_key", default=None)
    parser.add_argument("--amap_sample_size", type=int, default=8)
    parser.add_argument("--amap_sleep_sec", type=float, default=0.12)
    parser.add_argument("--amap_cache", default="data/yanjiao/amap_driving_cache.json")
    parser.add_argument("--min_amap_scale", type=float, default=0.55)
    parser.add_argument("--max_amap_scale", type=float, default=1.80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    res_file = DATA_DIR / "residential_pois.csv"
    bus_file = DATA_DIR / "bus_stops.csv"
    if not res_file.exists() or not bus_file.exists():
        raise FileNotFoundError("Expected data/yanjiao/residential_pois.csv and bus_stops.csv")

    residential_pois = load_csv(res_file)
    bus_stops = load_csv(bus_file)
    print(f"[INFO] Residential POIs={len(residential_pois)}, bus stops={len(bus_stops)}")
    print(f"[INFO] Output prefix={args.prefix}, passengers={args.passengers}, mp={args.mp}, k={args.k}")

    for seed in args.seeds:
        prefix = args.prefix.format(passengers=args.passengers, n_passengers=args.passengers, seed=seed)
        data = generate_one(args, seed, residential_pois, bus_stops)
        write_outputs(prefix, data, args.k)
        metric = data["metric_diag"]
        print(f"\n[INFO] Wrote {prefix}")
        print(f"  normalized span={data['bounds']['x_span_target']:.2f} x {data['bounds']['y_span_target']:.2f}")
        print(f"  chosen home-MP km mean/p90/max={metric['chosen_home_mp_km_mean']:.2f}/"
              f"{metric['chosen_home_mp_km_p90']:.2f}/{metric['chosen_home_mp_km_max']:.2f}")
        print(f"  unique adjacency rows={metric['adjacency_unique_rows']}/{args.passengers}")
        print(f"  duration max={data['duration_diag']['max_duration_s']}s")
        print("  choice utility sidecar=disabled")


if __name__ == "__main__":
    main()
