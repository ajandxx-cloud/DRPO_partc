#!/usr/bin/env python
"""Generate a Beijing-BUS-style Yanjiao instance.

The Beijing_bus case uses only normalized coordinates plus an adjacency10 file:
no duration sidecar, no separate choice utility, and no custom service-time file.
This script mirrors that data style for Yanjiao so DSPO/DRPO comparisons are not
confounded by a different instance construction.
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
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_yanjiao_instance import (  # noqa: E402
    DATA_DIR,
    OUT_DIR,
    DEFAULT_K,
    GUOMAO_DESTINATION,
    compute_density_weights,
    edge_ranked_indices,
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


BUS_X_SPAN = 55.51
BUS_Y_SPAN = 28.82
BUS_CUSTOMERS = 240
BUS_MEETING_POINTS = 169


def weighted_without_replacement(
    n_items: int,
    size: int,
    rng: np.random.RandomState,
    weights: np.ndarray | None = None,
    exclude: set[int] | None = None,
) -> list[int]:
    exclude = exclude or set()
    candidates = np.array([i for i in range(n_items) if i not in exclude], dtype=int)
    if size <= 0:
        return []
    if size > len(candidates):
        raise ValueError(f"Cannot sample {size} unique items from {len(candidates)} candidates")
    if weights is None:
        p = None
    else:
        p = weights[candidates].astype(float)
        p = p / p.sum()
    return [int(i) for i in rng.choice(candidates, size=size, replace=False, p=p)]


def sample_bus_style_homes(
    residential_pois: list[dict[str, str]],
    passengers: int,
    rng: np.random.RandomState,
    edge_share: float,
    sigma: float,
    max_perturb: float,
) -> list[tuple[float, float]]:
    weights = compute_density_weights(residential_pois)
    n_edge = int(round(passengers * edge_share))
    n_dense = passengers - n_edge

    dense = weighted_without_replacement(len(residential_pois), n_dense, rng, weights=weights)
    selected = set(dense)
    edge_rank = edge_ranked_indices(residential_pois)
    edge_pool = [int(i) for i in edge_rank[: max(n_edge * 3, int(len(residential_pois) * 0.45))]]
    edge_pool = [i for i in edge_pool if i not in selected]
    if len(edge_pool) < n_edge:
        edge_pool = [int(i) for i in edge_rank if int(i) not in selected]
    edge = [int(i) for i in rng.choice(edge_pool, size=n_edge, replace=False)] if n_edge else []

    indices = dense + edge
    rng.shuffle(indices)
    homes = []
    for idx in indices:
        lon, lat = parse_location(residential_pois[idx]["location"])
        homes.append(perturb_anchor(lon, lat, sigma, max_perturb, rng))
    return homes


def select_bus_style_mps(
    bus_stops: list[dict[str, str]],
    n_mp: int,
    rng: np.random.RandomState,
) -> list[dict[str, Any]]:
    records = build_stop_records(bus_stops)
    if len(records) < n_mp:
        raise ValueError(f"Need {n_mp} unique bus stops, found {len(records)}")

    lons = np.array([r["lon"] for r in records])
    lats = np.array([r["lat"] for r in records])
    center = (float(np.median(lons)), float(np.median(lats)))
    for r in records:
        r["edge_distance_km"] = km_distance_latlon((r["lon"], r["lat"]), center)
        r["selection_noise"] = float(rng.uniform(0.0, 0.02))

    selected: list[dict[str, Any]] = []
    quality_rank = sorted(
        records,
        key=lambda r: (
            float(r["is_direct"]),
            r["quality_score"] + r["selection_noise"],
            r["route_count"],
        ),
        reverse=True,
    )
    coverage_rank = sorted(records, key=lambda r: (r["selection_noise"], r["edge_distance_km"]), reverse=True)
    edge_rank = sorted(records, key=lambda r: (r["edge_distance_km"], r["quality_score"]), reverse=True)
    all_rank = sorted(records, key=lambda r: (r["quality_score"] + 0.10 * r["edge_distance_km"]), reverse=True)

    greedy_add_diverse(selected, quality_rank, int(round(n_mp * 0.40)),
                       min_separations_km=(0.55, 0.35, 0.20, 0.0))
    greedy_add_diverse(selected, coverage_rank, int(round(n_mp * 0.78)),
                       min_separations_km=(0.50, 0.30, 0.15, 0.0))
    greedy_add_diverse(selected, edge_rank, int(round(n_mp * 0.90)),
                       min_separations_km=(0.45, 0.25, 0.10, 0.0))
    greedy_add_diverse(selected, all_rank, n_mp,
                       min_separations_km=(0.30, 0.15, 0.0))

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
    raw_x = np.array([(lon - depot_lon) * 111.320 * math.cos(ref_lat) for lon, _ in all_latlon], dtype=float)
    raw_y = np.array([max(0.0, (lat - depot_lat) * 110.540) for _, lat in all_latlon], dtype=float)

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


def build_adjacency(coords: list[tuple[float, float]], passengers: int, n_mp: int, k: int,
                    far_mix: bool) -> np.ndarray:
    arr = np.array(coords, dtype=float)
    mp = arr[1 + passengers:]
    adjacency = np.zeros((1 + passengers, n_mp), dtype=np.int32)
    k_eff = min(int(k), n_mp)
    target_near = max(1, int(round(k_eff * (0.35 if far_mix else 0.45))))
    for customer_idx in range(1, 1 + passengers):
        dd = np.sqrt(((mp - arr[customer_idx]) ** 2).sum(axis=1))
        ranked = np.argsort(dd)
        selected = [int(x) for x in ranked[:target_near]]
        # Beijing_bus candidate sets are not purely nearest-neighbor tight.
        # Add medium-range options to create route/choice consequences similar
        # to the original BUS case while keeping every option spatially plausible.
        if far_mix:
            bands = [
                ranked[target_near:min(len(ranked), 35)],
                ranked[35:min(len(ranked), 85)],
                ranked[85:min(len(ranked), 140)],
            ]
            quotas = [2, 2, k_eff - target_near - 4]
        else:
            bands = [
                ranked[target_near:min(len(ranked), 25)],
                ranked[25:min(len(ranked), 60)],
                ranked[60:min(len(ranked), 100)],
            ]
            quotas = [2, 2, k_eff - target_near - 4]
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


def diagnostics(coords: list[tuple[float, float]], adjacency: np.ndarray, passengers: int, n_mp: int) -> dict[str, Any]:
    arr = np.array(coords, dtype=float)
    homes = arr[1:1 + passengers]
    mps = arr[1 + passengers:]
    all_d = []
    chosen_d = []
    for i, home in enumerate(homes, start=1):
        dd = np.sqrt(((mps - home) ** 2).sum(axis=1))
        all_d.extend(dd.tolist())
        chosen_d.extend(dd[np.where(adjacency[i] == 1)[0]].tolist())
    row_set = {tuple(row.tolist()) for row in adjacency[1:]}
    return {
        "bbox_x_span": float(np.max(arr[:, 0]) - np.min(arr[:, 0])),
        "bbox_y_span": float(np.max(arr[:, 1]) - np.min(arr[:, 1])),
        "bbox_ratio": float((np.max(arr[:, 0]) - np.min(arr[:, 0])) /
                            max(np.max(arr[:, 1]) - np.min(arr[:, 1]), 1e-9)),
        "all_customer_mp_distance_mean": float(np.mean(all_d)),
        "all_customer_mp_distance_p50": float(np.percentile(all_d, 50)),
        "all_customer_mp_distance_p90": float(np.percentile(all_d, 90)),
        "chosen_distance_mean": float(np.mean(chosen_d)),
        "chosen_distance_p50": float(np.percentile(chosen_d, 50)),
        "chosen_distance_p90": float(np.percentile(chosen_d, 90)),
        "chosen_distance_max": float(np.max(chosen_d)),
        "adjacency_unique_rows": int(len(row_set)),
        "adjacency_rows": int(passengers),
    }


def write_metadata(path: Path, data: dict[str, Any]) -> None:
    meta = {
        "instance": "Beijing_Yanjiao",
        "scenario": "Yanjiao bus-style normalized case",
        "variant": "bus_style",
        "seed": data["seed"],
        "passengers": data["passengers"],
        "n_meeting_points": data["n_meeting_points"],
        "k_nearest": data["k_nearest"],
        "coordinate_system": "beijing_bus_style_normalized",
        "projection": "bus_style",
        "depot": {
            "name": "Guomao",
            "lon": data["depot_latlon"][0],
            "lat": data["depot_latlon"][1],
        },
        "bounds": data["bounds"],
        "diagnostics": data["diagnostics"],
        "adjacency_mode": "near_medium_far_mix" if data.get("far_mix_adjacency") else "near_medium_mix",
        "generation_method": (
            "Mimics Beijing_bus data style: 240 customers, 169 OOH points, "
            "normalized 55.51x28.82 coordinate span, BUS-style adjacency, "
            "no duration sidecar, no choice-utility sidecar, no service-time sidecar."
        ),
        "data_sources": {
            "residential_pois": "data/yanjiao/residential_pois.csv",
            "bus_stops": "data/yanjiao/bus_stops.csv",
        },
    }
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def write_mp_features(path: Path, records: list[dict[str, Any]]) -> None:
    fields = ["mp_index", "lon", "lat", "route_count", "direct_count", "quality_score", "name", "address"]
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


def generate_one(args: argparse.Namespace, seed: int,
                 residential_pois: list[dict[str, str]],
                 bus_stops: list[dict[str, str]]) -> dict[str, Any]:
    rng = np.random.RandomState(seed)
    depot_latlon = (float(GUOMAO_DESTINATION["lon"]), float(GUOMAO_DESTINATION["lat"]))
    homes_latlon = sample_bus_style_homes(
        residential_pois,
        passengers=args.passengers,
        rng=rng,
        edge_share=args.home_edge_share,
        sigma=args.sigma,
        max_perturb=args.max_perturb,
    )
    mp_records = select_bus_style_mps(bus_stops, args.mp, rng)
    mps_latlon = [(float(r["lon"]), float(r["lat"])) for r in mp_records]
    all_latlon = [depot_latlon] + homes_latlon + mps_latlon
    coords, bounds = project_bus_style(all_latlon, depot_latlon)
    adjacency = build_adjacency(coords, args.passengers, args.mp, args.k, args.far_mix_adjacency)
    diag = diagnostics(coords, adjacency, args.passengers, args.mp)
    return {
        "seed": seed,
        "passengers": args.passengers,
        "n_meeting_points": args.mp,
        "k_nearest": args.k,
        "depot_latlon": depot_latlon,
        "bounds": bounds,
        "all_coords_latlon": all_latlon,
        "all_coords_rel": coords,
        "home_locations_latlon": homes_latlon,
        "home_locations_rel": coords[1:1 + args.passengers],
        "mp_locations_latlon": mps_latlon,
        "mp_locations_rel": coords[1 + args.passengers:],
        "adjacency": adjacency,
        "diagnostics": diag,
        "mp_records": mp_records,
        "far_mix_adjacency": bool(args.far_mix_adjacency),
    }


def write_outputs(prefix: str, data: dict[str, Any], k: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_coords_txt(OUT_DIR / f"{prefix}_coords.txt", data)
    write_coords_latlon(OUT_DIR / f"{prefix}_coords_latlon.txt", data)
    np.save(OUT_DIR / f"{prefix}_adjacency{k}.npy", data["adjacency"])
    write_metadata(OUT_DIR / f"{prefix}_metadata.json", data)
    write_mp_features(OUT_DIR / f"{prefix}_mp_features.csv", data["mp_records"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Beijing-BUS-style Yanjiao data")
    parser.add_argument("--passengers", type=int, default=BUS_CUSTOMERS)
    parser.add_argument("--mp", type=int, default=BUS_MEETING_POINTS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--prefix", default="yanjiao_bus_style_{passengers}_{seed}")
    parser.add_argument("--home_edge_share", type=float, default=0.35)
    parser.add_argument("--sigma", type=float, default=0.0018)
    parser.add_argument("--max_perturb", type=float, default=0.0040)
    parser.add_argument("--far_mix_adjacency", action="store_true",
                        help="Use a wider candidate-distance mix, closer to Beijing_bus route contrast.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    residential_pois = load_csv(DATA_DIR / "residential_pois.csv")
    bus_stops = load_csv(DATA_DIR / "bus_stops.csv")
    print(f"[INFO] Residential POIs={len(residential_pois)}, bus stops={len(bus_stops)}")
    for seed in args.seeds:
        prefix = args.prefix.format(passengers=args.passengers, n_passengers=args.passengers, seed=seed)
        data = generate_one(args, seed, residential_pois, bus_stops)
        write_outputs(prefix, data, args.k)
        d = data["diagnostics"]
        print(f"\n[INFO] Wrote {prefix}")
        print(f"  nodes={1 + args.passengers + args.mp}, passengers={args.passengers}, mp={args.mp}")
        print(f"  bbox span={d['bbox_x_span']:.2f} x {d['bbox_y_span']:.2f}, ratio={d['bbox_ratio']:.2f}")
        print(f"  chosen distance mean/p50/p90={d['chosen_distance_mean']:.2f}/"
              f"{d['chosen_distance_p50']:.2f}/{d['chosen_distance_p90']:.2f}")
        print(f"  unique adjacency rows={d['adjacency_unique_rows']}/{d['adjacency_rows']}")


if __name__ == "__main__":
    main()
