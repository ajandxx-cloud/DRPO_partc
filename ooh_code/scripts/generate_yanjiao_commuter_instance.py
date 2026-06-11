#!/usr/bin/env python
"""Generate a commuter-aware Beijing_Yanjiao instance.

This generator keeps the metric projection from the previous Yanjiao data but
separates two concepts that were previously bundled into one Euclidean matrix:

1. vehicle routing time, stored as *_duration_matrix.txt;
2. customer stop attractiveness, stored as *_choice_utility.npy.

Only a small number of Amap calls are used for calibration. The full matrix is
then produced by a deterministic corridor-time model, so the script stays cheap
and reproducible.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
import urllib.parse
import urllib.request
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
    GUOMAO_DESTINATION,
    DEFAULT_K,
    compute_density_weights,
    load_csv,
    make_projection_bounds,
    parse_location,
    perturb_anchor,
    project_latlon_points,
    sample_home_anchors,
    write_coords_latlon,
    write_coords_txt,
)


DIRECT_ROUTE_TOKENS = {
    "811", "813", "814", "816", "817", "818", "819", "882", "930"
}
SUSPENDED_MARKERS = (
    "\u505c\u8fd0",      # normal Chinese "stopped service"
    "\u6682\u505c",
    "\u934b\u6ec6\u7e4d",  # mojibake seen in the existing CSV
)
ROUTE_RE = re.compile(r"(?<!\d)(\d{2,4}[A-Z]?)(?!\d)")


def default_amap_key() -> str | None:
    for key_name in ("AMAP_API_KEY", "GAODE_API_KEY"):
        value = os.environ.get(key_name)
        if value:
            return value
    try:
        from fetch_yanjiao_full_data import API_KEY  # type: ignore

        return API_KEY
    except Exception:
        return None


def unique_by_location(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    out: list[dict[str, str]] = []
    for row in rows:
        loc = row.get("location", "")
        if not loc or loc in seen:
            continue
        seen.add(loc)
        out.append(row)
    return out


def km_distance_latlon(a: tuple[float, float], b: tuple[float, float]) -> float:
    lon1, lat1 = a
    lon2, lat2 = b
    ref_lat = math.radians((lat1 + lat2) / 2.0)
    dx = (lon1 - lon2) * 111.320 * math.cos(ref_lat)
    dy = (lat1 - lat2) * 110.540
    return math.sqrt(dx * dx + dy * dy)


def route_tokens(stop: dict[str, str]) -> list[str]:
    text = (stop.get("name", "") + " " + stop.get("address", "")).upper()
    return sorted(set(ROUTE_RE.findall(text)))


def has_suspended_marker(stop: dict[str, str]) -> bool:
    text = stop.get("name", "") + " " + stop.get("address", "")
    return any(marker in text for marker in SUSPENDED_MARKERS)


def build_stop_records(bus_stops: list[dict[str, str]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, stop in enumerate(unique_by_location(bus_stops)):
        lon, lat = parse_location(stop["location"])
        tokens = route_tokens(stop)
        token_roots = {re.sub(r"[A-Z]$", "", t) for t in tokens}
        direct_count = len(token_roots.intersection(DIRECT_ROUTE_TOKENS))
        route_count = len(tokens)
        suspended = has_suspended_marker(stop)
        quality = 1.0 + 0.16 * min(route_count, 8) + 0.75 * min(direct_count, 2)
        if "930" in token_roots or "818" in token_roots:
            quality += 0.35
        if suspended:
            quality -= 0.65
        quality = float(np.clip(quality, 0.20, 3.50))
        if direct_count > 0:
            headway_min = 6.0
            transfer_penalty_min = 0.0
        elif route_count >= 3:
            headway_min = 10.0
            transfer_penalty_min = 3.0
        else:
            headway_min = 15.0
            transfer_penalty_min = 6.0
        if suspended:
            headway_min += 6.0
            transfer_penalty_min += 4.0
        records.append({
            "source_index": idx,
            "raw": stop,
            "lon": lon,
            "lat": lat,
            "tokens": tokens,
            "route_count": route_count,
            "direct_count": direct_count,
            "is_direct": direct_count > 0,
            "is_suspended": suspended,
            "quality_score": quality,
            "headway_min": headway_min,
            "transfer_penalty_min": transfer_penalty_min,
        })
    return records


def greedy_add_diverse(
    selected: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    target_count: int,
    min_separations_km: tuple[float, ...] = (0.70, 0.45, 0.25, 0.0),
) -> None:
    selected_locs = {c["raw"]["location"] for c in selected}
    for min_sep in min_separations_km:
        if len(selected) >= target_count:
            return
        for cand in candidates:
            if len(selected) >= target_count:
                return
            loc = cand["raw"]["location"]
            if loc in selected_locs:
                continue
            if selected:
                nearest = min(
                    km_distance_latlon((cand["lon"], cand["lat"]), (s["lon"], s["lat"]))
                    for s in selected
                )
                if nearest < min_sep:
                    continue
            selected.append(cand)
            selected_locs.add(loc)


def select_commuter_meeting_points(
    bus_stops: list[dict[str, str]],
    n_mp: int,
    rng: np.random.RandomState,
) -> list[dict[str, Any]]:
    records = build_stop_records(bus_stops)
    if len(records) < n_mp:
        raise ValueError(f"Need at least {n_mp} unique bus stops, found {len(records)}")

    lons = np.array([r["lon"] for r in records])
    lats = np.array([r["lat"] for r in records])
    center = (float(np.median(lons)), float(np.median(lats)))
    for r in records:
        r["edge_distance_km"] = km_distance_latlon((r["lon"], r["lat"]), center)
        r["selection_noise"] = float(rng.uniform(0.0, 0.02))

    selected: list[dict[str, Any]] = []
    high_quality = sorted(
        [r for r in records if r["is_direct"] or r["route_count"] >= 3],
        key=lambda r: (r["quality_score"] + r["selection_noise"], r["route_count"]),
        reverse=True,
    )
    edge = sorted(records, key=lambda r: (r["edge_distance_km"], r["quality_score"]), reverse=True)
    all_ranked = sorted(
        records,
        key=lambda r: (r["quality_score"] + 0.12 * r["edge_distance_km"] + r["selection_noise"]),
        reverse=True,
    )

    coverage = sorted(records, key=lambda r: (r["selection_noise"], r["quality_score"]), reverse=True)

    greedy_add_diverse(selected, high_quality, int(round(n_mp * 0.50)))
    greedy_add_diverse(selected, coverage, int(round(n_mp * 0.82)))
    greedy_add_diverse(selected, edge, int(round(n_mp * 0.90)))
    greedy_add_diverse(selected, all_ranked, n_mp)
    if len(selected) < n_mp:
        remaining = [r for r in records if r["raw"]["location"] not in {s["raw"]["location"] for s in selected}]
        rng.shuffle(remaining)
        selected.extend(remaining[: n_mp - len(selected)])
    return selected[:n_mp]


def sample_home_latlons(
    residential_pois: list[dict[str, str]],
    passengers: int,
    rng: np.random.RandomState,
    home_edge_share: float,
    sigma: float,
    max_perturb: float,
) -> list[tuple[float, float]]:
    weights = compute_density_weights(residential_pois)
    anchors = sample_home_anchors(residential_pois, weights, passengers, rng, edge_share=home_edge_share)
    homes = []
    for anchor in anchors:
        lon, lat = parse_location(anchor["location"])
        homes.append(perturb_anchor(lon, lat, sigma, max_perturb, rng))
    return homes


def road_time_seconds(
    a: tuple[float, float],
    b: tuple[float, float],
    east_west_kmh: float,
    north_south_kmh: float,
    local_kmh: float,
    fixed_sec: float,
) -> float:
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    euclid = math.sqrt(dx * dx + dy * dy)
    if euclid < 1e-9:
        return 0.0
    if dx < 2.0 and dy < 1.2:
        return fixed_sec + 3600.0 * euclid / max(local_kmh, 1e-6)
    return fixed_sec + 3600.0 * ((0.88 * dx) / max(east_west_kmh, 1e-6) +
                                 (1.12 * dy) / max(north_south_kmh, 1e-6))


def build_duration_matrix(
    coords: list[tuple[float, float]],
    scale: float,
    east_west_kmh: float,
    north_south_kmh: float,
    local_kmh: float,
    fixed_sec: float,
) -> np.ndarray:
    n = len(coords)
    matrix = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i + 1, n):
            sec = road_time_seconds(coords[i], coords[j], east_west_kmh, north_south_kmh, local_kmh, fixed_sec)
            val = int(max(1, round(sec * scale)))
            matrix[i, j] = val
            matrix[j, i] = val
    return matrix


def load_cache(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def amap_driving_duration(
    origin: tuple[float, float],
    destination: tuple[float, float],
    api_key: str,
    cache: dict[str, Any],
    sleep_sec: float,
    timeout_sec: float = 15.0,
) -> dict[str, Any]:
    cache_key = f"{origin[0]:.6f},{origin[1]:.6f}|{destination[0]:.6f},{destination[1]:.6f}"
    if cache_key in cache:
        return cache[cache_key]

    params = {
        "key": api_key,
        "origin": f"{origin[0]:.6f},{origin[1]:.6f}",
        "destination": f"{destination[0]:.6f},{destination[1]:.6f}",
        "extensions": "base",
        "strategy": "0",
    }
    url = "https://restapi.amap.com/v3/direction/driving?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=timeout_sec) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if payload.get("status") != "1":
        raise RuntimeError(f"Amap driving API failed: {payload}")
    path0 = payload["route"]["paths"][0]
    result = {
        "distance_m": int(float(path0["distance"])),
        "duration_s": int(float(path0["duration"])),
        "api_status": payload.get("info", "OK"),
    }
    cache[cache_key] = result
    if sleep_sec > 0:
        time.sleep(sleep_sec)
    return result


def calibrate_with_amap(
    latlons: list[tuple[float, float]],
    coords: list[tuple[float, float]],
    depot_latlon: tuple[float, float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not args.use_amap:
        return {"enabled": False, "scale": 1.0, "samples": []}
    api_key = args.amap_key or default_amap_key()
    if not api_key:
        print("[WARN] Amap calibration requested but no key was found; using scale=1.0")
        return {"enabled": False, "scale": 1.0, "samples": [], "warning": "missing_api_key"}

    cache_path = Path(args.amap_cache)
    if not cache_path.is_absolute():
        cache_path = ROOT / cache_path
    cache = load_cache(cache_path)

    n_nodes = len(coords)
    candidate_ids = list(range(1, n_nodes))
    if len(candidate_ids) > args.amap_sample_size:
        xs = np.array([coords[i][0] for i in candidate_ids])
        order = [candidate_ids[int(i)] for i in np.argsort(xs)]
        picks = np.linspace(0, len(order) - 1, args.amap_sample_size).round().astype(int)
        candidate_ids = [order[int(i)] for i in picks]

    raw_samples = []
    ratios = []
    for node_id in candidate_ids:
        try:
            api = amap_driving_duration(
                latlons[node_id],
                depot_latlon,
                api_key=api_key,
                cache=cache,
                sleep_sec=args.amap_sleep_sec,
            )
        except Exception as exc:
            raw_samples.append({"node_id": node_id, "error": str(exc)})
            continue
        model = road_time_seconds(
            coords[node_id],
            coords[0],
            args.east_west_kmh,
            args.north_south_kmh,
            args.local_kmh,
            args.fixed_sec,
        )
        ratio = float(api["duration_s"]) / max(model, 1.0)
        ratios.append(ratio)
        raw_samples.append({
            "node_id": node_id,
            "api_duration_s": api["duration_s"],
            "api_distance_m": api["distance_m"],
            "model_duration_s": round(model, 2),
            "ratio": round(ratio, 4),
        })

    save_cache(cache_path, cache)
    if not ratios:
        return {"enabled": True, "scale": 1.0, "samples": raw_samples, "warning": "no_valid_samples"}
    scale = float(np.clip(np.median(ratios), args.min_amap_scale, args.max_amap_scale))
    return {
        "enabled": True,
        "scale": scale,
        "median_raw_ratio": float(np.median(ratios)),
        "mean_raw_ratio": float(np.mean(ratios)),
        "n_valid_samples": len(ratios),
        "samples": raw_samples,
        "cache_path": str(cache_path),
    }


def build_choice_utility_and_adjacency(
    home_coords: list[tuple[float, float]],
    mp_coords: list[tuple[float, float]],
    mp_records: list[dict[str, Any]],
    passengers: int,
    n_mp: int,
    k: int,
    walk_speed_kmh: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    n_total = 1 + passengers + n_mp
    choice_util = np.full((n_total, n_total), np.nan, dtype=np.float32)
    walk_minutes = np.zeros((passengers, n_mp), dtype=np.float32)
    utility_values = np.zeros((passengers, n_mp), dtype=np.float32)

    for i, home in enumerate(home_coords):
        row_idx = 1 + i
        for j, mp in enumerate(mp_coords):
            col_idx = 1 + passengers + j
            d_km = math.sqrt((home[0] - mp[0]) ** 2 + (home[1] - mp[1]) ** 2)
            walk_min = d_km / max(walk_speed_kmh, 1e-6) * 60.0
            record = mp_records[j]
            util = (
                -0.038 * walk_min
                -0.012 * (float(record["headway_min"]) / 2.0)
                -0.030 * float(record["transfer_penalty_min"])
                + 0.38 * float(record["direct_count"] > 0)
                + 0.075 * min(float(record["route_count"]), 6.0)
                + 0.130 * (float(record["quality_score"]) - 1.0)
            )
            if record["is_suspended"]:
                util -= 0.25
            choice_util[row_idx, col_idx] = util
            walk_minutes[i, j] = walk_min
            utility_values[i, j] = util

    adjacency = np.zeros((1 + passengers, n_mp), dtype=np.int32)
    k_eff = min(int(k), n_mp)
    quality = np.array([float(r["quality_score"]) for r in mp_records])
    for i in range(passengers):
        selected: list[int] = []

        def add_many(indices: np.ndarray, count: int) -> None:
            for idx in indices:
                jj = int(idx)
                if jj not in selected:
                    selected.append(jj)
                if len(selected) >= count:
                    break

        add_many(np.argsort(walk_minutes[i]), min(k_eff, 5))
        add_many(np.argsort(-utility_values[i]), min(k_eff, 8))
        corridor_rank = np.argsort(-(quality - 0.025 * walk_minutes[i]))
        add_many(corridor_rank, k_eff)
        if len(selected) < k_eff:
            add_many(np.argsort(-utility_values[i]), k_eff)
        adjacency[1 + i, selected[:k_eff]] = 1

    diag = {
        "walk_min_mean": float(np.mean(walk_minutes)),
        "walk_min_p50": float(np.percentile(walk_minutes, 50)),
        "walk_min_p90": float(np.percentile(walk_minutes, 90)),
        "chosen_walk_min_mean": float(np.mean(walk_minutes[adjacency[1:] == 1])),
        "choice_util_mean": float(np.nanmean(choice_util)),
        "choice_util_p10": float(np.nanpercentile(choice_util, 10)),
        "choice_util_p90": float(np.nanpercentile(choice_util, 90)),
        "adjacency_unique_rows": int(len({tuple(row.tolist()) for row in adjacency[1:]})),
    }
    return choice_util, adjacency, diag


def write_duration_matrix(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("EDGE_WEIGHT_SECTION\n")
        for row in matrix:
            handle.write("\t".join(str(int(v)) for v in row.tolist()))
            handle.write("\n")


def write_metadata(path: Path, data: dict[str, Any]) -> None:
    meta = {
        "instance": "Beijing_Yanjiao",
        "scenario": "Yanjiao-Guomao commuter-aware DRT",
        "variant": "commuter_metric",
        "seed": data["seed"],
        "passengers": data["passengers"],
        "n_meeting_points": data["n_meeting_points"],
        "k_nearest": data["k_nearest"],
        "projection": "metric",
        "coordinate_system": "local_metric_km",
        "bounds": data["bounds"],
        "depot": {
            "name": "Guomao",
            "lon": data["depot_latlon"][0],
            "lat": data["depot_latlon"][1],
        },
        "generation_method": (
            "Residential POI density sampling with edge heterogeneity; "
            "commuter-quality bus-stop selection; calibrated corridor vehicle times; "
            "separate static choice utility sidecar."
        ),
        "duration_matrix": data["duration_diag"],
        "choice_utility": data["choice_diag"],
        "mp_quality": data["mp_quality_diag"],
        "amap_calibration": data["amap_calibration"],
        "data_sources": {
            "residential_pois": "data/yanjiao/residential_pois.csv",
            "bus_stops": "data/yanjiao/bus_stops.csv",
            "amap_driving_api": bool(data["amap_calibration"].get("enabled", False)),
        },
    }
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def write_mp_features(path: Path, mp_records: list[dict[str, Any]]) -> None:
    fields = [
        "mp_index", "source_index", "lon", "lat", "route_count", "direct_count",
        "is_direct", "is_suspended", "quality_score", "headway_min",
        "transfer_penalty_min", "tokens", "name", "address",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, record in enumerate(mp_records):
            raw = record["raw"]
            writer.writerow({
                "mp_index": i,
                "source_index": record["source_index"],
                "lon": f"{record['lon']:.6f}",
                "lat": f"{record['lat']:.6f}",
                "route_count": record["route_count"],
                "direct_count": record["direct_count"],
                "is_direct": int(record["is_direct"]),
                "is_suspended": int(record["is_suspended"]),
                "quality_score": f"{record['quality_score']:.4f}",
                "headway_min": f"{record['headway_min']:.2f}",
                "transfer_penalty_min": f"{record['transfer_penalty_min']:.2f}",
                "tokens": " ".join(record["tokens"]),
                "name": raw.get("name", ""),
                "address": raw.get("address", ""),
            })


def generate_one(args: argparse.Namespace, seed: int, residential_pois: list[dict[str, str]],
                 bus_stops: list[dict[str, str]]) -> dict[str, Any]:
    rng = np.random.RandomState(seed)
    passengers = int(args.passengers)
    n_mp = int(args.mp)
    depot_latlon = (float(GUOMAO_DESTINATION["lon"]), float(GUOMAO_DESTINATION["lat"]))

    homes_latlon = sample_home_latlons(
        residential_pois,
        passengers=passengers,
        rng=rng,
        home_edge_share=args.home_edge_share,
        sigma=args.sigma,
        max_perturb=args.max_perturb,
    )
    mp_records = select_commuter_meeting_points(bus_stops, n_mp, rng)
    mps_latlon = [(float(r["lon"]), float(r["lat"])) for r in mp_records]
    all_latlon = [depot_latlon] + homes_latlon + mps_latlon
    bounds = make_projection_bounds([p[0] for p in all_latlon], [p[1] for p in all_latlon], "metric")
    all_coords = project_latlon_points(all_latlon, bounds, "metric")
    home_coords = all_coords[1:1 + passengers]
    mp_coords = all_coords[1 + passengers:]

    calibration = calibrate_with_amap(all_latlon, all_coords, depot_latlon, args)
    duration_matrix = build_duration_matrix(
        all_coords,
        scale=float(calibration.get("scale", 1.0)),
        east_west_kmh=args.east_west_kmh,
        north_south_kmh=args.north_south_kmh,
        local_kmh=args.local_kmh,
        fixed_sec=args.fixed_sec,
    )
    choice_util, adjacency, choice_diag = build_choice_utility_and_adjacency(
        home_coords,
        mp_coords,
        mp_records,
        passengers=passengers,
        n_mp=n_mp,
        k=args.k,
        walk_speed_kmh=args.walk_speed_kmh,
    )

    service_times = np.zeros(len(all_coords), dtype=np.int32)
    mp_quality = np.array([float(r["quality_score"]) for r in mp_records])
    mp_direct = np.array([float(r["is_direct"]) for r in mp_records])
    mp_quality_diag = {
        "quality_mean": float(np.mean(mp_quality)),
        "quality_min": float(np.min(mp_quality)),
        "quality_max": float(np.max(mp_quality)),
        "direct_share": float(np.mean(mp_direct)),
        "suspended_count": int(sum(1 for r in mp_records if r["is_suspended"])),
    }
    duration_diag = {
        "max_duration_s": int(np.max(duration_matrix)),
        "depot_to_node_mean_s": float(np.mean(duration_matrix[0, 1:])),
        "mp_to_depot_mean_s": float(np.mean(duration_matrix[1 + passengers:, 0])),
        "home_to_depot_mean_s": float(np.mean(duration_matrix[1:1 + passengers, 0])),
    }

    return {
        "seed": seed,
        "passengers": passengers,
        "n_meeting_points": n_mp,
        "k_nearest": int(args.k),
        "bounds": bounds,
        "depot_latlon": depot_latlon,
        "depot_rel": all_coords[0],
        "home_locations_latlon": homes_latlon,
        "home_locations_rel": home_coords,
        "mp_locations_latlon": mps_latlon,
        "mp_locations_rel": mp_coords,
        "all_coords_latlon": all_latlon,
        "all_coords_rel": all_coords,
        "adjacency": adjacency,
        "duration_matrix": duration_matrix,
        "choice_utility": choice_util,
        "service_times": service_times,
        "mp_records": mp_records,
        "choice_diag": choice_diag,
        "duration_diag": duration_diag,
        "mp_quality_diag": mp_quality_diag,
        "amap_calibration": calibration,
    }


def write_outputs(prefix: str, data: dict[str, Any], k: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_coords_txt(OUT_DIR / f"{prefix}_coords.txt", data)
    write_coords_latlon(OUT_DIR / f"{prefix}_coords_latlon.txt", data)
    write_duration_matrix(OUT_DIR / f"{prefix}_duration_matrix.txt", data["duration_matrix"])
    np.save(OUT_DIR / f"{prefix}_adjacency{k}.npy", data["adjacency"])
    np.save(OUT_DIR / f"{prefix}_choice_utility.npy", data["choice_utility"])
    np.savetxt(OUT_DIR / f"{prefix}_service_times.txt", data["service_times"], fmt="%d")
    write_metadata(OUT_DIR / f"{prefix}_metadata.json", data)
    write_mp_features(OUT_DIR / f"{prefix}_mp_features.csv", data["mp_records"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate commuter-aware Yanjiao instances")
    parser.add_argument("--passengers", type=int, default=400)
    parser.add_argument("--mp", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--prefix", default="yanjiao_commuter_metric_{passengers}_{seed}")
    parser.add_argument("--home_edge_share", type=float, default=0.25)
    parser.add_argument("--sigma", type=float, default=0.0013)
    parser.add_argument("--max_perturb", type=float, default=0.0035)
    parser.add_argument("--walk_speed_kmh", type=float, default=4.8)
    parser.add_argument("--east_west_kmh", type=float, default=36.0)
    parser.add_argument("--north_south_kmh", type=float, default=18.0)
    parser.add_argument("--local_kmh", type=float, default=20.0)
    parser.add_argument("--fixed_sec", type=float, default=60.0)
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
    print(f"[INFO] Amap calibration={'on' if args.use_amap else 'off'}, sample_size={args.amap_sample_size}")

    for seed in args.seeds:
        prefix = args.prefix.format(passengers=args.passengers, n_passengers=args.passengers, seed=seed)
        print(f"\n[INFO] Generating {prefix}")
        data = generate_one(args, seed, residential_pois, bus_stops)
        write_outputs(prefix, data, args.k)
        cal = data["amap_calibration"]
        print(f"  duration max={data['duration_diag']['max_duration_s']}s, scale={float(cal.get('scale', 1.0)):.3f}")
        print(f"  choice util p10/p90={data['choice_diag']['choice_util_p10']:.3f}/{data['choice_diag']['choice_util_p90']:.3f}")
        print(f"  chosen walk mean={data['choice_diag']['chosen_walk_min_mean']:.2f} min")
        print(f"  adjacency unique rows={data['choice_diag']['adjacency_unique_rows']}/{args.passengers}")
        print(f"  mp direct share={data['mp_quality_diag']['direct_share']:.2%}")
        print(f"  wrote files under {OUT_DIR}")


if __name__ == "__main__":
    main()
