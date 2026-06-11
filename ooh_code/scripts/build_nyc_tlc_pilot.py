#!/usr/bin/env python3
"""Build the NYC_TLC real-world pilot instance.

The script converts public NYC TLC HVFHV trips, taxi-zone centroids, MTA GTFS
stops, and OSRM durations into the legacy instance files consumed by this
workspace. It intentionally treats OSRM as required: if a complete duration
matrix cannot be obtained, the pilot is marked blocked instead of silently
falling back to Euclidean travel times.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TLC_TRIP_URL_TEMPLATE = "https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_{month}.parquet"
TAXI_ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"

GTFS_FEEDS = {
    "subway": "http://web.mta.info/developers/data/nyct/subway/google_transit.zip",
    "bus_manhattan": "http://web.mta.info/developers/data/nyct/bus/google_transit_manhattan.zip",
    "bus_bronx": "http://web.mta.info/developers/data/nyct/bus/google_transit_bronx.zip",
    "bus_brooklyn": "http://web.mta.info/developers/data/nyct/bus/google_transit_brooklyn.zip",
    "bus_queens": "http://web.mta.info/developers/data/nyct/bus/google_transit_queens.zip",
    "bus_staten_island": "http://web.mta.info/developers/data/nyct/bus/google_transit_staten_island.zip",
}

PILOT_DATES = ["2024-03-05", "2024-03-06", "2024-03-07", "2024-03-12"]
GRAND_CENTRAL = {"name": "Grand Central Terminal", "lat": 40.7527, "lon": -73.9772}


@dataclass(frozen=True)
class Point:
    lat: float
    lon: float


def download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    print(f"[download] {url} -> {target}")
    with urllib.request.urlopen(url, timeout=120) as response:
        with target.open("wb") as handle:
            handle.write(response.read())


def haversine_km(a: Point, b: Point) -> float:
    radius_km = 6371.0088
    lat1 = math.radians(a.lat)
    lat2 = math.radians(b.lat)
    dlat = lat2 - lat1
    dlon = math.radians(b.lon - a.lon)
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(h))


def local_xy_km(points: list[Point]) -> list[tuple[float, float]]:
    min_lon = min(point.lon for point in points)
    min_lat = min(point.lat for point in points)
    mean_lat = sum(point.lat for point in points) / len(points)
    lon_scale = 111.320 * math.cos(math.radians(mean_lat))
    lat_scale = 110.574
    return [((point.lon - min_lon) * lon_scale, (point.lat - min_lat) * lat_scale) for point in points]


def read_taxi_zone_centroids(taxi_zip: Path) -> pd.DataFrame:
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise RuntimeError("geopandas is required to read NYC Taxi Zone shapefiles") from exc

    shapefile = taxi_zip.parent / "taxi_zones" / "taxi_zones.shp"
    if not shapefile.exists():
        with zipfile.ZipFile(taxi_zip) as zf:
            zf.extractall(taxi_zip.parent)
    zones = gpd.read_file(shapefile)
    if "LocationID" not in zones.columns:
        raise RuntimeError("Taxi Zone shapefile is missing LocationID")
    projected = zones.to_crs(2263)
    centroids = gpd.GeoDataFrame(
        zones[["LocationID", "borough", "zone"]].copy(),
        geometry=projected.geometry.centroid,
        crs=projected.crs,
    ).to_crs(4326)
    return pd.DataFrame(
        {
            "LocationID": centroids["LocationID"].astype(int),
            "borough": centroids["borough"].astype(str),
            "zone": centroids["zone"].astype(str),
            "lat": centroids.geometry.y.astype(float),
            "lon": centroids.geometry.x.astype(float),
        }
    )


def read_gtfs_stops(feed_paths: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for feed_path in feed_paths:
        if not feed_path.exists():
            print(f"[warn] GTFS feed missing, skipping: {feed_path}")
            continue
        with zipfile.ZipFile(feed_path) as zf:
            with zf.open("stops.txt") as handle:
                frame = pd.read_csv(handle)
        if not {"stop_id", "stop_name", "stop_lat", "stop_lon"}.issubset(frame.columns):
            print(f"[warn] GTFS feed missing stop columns, skipping: {feed_path}")
            continue
        frame = frame[["stop_id", "stop_name", "stop_lat", "stop_lon"]].copy()
        frame["feed"] = feed_path.stem
        frames.append(frame)
    if not frames:
        raise RuntimeError("No readable GTFS stops were found")
    stops = pd.concat(frames, ignore_index=True)
    stops = stops.dropna(subset=["stop_lat", "stop_lon"])
    stops["stop_lat"] = stops["stop_lat"].astype(float)
    stops["stop_lon"] = stops["stop_lon"].astype(float)
    stops["dedupe"] = (
        stops["stop_name"].astype(str)
        + "|"
        + stops["stop_lat"].round(5).astype(str)
        + "|"
        + stops["stop_lon"].round(5).astype(str)
    )
    stops = stops.drop_duplicates("dedupe").reset_index(drop=True)
    return stops.drop(columns=["dedupe"])


def read_trip_columns(parquet_path: Path, start: str, end: str) -> pd.DataFrame:
    needed = ["pickup_datetime", "dropoff_datetime", "PULocationID", "DOLocationID"]
    try:
        import pyarrow.dataset as ds

        dataset = ds.dataset(parquet_path, format="parquet")
        columns = set(dataset.schema.names)
        pickup_col = "pickup_datetime" if "pickup_datetime" in columns else "request_datetime"
        required = [pickup_col, "dropoff_datetime", "PULocationID", "DOLocationID"]
        missing = [column for column in required if column not in columns]
        if missing:
            raise RuntimeError(f"TLC parquet missing required columns: {missing}")
        table = dataset.to_table(
            columns=required,
            filter=(ds.field(pickup_col) >= pd.Timestamp(start).to_pydatetime())
            & (ds.field(pickup_col) < pd.Timestamp(end).to_pydatetime()),
        )
        trips = table.to_pandas()
        if pickup_col != "pickup_datetime":
            trips = trips.rename(columns={pickup_col: "pickup_datetime"})
        return trips
    except ImportError:
        trips = pd.read_parquet(parquet_path, columns=needed)
        trips = trips[(trips["pickup_datetime"] >= start) & (trips["pickup_datetime"] < end)]
        return trips


def filter_terminal_trips(trips: pd.DataFrame, zones: pd.DataFrame, service_date: str, terminal_radius_km: float) -> pd.DataFrame:
    terminal = Point(GRAND_CENTRAL["lat"], GRAND_CENTRAL["lon"])
    zone_points = {
        int(row.LocationID): Point(float(row.lat), float(row.lon))
        for row in zones.itertuples(index=False)
    }
    trips = trips.copy()
    trips["pickup_datetime"] = pd.to_datetime(trips["pickup_datetime"])
    target_date = pd.Timestamp(service_date).date()
    trips = trips[trips["pickup_datetime"].dt.date == target_date]
    trips = trips[
        (trips["pickup_datetime"].dt.time >= dtime(7, 0))
        & (trips["pickup_datetime"].dt.time <= dtime(9, 0))
    ]
    trips = trips[trips["pickup_datetime"].dt.weekday < 5]
    trips = trips.dropna(subset=["PULocationID", "DOLocationID"])
    trips["PULocationID"] = trips["PULocationID"].astype(int)
    trips["DOLocationID"] = trips["DOLocationID"].astype(int)
    trips = trips[trips["PULocationID"].isin(zone_points) & trips["DOLocationID"].isin(zone_points)]

    def dist_to_terminal(location_id: int) -> float:
        return haversine_km(zone_points[location_id], terminal)

    trips["dropoff_terminal_km"] = trips["DOLocationID"].map(dist_to_terminal)
    trips["pickup_terminal_km"] = trips["PULocationID"].map(dist_to_terminal)
    return trips[
        (trips["dropoff_terminal_km"] <= terminal_radius_km)
        & (trips["pickup_terminal_km"] > terminal_radius_km)
    ].sort_values("pickup_datetime")


def choose_meeting_points(origins: list[Point], stops: pd.DataFrame, max_meeting_points: int, k: int) -> pd.DataFrame:
    stop_points = [Point(float(row.stop_lat), float(row.stop_lon)) for row in stops.itertuples(index=False)]
    scores = np.zeros(len(stop_points), dtype=int)
    nearest_by_origin: list[np.ndarray] = []
    for origin in origins:
        distances = np.array([haversine_km(origin, stop) for stop in stop_points])
        nearest = np.argsort(distances)[: max(k, 10)]
        nearest_by_origin.append(nearest)
        scores[nearest] += 1
    selected: list[int] = list(np.argsort(-scores)[:max_meeting_points])
    selected_set = set(selected)
    for nearest in nearest_by_origin:
        covered = [idx for idx in nearest if idx in selected_set]
        if len(covered) < k:
            for idx in nearest:
                if len(selected_set) >= max_meeting_points:
                    break
                selected_set.add(int(idx))
            selected = list(selected_set)
    selected = sorted(selected, key=lambda idx: (-scores[idx], idx))[:max_meeting_points]
    return stops.iloc[selected].reset_index(drop=True)


def osrm_duration_matrix(points: list[Point], osrm_url: str, retries: int = 2) -> np.ndarray:
    coords = ";".join(f"{point.lon:.7f},{point.lat:.7f}" for point in points)
    url = f"{osrm_url.rstrip('/')}/table/v1/driving/{coords}?annotations=duration"
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=180) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("code") != "Ok":
                raise RuntimeError(payload.get("message", payload.get("code", "unknown OSRM error")))
            durations = payload.get("durations")
            matrix = np.array(durations, dtype=float)
            if matrix.shape != (len(points), len(points)) or np.isnan(matrix).any():
                raise RuntimeError("OSRM returned missing or malformed durations")
            return np.rint(matrix).astype(int)
        except Exception as exc:  # noqa: BLE001 - report final OSRM failure clearly.
            last_error = exc
            time.sleep(1 + attempt)
    raise RuntimeError(f"OSRM duration matrix failed: {last_error}") from last_error


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("EDGE_WEIGHT_SECTION\n")
        for row in matrix:
            handle.write("\t".join(str(int(value)) for value in row) + "\n")


def write_coords(path: Path, points: list[Point]) -> None:
    xy = local_xy_km(points)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("NODE_COORD_SECTION\n")
        for idx, (x_coord, y_coord) in enumerate(xy):
            handle.write(f"{idx}\t{x_coord:.6f}\t{y_coord:.6f}\n")


def write_map(path: Path, origins: list[Point], meeting_locations: list[Point]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available; skipping pilot map")
        return
    terminal = Point(GRAND_CENTRAL["lat"], GRAND_CENTRAL["lon"])
    plt.figure(figsize=(7, 6))
    plt.scatter([p.lon for p in origins], [p.lat for p in origins], s=14, alpha=0.65, label="TLC origins")
    plt.scatter(
        [p.lon for p in meeting_locations],
        [p.lat for p in meeting_locations],
        s=28,
        marker="s",
        alpha=0.85,
        label="MTA meeting points",
    )
    plt.scatter([terminal.lon], [terminal.lat], s=80, marker="*", label="Grand Central")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def build_seed(args: argparse.Namespace, seed: int, service_date: str, trips: pd.DataFrame, zones: pd.DataFrame, stops: pd.DataFrame) -> dict:
    filtered = filter_terminal_trips(trips, zones, service_date, args.terminal_radius_km)
    if len(filtered) == 0:
        raise RuntimeError(f"No terminal-bound trips found for {service_date}")
    sampled = filtered.head(args.max_passengers).copy()
    zone_lookup = {
        int(row.LocationID): Point(float(row.lat), float(row.lon))
        for row in zones.itertuples(index=False)
    }
    origins = [zone_lookup[int(location_id)] for location_id in sampled["PULocationID"]]
    meeting_points = choose_meeting_points(origins, stops, args.max_meeting_points, args.k)
    meeting_locations = [Point(float(row.stop_lat), float(row.stop_lon)) for row in meeting_points.itertuples(index=False)]
    if len(meeting_locations) < args.k:
        raise RuntimeError(f"Only {len(meeting_locations)} meeting points available; k={args.k}")

    all_points = [Point(GRAND_CENTRAL["lat"], GRAND_CENTRAL["lon"])] + origins + meeting_locations
    duration_matrix = osrm_duration_matrix(all_points, args.osrm_url)

    adjacency = np.zeros((1 + len(origins), len(meeting_locations)), dtype=int)
    for customer_idx, origin in enumerate(origins, start=1):
        distances = np.array([haversine_km(origin, stop) for stop in meeting_locations])
        nearest = np.argsort(distances)[:args.k]
        adjacency[customer_idx, nearest] = 1
        if adjacency[customer_idx].sum() != args.k:
            raise RuntimeError(f"Customer {customer_idx} does not have exactly k={args.k} meeting points")

    output_dir = args.output_root / "Environments" / "OOH" / "NYC_TLC" / "pilot"
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / f"nyc_tlc_pilot_{seed}"
    write_coords(base.with_name(base.name + "_coords.txt"), all_points)
    write_matrix(base.with_name(base.name + "_duration_matrix.txt"), duration_matrix)
    np.save(base.with_name(base.name + f"_adjacency{args.k}.npy"), adjacency)
    np.savetxt(base.with_name(base.name + "_service_times.txt"), np.zeros(len(all_points)), fmt="%.1f")
    write_map(base.with_name(base.name + "_map.png"), origins, meeting_locations)

    metadata = {
        "instance": "NYC_TLC",
        "service_date": service_date,
        "seed": seed,
        "source_month": args.month,
        "terminal": GRAND_CENTRAL,
        "time_window": "07:00-09:00",
        "terminal_radius_km": args.terminal_radius_km,
        "candidate_trip_count": int(len(filtered)),
        "n_passengers": int(len(origins)),
        "n_meeting_points": int(len(meeting_locations)),
        "k": int(args.k),
        "coords_order": "depot/terminal id 0, passengers id 1..N, meeting points id N+1..N+M",
        "adjacency_runtime_shape": list(adjacency.shape),
        "adjacency_customer_shape": [int(len(origins)), int(len(meeting_locations))],
        "duration_matrix_shape": list(duration_matrix.shape),
        "osrm_url": args.osrm_url,
        "osrm_status": "ok",
        "meeting_point_source": "MTA GTFS stops",
        "origin_source": "TLC HVFHV pickup taxi-zone centroids",
    }
    base.with_name(base.name + "_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def audit(args: argparse.Namespace, trips: pd.DataFrame, zones: pd.DataFrame, stops: pd.DataFrame) -> dict:
    rows = []
    for service_date in args.pilot_dates:
        filtered = filter_terminal_trips(trips, zones, service_date, args.terminal_radius_km)
        rows.append({"service_date": service_date, "candidate_trip_count": int(len(filtered))})
    return {
        "month": args.month,
        "pilot_dates": rows,
        "taxi_zone_count": int(len(zones)),
        "gtfs_stop_count": int(len(stops)),
        "terminal": GRAND_CENTRAL,
        "time_window": "07:00-09:00",
        "terminal_radius_km": args.terminal_radius_km,
    }


def prepare_raw_files(args: argparse.Namespace) -> tuple[Path, Path, list[Path]]:
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    trip_path = args.raw_dir / f"fhvhv_tripdata_{args.month}.parquet"
    taxi_zip = args.raw_dir / "taxi_zones.zip"
    if args.download:
        download_file(TLC_TRIP_URL_TEMPLATE.format(month=args.month), trip_path)
        download_file(TAXI_ZONES_URL, taxi_zip)
        for feed_name, feed_url in GTFS_FEEDS.items():
            download_file(feed_url, args.raw_dir / f"{feed_name}.zip")
    gtfs_paths = [args.raw_dir / f"{feed_name}.zip" for feed_name in GTFS_FEEDS]
    return trip_path, taxi_zip, gtfs_paths


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NYC_TLC pilot instances.")
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--raw-dir", type=Path, default=repo_root / "data_raw" / "nyc_tlc")
    parser.add_argument("--output-root", type=Path, default=repo_root)
    parser.add_argument("--month", default="2024-03")
    parser.add_argument("--pilot-dates", nargs="+", default=PILOT_DATES)
    parser.add_argument("--download", action="store_true", help="Download missing public input files.")
    parser.add_argument("--audit-only", action="store_true", help="Only write audit JSON; do not call OSRM or build instances.")
    parser.add_argument("--max-passengers", type=int, default=80)
    parser.add_argument("--max-meeting-points", type=int, default=40)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--terminal-radius-km", type=float, default=1.5)
    parser.add_argument("--osrm-url", default="http://localhost:5000")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    trip_path, taxi_zip, gtfs_paths = prepare_raw_files(args)
    missing = [str(path) for path in [trip_path, taxi_zip, *gtfs_paths] if not path.exists()]
    if missing:
        print("[blocked] Missing input files. Re-run with --download or place files manually:")
        for item in missing:
            print("  - " + item)
        return 2

    start = min(args.pilot_dates)
    end = (pd.Timestamp(max(args.pilot_dates)) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    zones = read_taxi_zone_centroids(taxi_zip)
    stops = read_gtfs_stops(gtfs_paths)
    trips = read_trip_columns(trip_path, start, end)

    audit_payload = audit(args, trips, zones, stops)
    audit_dir = args.output_root / "Experiments" / "analysis" / "nyc_tlc_pilot"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "data_audit.json").write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")
    print(json.dumps(audit_payload, indent=2))
    if args.audit_only:
        return 0

    built = []
    for seed, service_date in enumerate(args.pilot_dates):
        built.append(build_seed(args, seed, service_date, trips, zones, stops))
    (audit_dir / "instance_metadata_summary.json").write_text(json.dumps(built, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
