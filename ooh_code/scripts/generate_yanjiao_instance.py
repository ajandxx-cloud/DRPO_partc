"""
Step 3: 生成燕郊-国贸实验数据实例。

输入: data/yanjiao/residential_pois.csv, bus_stops.csv, guomao_bus_stops.csv
输出: ooh_code/Environments/OOH/Beijing_Yanjiao/ 下的 coords.txt + adjacency.npy + metadata.json

生成逻辑:
  1. 住宅POI → 密度加权 → 采样N个anchors → 加高斯扰动 → home locations
  2. 公交站POI → 筛选燕郊侧 → 均匀采样M个 → meeting points
  3. 固定 destination = 国贸/郎家园
  4. 双输出: 实验坐标 + WGS84经纬度 (用于地图)

用法:
  python scripts/generate_yanjiao_instance.py                          # 默认: 300人/100点, seed=0,1
  python scripts/generate_yanjiao_instance.py --passengers 150 --mp 80  # 自定义规模
  python scripts/generate_yanjiao_instance.py --all-scales              # 生成全部4个规模
  python scripts/generate_yanjiao_instance.py --projection metric --prefix yanjiao_metric_{passengers}_{seed}
"""
import csv
import json
import math
import argparse
import sys
from pathlib import Path

import numpy as np

# --- 路径 ---
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "yanjiao"
OUT_DIR = ROOT / "Environments" / "OOH" / "Beijing_Yanjiao"

# --- 国贸/郎家园 destination 坐标 (Amap GCJ-02) ---
GUOMAO_DESTINATION = {
    "name": "郎家园/国贸",
    "lon": 116.4610,  # 郎家园附近
    "lat": 39.9087,
}

# --- 默认参数 ---
DEFAULT_PASSENGERS = 300
DEFAULT_MEETING_POINTS = 100
DEFAULT_K = 10  # 邻接矩阵k值
DEFAULT_SIGMA = 0.0008  # 扰动sigma (~80m in lat/lon)
DEFAULT_MAX_PERTURB = 0.002  # 最大扰动 ~200m
DENSITY_RADIUS = 0.005  # 密度计算半径 ~500m
SEEDS = [0, 1]  # train/test seeds
LON_LAT_MARGIN = 0.005
METERS_PER_DEG_LAT = 110540.0
METERS_PER_DEG_LON = 111320.0

VARIANT_SETTINGS = {
    "base": {
        "prefix": "yanjiao_{passengers}_{seed}",
        "home_edge_share": 0.0,
        "mp_edge_share": 0.0,
        "sigma": DEFAULT_SIGMA,
        "max_perturb": DEFAULT_MAX_PERTURB,
        "description": "Original density-weighted homes and priority/random meeting points.",
    },
    "het_home": {
        "prefix": "yanjiao_het_home_{passengers}_{seed}",
        "home_edge_share": 0.30,
        "mp_edge_share": 0.0,
        "sigma": 0.0015,
        "max_perturb": 0.004,
        "description": "70% dense residential anchors and 30% edge residential anchors.",
    },
    "mixed_mp": {
        "prefix": "yanjiao_mixed_mp_{passengers}_{seed}",
        "home_edge_share": 0.0,
        "mp_edge_share": 0.30,
        "sigma": DEFAULT_SIGMA,
        "max_perturb": DEFAULT_MAX_PERTURB,
        "description": "70% priority/ordinary meeting points and 30% edge meeting points.",
    },
    "het_home_mixed_mp": {
        "prefix": "yanjiao_het_home_mixed_mp_{passengers}_{seed}",
        "home_edge_share": 0.30,
        "mp_edge_share": 0.30,
        "sigma": 0.0015,
        "max_perturb": 0.004,
        "description": "Heterogeneous homes plus mixed-quality meeting points.",
    },
    "dispersed": {
        "prefix": "yanjiao_dispersed_{passengers}_{seed}",
        "home_edge_share": 0.40,
        "mp_edge_share": 0.30,
        "sigma": 0.0030,
        "max_perturb": 0.0060,
        "description": "Uniform random home sampling within bus-stop bounding box (Beijing_bus style).",
    },
}


def load_csv(filepath: Path) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_location(loc_str: str) -> tuple[float, float]:
    """解析高德 'lon,lat' 格式字符串。"""
    parts = loc_str.split(",")
    return float(parts[0]), float(parts[1])


def compute_density_weights(pois: list[dict], radius: float = DENSITY_RADIUS) -> np.ndarray:
    """
    对每个住宅POI计算密度权重: 以该POI为中心radius内的其他住宅POI数量。
    这是人口网格的简化替代 — 住宅密度越高，权重越大。
    """
    lons = np.array([float(p["location"].split(",")[0]) for p in pois])
    lats = np.array([float(p["location"].split(",")[1]) for p in pois])
    weights = np.zeros(len(pois), dtype=np.float64)

    for i in range(len(pois)):
        # 简化的距离: 在燕郊尺度上直接用经纬度差近似
        dlon = (lons - lons[i]) * 111320 * math.cos(math.radians(lats[i]))
        dlat = (lats - lats[i]) * 110540
        dist = np.sqrt(dlon**2 + dlat**2)
        weights[i] = np.sum(dist < (radius * 111320))  # radius转米

    # 确保最小权重
    weights = np.maximum(weights, 1.0)
    return weights / weights.sum()


def weighted_sample_anchors(pois: list[dict], weights: np.ndarray, n: int, rng: np.random.RandomState) -> list[dict]:
    """按权重采样n个住宅POI作为home anchor。"""
    indices = rng.choice(len(pois), size=n, replace=True, p=weights)
    return [pois[i] for i in indices]


def edge_ranked_indices(pois: list[dict]) -> np.ndarray:
    lons = np.array([parse_location(p["location"])[0] for p in pois], dtype=np.float64)
    lats = np.array([parse_location(p["location"])[1] for p in pois], dtype=np.float64)
    center_lon = float(np.median(lons))
    center_lat = float(np.median(lats))
    dx = (lons - center_lon) * 111320 * math.cos(math.radians(center_lat))
    dy = (lats - center_lat) * 110540
    return np.argsort(np.sqrt(dx**2 + dy**2))[::-1]


def sample_home_anchors(
        pois: list[dict], weights: np.ndarray, n: int, rng: np.random.RandomState,
        edge_share: float) -> list[dict]:
    if edge_share <= 0.0:
        return weighted_sample_anchors(pois, weights, n, rng)
    n_edge = int(round(n * edge_share))
    n_dense = n - n_edge
    anchors = weighted_sample_anchors(pois, weights, n_dense, rng)
    ranked = edge_ranked_indices(pois)
    edge_pool_size = max(n_edge, int(math.ceil(len(pois) * 0.35)))
    edge_pool = ranked[:edge_pool_size]
    if n_edge > 0:
        edge_indices = rng.choice(edge_pool, size=n_edge, replace=True)
        anchors.extend(pois[int(i)] for i in edge_indices)
    rng.shuffle(anchors)
    return anchors


def sample_homes_grid_uniform(
        pois: list[dict], n: int, rng: np.random.RandomState,
        n_lon_bins: int = 6, n_lat_bins: int = 3,
        edge_share: float = 0.40) -> list[dict]:
    """将燕郊 POI 划分为均匀网格，每格等量采样，避免密集区域过度集中。
    edge_share 比例的采样量从边缘 POI 中选取（距中心最远的 POI）。"""
    lons = np.array([parse_location(p["location"])[0] for p in pois])
    lats = np.array([parse_location(p["location"])[1] for p in pois])

    lon_edges = np.linspace(float(lons.min()) - 1e-6, float(lons.max()) + 1e-6, n_lon_bins + 1)
    lat_edges = np.linspace(float(lats.min()) - 1e-6, float(lats.max()) + 1e-6, n_lat_bins + 1)

    # Assign each POI to a grid cell
    cells: dict[tuple[int, int], list[int]] = {}
    for i, p in enumerate(pois):
        lon, lat = parse_location(p["location"])
        ci = int(np.searchsorted(lon_edges, lon, side="right")) - 1
        ri = int(np.searchsorted(lat_edges, lat, side="right")) - 1
        ci = max(0, min(ci, n_lon_bins - 1))
        ri = max(0, min(ri, n_lat_bins - 1))
        cells.setdefault((ci, ri), []).append(i)

    non_empty = [k for k, v in cells.items() if v]
    n_grid = int(round(n * (1 - edge_share)))
    n_cells = len(non_empty)

    # Equal allocation across grid cells
    base_per_cell = n_grid // max(n_cells, 1)
    remainder = n_grid - base_per_cell * n_cells
    rng.shuffle(non_empty)

    anchors: list[dict] = []
    quota: dict[tuple[int, int], int] = {}
    for i, key in enumerate(non_empty):
        quota[key] = base_per_cell + (1 if i < remainder else 0)

    for key in non_empty:
        pool = cells[key]
        n_pick = quota[key]
        idx = rng.choice(pool, size=n_pick, replace=True)
        anchors.extend(pois[int(j)] for j in idx)

    # Edge POI sampling
    if edge_share > 0:
        n_edge = n - len(anchors)
        ranked = edge_ranked_indices(pois)
        edge_pool_size = max(n_edge, int(math.ceil(len(pois) * 0.35)))
        edge_pool = ranked[:edge_pool_size]
        edge_indices = rng.choice(edge_pool, size=n_edge, replace=True)
        anchors.extend(pois[int(i)] for i in edge_indices)

    rng.shuffle(anchors)
    return anchors[:n]


def sample_homes_uniform_random(
        bus_stops: list[dict], n: int, rng: np.random.RandomState,
        margin: float = 0.002) -> list[tuple[float, float]]:
    """在 bus stops bounding box 内均匀随机生成 home 坐标 (lon, lat)。
    margin: 向外扩展的经纬度余量（约 200m）。"""
    lons = np.array([parse_location(s["location"])[0] for s in bus_stops])
    lats = np.array([parse_location(s["location"])[1] for s in bus_stops])
    lon_min, lon_max = float(lons.min()) - margin, float(lons.max()) + margin
    lat_min, lat_max = float(lats.min()) - margin, float(lats.max()) + margin
    home_lons = rng.uniform(lon_min, lon_max, size=n)
    home_lats = rng.uniform(lat_min, lat_max, size=n)
    return list(zip(home_lons.tolist(), home_lats.tolist()))


def perturb_anchor(lon: float, lat: float, sigma: float, max_d: float, rng: np.random.RandomState) -> tuple[float, float]:
    """在anchor周围添加高斯扰动，限制最大偏移。"""
    # 采样直到满足max_d约束
    for _ in range(10):
        dx = rng.normal(0, sigma)
        dy = rng.normal(0, sigma)
        if math.sqrt(dx**2 + dy**2) <= max_d:
            return lon + dx, lat + dy
    # fallback: 截断
    d = math.sqrt(dx**2 + dy**2)
    if d > max_d:
        dx = dx / d * max_d
        dy = dy / d * max_d
    return lon + dx, lat + dy


def select_meeting_points(bus_stops: list[dict], n: int, rng: np.random.RandomState,
                           priority_stops: list[dict] | None = None) -> list[dict]:
    """
    从公交站中选取meeting points。
    优先包含关键字站点（定制快巴/818路线），其余均匀随机采样。
    """
    if priority_stops is None:
        priority_stops = []

    # 按经纬度去重
    seen = set()
    unique_stops = []
    for s in bus_stops:
        loc = s["location"]
        if loc not in seen:
            seen.add(loc)
            unique_stops.append(s)

    # 优先站点去重
    priority_locs = set()
    priority_list = []
    for s in priority_stops:
        loc = s["location"]
        if loc not in priority_locs and loc in seen:
            priority_locs.add(loc)
            priority_list.append(s)

    # 先从优先站点中选取(最多占50%)
    n_priority = min(len(priority_list), n // 2)
    selected_priority = list(rng.choice(priority_list, size=n_priority, replace=False)) if n_priority > 0 else []

    # 剩余从所有公交站中均匀采样
    remaining = n - n_priority
    # 排除已选的优先站点
    pool = [s for s in unique_stops if s["location"] not in priority_locs]
    selected_remaining = list(rng.choice(pool, size=remaining, replace=False)) if remaining > 0 else []

    return selected_priority + selected_remaining


def select_mixed_meeting_points(bus_stops: list[dict], n: int, rng: np.random.RandomState,
                                priority_stops: list[dict] | None = None,
                                edge_share: float = 0.0) -> list[dict]:
    if edge_share <= 0.0:
        return select_meeting_points(bus_stops, n, rng, priority_stops=priority_stops)
    if priority_stops is None:
        priority_stops = []

    seen = set()
    unique_stops = []
    for stop in bus_stops:
        loc = stop["location"]
        if loc not in seen:
            seen.add(loc)
            unique_stops.append(stop)

    priority_locs = {stop["location"] for stop in priority_stops if stop.get("location") in seen}
    n_edge = int(round(n * edge_share))
    n_core = n - n_edge
    core = select_meeting_points(unique_stops, n_core, rng, priority_stops=priority_stops)
    selected_locs = {stop["location"] for stop in core}
    edge_pool = [
        unique_stops[int(i)] for i in edge_ranked_indices(unique_stops)
        if unique_stops[int(i)]["location"] not in selected_locs
        and unique_stops[int(i)]["location"] not in priority_locs
    ]
    if len(edge_pool) < n_edge:
        edge_pool = [
            unique_stops[int(i)] for i in edge_ranked_indices(unique_stops)
            if unique_stops[int(i)]["location"] not in selected_locs
        ]
    edge = list(rng.choice(edge_pool, size=n_edge, replace=False)) if n_edge > 0 else []
    return core + edge


def latlon_to_relative(lon: float, lat: float, bounds: dict) -> tuple[float, float]:
    """将经纬度投影到相对坐标 (0~range)。"""
    rel_x = (lon - bounds["lon_min"]) / (bounds["lon_max"] - bounds["lon_min"]) * bounds["rel_range"]
    rel_y = (lat - bounds["lat_min"]) / (bounds["lat_max"] - bounds["lat_min"]) * bounds["rel_range"]
    return round(rel_x, 4), round(rel_y, 4)


def latlon_to_metric_km(lon: float, lat: float, bounds: dict) -> tuple[float, float]:
    """Project lon/lat to local equirectangular coordinates in kilometers."""
    ref_lat = float(bounds["ref_lat"])
    x_m = (lon - bounds["lon_origin"]) * METERS_PER_DEG_LON * math.cos(math.radians(ref_lat))
    y_m = (lat - bounds["lat_origin"]) * METERS_PER_DEG_LAT
    return round(x_m / 1000.0, 4), round(y_m / 1000.0, 4)


def make_projection_bounds(all_lons: list[float], all_lats: list[float], projection: str) -> dict:
    lon_min = min(all_lons) - LON_LAT_MARGIN
    lon_max = max(all_lons) + LON_LAT_MARGIN
    lat_min = min(all_lats) - LON_LAT_MARGIN
    lat_max = max(all_lats) + LON_LAT_MARGIN

    if projection == "relative":
        return {
            "lon_min": lon_min,
            "lon_max": lon_max,
            "lat_min": lat_min,
            "lat_max": lat_max,
            "rel_range": 60.0,
        }

    if projection == "metric":
        ref_lat = (lat_min + lat_max) / 2.0
        x_span_km = (lon_max - lon_min) * METERS_PER_DEG_LON * math.cos(math.radians(ref_lat)) / 1000.0
        y_span_km = (lat_max - lat_min) * METERS_PER_DEG_LAT / 1000.0
        return {
            "lon_min": lon_min,
            "lon_max": lon_max,
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_origin": lon_min,
            "lat_origin": lat_min,
            "ref_lat": ref_lat,
            "x_span_km": x_span_km,
            "y_span_km": y_span_km,
            "coordinate_unit": "km",
        }

    raise ValueError(f"Unsupported projection: {projection}")


def project_latlon_points(points: list[tuple[float, float]], bounds: dict, projection: str) -> list[tuple[float, float]]:
    if projection == "relative":
        return [latlon_to_relative(lon, lat, bounds) for lon, lat in points]
    if projection == "metric":
        return [latlon_to_metric_km(lon, lat, bounds) for lon, lat in points]
    raise ValueError(f"Unsupported projection: {projection}")


def projection_coordinate_system(projection: str) -> str:
    if projection == "relative":
        return "relative_projection"
    if projection == "metric":
        return "local_metric_km"
    raise ValueError(f"Unsupported projection: {projection}")


def generate_instance(passengers: int, n_mp: int, seed: int,
                       residential_pois: list[dict], bus_stops: list[dict],
                       key_stops: list[dict], destination: dict,
                       variant: str = "base", k: int = DEFAULT_K,
                       projection: str = "relative") -> dict:
    """生成一个seed的完整数据实例。"""
    rng = np.random.RandomState(seed)
    rng_mp = np.random.RandomState(42)  # 固定种子: MP(公交站点)跨seed一致
    settings = VARIANT_SETTINGS[variant]

    # --- 1. 采样home anchors ---
    anchors = None
    home_locations_latlon: list[tuple[float, float]] = []
    if variant == "dispersed":
        home_locations_latlon = sample_homes_uniform_random(
            bus_stops, passengers, rng)
    else:
        weights = compute_density_weights(residential_pois)
        anchors = sample_home_anchors(
            residential_pois, weights, passengers, rng,
            edge_share=float(settings["home_edge_share"]),
        )

    # --- 2. 生成home locations (加扰动, dispersed 已直接生成) ---
    if anchors is not None:
        for a in anchors:
            lon, lat = parse_location(a["location"])
            plon, plat = perturb_anchor(
                lon, lat,
                float(settings["sigma"]),
                float(settings["max_perturb"]),
                rng,
            )
            home_locations_latlon.append((plon, plat))

    # --- 3. 选取meeting points ---
    mps = select_mixed_meeting_points(
        bus_stops, n_mp, rng_mp,
        priority_stops=key_stops,
        edge_share=float(settings["mp_edge_share"]),
    )
    mp_locations_latlon = [parse_location(m["location"]) for m in mps]

    # --- 4. Depot = 国贸destination (many-to-one通勤终点) ---
    depot_lon = destination["lon"]
    depot_lat = destination["lat"]

    # --- 5. 计算投影边界 (覆盖燕郊home + MP + 国贸depot) ---
    all_lons = [hl[0] for hl in home_locations_latlon] + [mp[0] for mp in mp_locations_latlon] + [depot_lon]
    all_lats = [hl[1] for hl in home_locations_latlon] + [mp[1] for mp in mp_locations_latlon] + [depot_lat]

    bounds = make_projection_bounds(all_lons, all_lats, projection)

    # --- 6. 投影坐标 ---
    depot_rel = project_latlon_points([(depot_lon, depot_lat)], bounds, projection)[0]
    home_rel = project_latlon_points(home_locations_latlon, bounds, projection)
    mp_rel = project_latlon_points(mp_locations_latlon, bounds, projection)

    # --- 7. 构建coords数组 (Location格式) ---
    # 顺序: depot(国贸), home_1..home_N, mp_1..mp_M
    # Depot就是many-to-one通勤的终点，不需要单独的destination节点
    all_coords_rel = [depot_rel] + home_rel + mp_rel

    # --- 8. 构建邻接矩阵 ---
    # 形状: (n_non_mp, n_mp) = (1 + passengers, n_mp)
    # 行0 = depot (不会被customer采样到，但保持索引对齐)
    n_total = 1 + passengers + n_mp  # depot + homes + mps
    dist_matrix = np.zeros((n_total, n_total))
    for i in range(n_total):
        for j in range(n_total):
            dx = all_coords_rel[i][0] - all_coords_rel[j][0]
            dy = all_coords_rel[i][1] - all_coords_rel[j][1]
            dist_matrix[i][j] = math.sqrt(dx**2 + dy**2)

    n_non_mp = 1 + passengers  # depot + homes
    adjacency = np.zeros((n_non_mp, n_mp), dtype=np.int32)
    k_eff = min(int(k), n_mp)
    for i in range(1, n_non_mp):  # 跳过depot (row 0 保持全0)
        mp_start = n_non_mp  # MPs从索引 passengers+1 开始
        mp_dists = dist_matrix[i][mp_start:mp_start + n_mp]
        closest = np.argsort(mp_dists)[:k_eff]
        adjacency[i][closest] = 1

    # --- 9. 诊断输出 ---
    arr = np.array(all_coords_rel)
    home_arr = arr[1:1 + passengers]
    depot_arr = arr[0]
    mp_arr = arr[1 + passengers:]
    home_depot_dists = np.sqrt(((home_arr - depot_arr) ** 2).sum(axis=1))
    home_depot_cv = home_depot_dists.std() / max(home_depot_dists.mean(), 1e-9) * 100
    nearest_mp_dists = np.sqrt(
        ((mp_arr[np.newaxis, :, :] - home_arr[:, np.newaxis, :]) ** 2).sum(axis=2)
    ).min(axis=1)
    print(f"  Diagnostics: home-depot CV={home_depot_cv:.1f}%, "
          f"home-depot mean={home_depot_dists.mean():.1f}, "
          f"nearest-MP mean={nearest_mp_dists.mean():.2f} "
          f"(p50={np.percentile(nearest_mp_dists, 50):.2f}, max={nearest_mp_dists.max():.2f})")

    # --- 10. 构建输出 ---
    return {
        "seed": seed,
        "passengers": passengers,
        "n_meeting_points": n_mp,
        "k_nearest": int(k),
        "bounds": bounds,
        "depot_name": destination["name"],
        "depot_latlon": (depot_lon, depot_lat),
        "depot_rel": depot_rel,
        "home_locations_latlon": home_locations_latlon,
        "home_locations_rel": home_rel,
        "mp_locations_latlon": mp_locations_latlon,
        "mp_locations_rel": mp_rel,
        "all_coords_rel": all_coords_rel,
        "all_coords_latlon": ([(depot_lon, depot_lat)] + home_locations_latlon +
                               mp_locations_latlon),
        "adjacency": adjacency,
        "variant": variant,
        "projection": projection,
        "mp_fixed_seed": 42,
    }


def write_coords_txt(filepath: Path, data: dict):
    """写coords.txt文件，格式兼容现有load_demand_data。"""
    all_rel = data["all_coords_rel"]
    home_id = "BJ_YANJIAO_HOME"
    mp_ids = [f"BJ_YANJIAO_MP_{i:04d}" for i in range(data["n_meeting_points"])]

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("NODE_COORD_SECTION (Yanjiao-Guomao DRT)\n")
        # Depot = 国贸destination
        f.write(f"DEPOT_GUOMAO\t{all_rel[0][0]}\t{all_rel[0][1]}\n")
        # Home locations (Yanjiao residential area)
        for i in range(data["passengers"]):
            x, y = all_rel[1 + i]
            f.write(f"{home_id}\t{x}\t{y}\n")
        # Meeting points (Yanjiao bus stops)
        for i in range(data["n_meeting_points"]):
            x, y = all_rel[1 + data["passengers"] + i]
            f.write(f"{mp_ids[i]}\t{x}\t{y}\n")


def write_coords_latlon(filepath: Path, data: dict):
    """写经纬度版本coords文件 (用于可视化和地图)。"""
    all_ll = data["all_coords_latlon"]
    home_id = "BJ_YANJIAO_HOME"
    mp_ids = [f"BJ_YANJIAO_MP_{i:04d}" for i in range(data["n_meeting_points"])]

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("NODE_COORD_SECTION (Yanjiao-Guomao DRT) [WGS84 LatLon]\n")
        # Depot = 国贸
        f.write(f"DEPOT_GUOMAO\t{all_ll[0][0]:.6f}\t{all_ll[0][1]:.6f}\n")
        for i in range(data["passengers"]):
            lon, lat = all_ll[1 + i]
            f.write(f"{home_id}\t{lon:.6f}\t{lat:.6f}\n")
        for i in range(data["n_meeting_points"]):
            lon, lat = all_ll[1 + data["passengers"] + i]
            f.write(f"{mp_ids[i]}\t{lon:.6f}\t{lat:.6f}\n")


def write_metadata(filepath: Path, data: dict):
    """写元数据JSON。"""
    meta = {
        "instance": "Beijing_Yanjiao",
        "scenario": "Yanjiao-Guomao many-to-one commuter DRT",
        "variant": data.get("variant", "base"),
        "variant_description": VARIANT_SETTINGS[data.get("variant", "base")]["description"],
        "seed": data["seed"],
        "passengers": data["passengers"],
        "n_meeting_points": data["n_meeting_points"],
        "k_nearest": data.get("k_nearest", DEFAULT_K),
        "perturbation_sigma": VARIANT_SETTINGS[data.get("variant", "base")]["sigma"],
        "perturbation_max": VARIANT_SETTINGS[data.get("variant", "base")]["max_perturb"],
        "home_edge_share": VARIANT_SETTINGS[data.get("variant", "base")]["home_edge_share"],
        "mp_edge_share": VARIANT_SETTINGS[data.get("variant", "base")]["mp_edge_share"],
        "density_radius": DENSITY_RADIUS,
        "depot": {
            "name": data["depot_name"],
            "lon": data["depot_latlon"][0],
            "lat": data["depot_latlon"][1],
        },
        "coordinate_system": projection_coordinate_system(data.get("projection", "relative")),
        "projection": data.get("projection", "relative"),
        "bounds": data["bounds"],
        "data_sources": {
            "residential_pois": "Amap POI 2.0 (120300/120302/120303/120304)",
            "bus_stops": "Amap POI 2.0 (150700)",
            "key_stops": "Customized commuter bus + 818 route keyword search",
        },
        "generation_method": (
            "Uniform random sampling within bus-stop bounding box"
            if data.get("variant", "base") == "dispersed"
            else "Residential POI density-weighted sampling + Gaussian perturbation"
        ),
        "mp_fixed_seed": data.get("mp_fixed_seed", None),
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate Beijing_Yanjiao DRT instance data")
    parser.add_argument("--passengers", type=int, default=DEFAULT_PASSENGERS,
                        help=f"Number of home locations (default: {DEFAULT_PASSENGERS})")
    parser.add_argument("--mp", type=int, default=DEFAULT_MEETING_POINTS,
                        help=f"Number of meeting points (default: {DEFAULT_MEETING_POINTS})")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                        help="Seeds to generate (default: 0 1)")
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help=f"K-nearest for adjacency matrix (default: {DEFAULT_K})")
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA,
                        help=f"Perturbation sigma in degrees (default: {DEFAULT_SIGMA})")
    parser.add_argument("--variant", choices=sorted(VARIANT_SETTINGS), default="base",
                        help="Yanjiao instance variant. Non-base variants write distinct prefixes.")
    parser.add_argument("--prefix", default=None,
                        help="Optional output prefix template with {passengers}, {seed}, and {variant}.")
    parser.add_argument("--projection", choices=["relative", "metric"], default="relative",
                        help="Coordinate projection for runtime coords. 'metric' preserves local km aspect ratio.")
    parser.add_argument("--all-scales", action="store_true",
                        help="Generate all 4 sensitivity scales (150/300/400/490)")
    args = parser.parse_args()

    # --- 加载数据 ---
    print("Loading data...")
    res_file = DATA_DIR / "residential_pois.csv"
    bus_file = DATA_DIR / "bus_stops.csv"
    key_file = DATA_DIR / "key_bus_stops.csv"

    if not res_file.exists():
        print(f"ERROR: Residential POI file not found: {res_file}")
        print("Run fetch_yanjiao_full_data.py first.")
        sys.exit(1)

    residential_pois = load_csv(res_file)
    bus_stops = load_csv(bus_file) if bus_file.exists() else []
    key_stops = load_csv(key_file) if key_file.exists() else []

    print(f"  Residential POIs: {len(residential_pois)}")
    print(f"  Bus stops: {len(bus_stops)}")
    print(f"  Key stops: {len(key_stops)}")

    # --- 生成 ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    scales = []
    if args.all_scales:
        scales = [
            (150, args.mp, "low"),
            (300, args.mp, "main"),
            (400, args.mp, "high"),
            (min(490, len(residential_pois)), args.mp, "full"),
        ]
    else:
        scales = [(args.passengers, args.mp, "custom")]

    for n_pass, n_mp, label in scales:
        for seed in args.seeds:
            print(f"\nGenerating: {n_pass} passengers, {n_mp} MPs, seed={seed} ({label})")

            data = generate_instance(
                passengers=n_pass,
                n_mp=n_mp,
                seed=seed,
                residential_pois=residential_pois,
                bus_stops=bus_stops,
                key_stops=key_stops,
                destination=GUOMAO_DESTINATION,
                variant=args.variant,
                k=args.k,
                projection=args.projection,
            )

            prefix_template = args.prefix or VARIANT_SETTINGS[args.variant]["prefix"]
            prefix = prefix_template.format(
                passengers=n_pass,
                n_passengers=n_pass,
                seed=seed,
                variant=args.variant,
            )

            # 相对坐标 (实验用)
            coords_file = OUT_DIR / f"{prefix}_coords.txt"
            write_coords_txt(coords_file, data)
            print(f"  -> {coords_file}")

            # 经纬度 (地图用)
            ll_file = OUT_DIR / f"{prefix}_coords_latlon.txt"
            write_coords_latlon(ll_file, data)
            print(f"  -> {ll_file}")

            # 邻接矩阵
            adj_file = OUT_DIR / f"{prefix}_adjacency{args.k}.npy"
            np.save(adj_file, data["adjacency"])
            print(f"  -> {adj_file} (shape: {data['adjacency'].shape})")

            # 元数据
            meta_file = OUT_DIR / f"{prefix}_metadata.json"
            write_metadata(meta_file, data)
            print(f"  -> {meta_file}")

    print(f"\nDone. Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
