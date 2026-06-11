"""
Step 1: 走廊全段数据采集 — 国贸→通州→燕郊的公交站POI + 住宅POI。

将搜索范围从燕郊扩展到整条通勤走廊，使用子网格绕过 Amap 225条/查询的分页限制。
"""
import requests
import time
import csv
import sys
from pathlib import Path

API_KEY = "bddf037ee91c85105519ff28b31ba0b5"
BASE_URL = "https://restapi.amap.com/v3/place/polygon"

# 走廊全段: 国贸(116.46) → 通州(116.65) → 燕郊(117.05)
# 按经度分成6个子区域，纬度分为2层 → 共12个子网格
LON_BANDS = [
    ("CBD-通州西", "116.46,39.88|116.55,39.95"),
    ("通州西-通州中", "116.55,39.88|116.65,39.96"),
    ("通州中-通州东", "116.65,39.88|116.75,39.97"),
    ("通州东-燕郊西", "116.75,39.88|116.85,39.98"),
    ("燕郊中", "116.85,39.92|116.95,39.98"),
    ("燕郊东", "116.95,39.92|117.06,39.98"),
]

# 住宅POI类型
RESIDENTIAL_TYPES = [
    ("120300", "住宅区大类"),
    ("120302", "住宅小区"),
    ("120303", "宿舍"),
    ("120304", "社区中心"),
]

# 公交站POI类型
BUS_STOP_TYPES = [
    ("150700", "公交车站相关"),
    ("150702", "普通公交站"),
]

OUT_DIR = Path(__file__).parent.parent / "data" / "corridor"


def search_poi_polygon(polygon: str, types: str, page: int = 1):
    params = {
        "key": API_KEY,
        "polygon": polygon,
        "types": types,
        "offset": 25,
        "page": page,
        "extensions": "all",
    }
    resp = requests.get(BASE_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "1":
        raise RuntimeError(f"API error: {data.get('info', 'unknown')}")
    return data.get("pois", []), int(data.get("count", 0))


def fetch_all_in_region(region_name: str, polygon: str, types: str) -> list[dict]:
    all_pois = []
    page = 1
    while True:
        try:
            pois, total = search_poi_polygon(polygon, types, page=page)
        except Exception as e:
            print(f"    [{region_name}] page={page} ERROR: {e}")
            break
        if not pois:
            break
        all_pois.extend(pois)
        print(f"    [{region_name}] page={page}: +{len(pois)}, total={len(all_pois)}")
        if len(pois) < 25 or len(all_pois) >= total:
            break
        page += 1
        time.sleep(0.25)
    return all_pois


def save_pois_to_csv(pois: list[dict], filepath: Path, fields: list[str]):
    seen = set()
    unique_pois = []
    for p in pois:
        loc = p.get("location", "")
        name = p.get("name", "")
        key = (loc, name)
        if key not in seen:
            seen.add(key)
            unique_pois.append(p)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(unique_pois)
    print(f"  Saved {len(unique_pois)} unique POIs to {filepath}")
    return unique_pois


def main():
    print("=" * 60)
    print("走廊全段数据采集: 国贸→通州→燕郊")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_fields = ["id", "name", "location", "type", "typecode", "address",
                  "adcode", "pname", "cityname", "adname", "business_area"]

    # --- 1. 住宅POI ---
    print("\n[1] 住宅POI采集（走廊全段）")
    all_residential = []
    for typecode, typename in RESIDENTIAL_TYPES:
        for region_name, polygon in LON_BANDS:
            print(f"\n  {typename} / {region_name}")
            pois = fetch_all_in_region(region_name, polygon, typecode)
            all_residential.extend(pois)
            time.sleep(0.3)
    res_file = OUT_DIR / "corridor_residential_pois.csv"
    save_pois_to_csv(all_residential, res_file, csv_fields)

    # --- 2. 公交站POI ---
    print("\n[2] 公交站POI采集（走廊全段）")
    all_bus_stops = []
    for typecode, typename in BUS_STOP_TYPES:
        for region_name, polygon in LON_BANDS:
            print(f"\n  {typename} / {region_name}")
            pois = fetch_all_in_region(region_name, polygon, typecode)
            all_bus_stops.extend(pois)
            time.sleep(0.3)
    bus_file = OUT_DIR / "corridor_bus_stops.csv"
    save_pois_to_csv(all_bus_stops, bus_file, csv_fields)

    # --- 汇总 ---
    print("\n" + "=" * 60)
    print("采集完成汇总")
    print(f"  住宅POI (去重):  {len(set((p['location'], p['name']) for p in all_residential))}")
    print(f"  公交站 (去重):   {len(set((p['location'], p['name']) for p in all_bus_stops))}")
    print(f"  数据目录:        {OUT_DIR}")


if __name__ == "__main__":
    main()
