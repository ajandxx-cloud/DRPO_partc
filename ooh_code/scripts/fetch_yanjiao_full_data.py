"""
Step 2: 完整数据采集 — 燕郊住宅POI + 公交站POI + 关键字站点验证。

通过将燕郊区域划分为子网格绕过 Amap API 225条/查询的分页限制，
获取尽可能完整的数据集。
"""
import requests
import time
import json
import csv
import sys
from pathlib import Path

API_KEY = "bddf037ee91c85105519ff28b31ba0b5"
BASE_URL = "https://restapi.amap.com/v3/place/polygon"

# 燕郊住宅区 — 划分为4个子区域以绕过225条分页限制
# 主矩形: (116.78,39.90) 到 (117.05,39.98)
SUB_REGIONS_YANJIAO = [
    ("燕郊-NW", "116.78,39.94|116.91,39.98"),
    ("燕郊-NE", "116.91,39.94|117.05,39.98"),
    ("燕郊-SW", "116.78,39.90|116.91,39.94"),
    ("燕郊-SE", "116.91,39.90|117.05,39.94"),
]

# 国贸区域
GUOMAO_POLYGON = "116.44,39.89|116.47,39.92"

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

# 关键站点（用于关键字搜索验证）
KEY_STOPS_YANJIAO = [
    "天洋城", "燕灵路口", "兴达广场", "迎宾路口", "行宫花园",
    "上上城", "潮白人家", "燕郊学院街", "中赵甫村", "星月云河",
    "紫竹园", "忆江南", "盛恒时代", "御东瑞璟", "金谷爱舒荷",
]
KEY_STOPS_GUOMAO = ["郎家园", "大北窑", "大北窑南", "招商局", "国贸"]

OUT_DIR = Path(__file__).parent.parent / "data" / "yanjiao"


def search_poi_polygon(polygon: str, types: str, page: int = 1) -> list[dict]:
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
    """翻页获取某个子区域的全部POI。"""
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


def search_by_keyword(keyword: str, city: str = "131000") -> list[dict]:
    """使用关键字搜索API验证特定站点。"""
    url = "https://restapi.amap.com/v3/place/text"
    params = {
        "key": API_KEY,
        "keywords": keyword,
        "city": city,
        "citylimit": "true",
        "extensions": "all",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "1":
        return []
    return data.get("pois", [])


def save_pois_to_csv(pois: list[dict], filepath: Path, fields: list[str]):
    """保存POI数据到CSV，按经纬度去重。"""
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
    print(f"  Saved {len(unique_pois)} unique POIs (removed {len(pois) - len(unique_pois)} duplicates) to {filepath}")


def main():
    print("=" * 60)
    print("Step 2: 完整数据采集")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_fields = ["id", "name", "location", "type", "typecode", "address", "adcode",
                  "pname", "cityname", "adname", "business_area"]

    # --- 1. 住宅POI采集（分区域） ---
    print("\n[1] 住宅POI采集（4个子区域）")
    all_residential = []
    for typecode, typename in RESIDENTIAL_TYPES:
        for region_name, polygon in SUB_REGIONS_YANJIAO:
            print(f"\n  {typename} / {region_name}")
            pois = fetch_all_in_region(region_name, polygon, typecode)
            all_residential.extend(pois)
            time.sleep(0.3)

    res_file = OUT_DIR / "residential_pois.csv"
    save_pois_to_csv(all_residential, res_file, csv_fields)

    # --- 2. 公交站POI采集（分区域） ---
    print("\n[2] 公交站POI采集（4个子区域）")
    all_bus_stops = []
    for typecode, typename in BUS_STOP_TYPES:
        for region_name, polygon in SUB_REGIONS_YANJIAO:
            print(f"\n  {typename} / {region_name}")
            pois = fetch_all_in_region(region_name, polygon, typecode)
            all_bus_stops.extend(pois)
            time.sleep(0.3)

    bus_file = OUT_DIR / "bus_stops.csv"
    save_pois_to_csv(all_bus_stops, bus_file, csv_fields)

    # --- 3. 国贸公交站 ---
    print("\n[3] 国贸区域公交站")
    guomao_bus = []
    for typecode, typename in BUS_STOP_TYPES:
        print(f"  {typename}")
        pois = fetch_all_in_region("国贸", GUOMAO_POLYGON, typecode)
        guomao_bus.extend(pois)

    guomao_file = OUT_DIR / "guomao_bus_stops.csv"
    save_pois_to_csv(guomao_bus, guomao_file, csv_fields)

    # --- 4. 关键字站点验证 ---
    print("\n[4] 关键字站点验证")
    key_stops_all = []
    for kw in KEY_STOPS_YANJIAO:
        print(f"  搜索: {kw}")
        pois = search_by_keyword(kw, city="131000")  # 廊坊
        for p in pois:
            p["search_keyword"] = kw
        key_stops_all.extend(pois)
        time.sleep(0.2)

    for kw in KEY_STOPS_GUOMAO:
        print(f"  搜索: {kw}")
        pois = search_by_keyword(kw, city="110000")  # 北京
        for p in pois:
            p["search_keyword"] = kw
        key_stops_all.extend(pois)
        time.sleep(0.2)

    key_file = OUT_DIR / "key_bus_stops.csv"
    key_fields = csv_fields + ["search_keyword"]
    seen_keys = set()
    unique_keys = []
    for p in key_stops_all:
        k = (p.get("location", ""), p.get("name", ""))
        if k not in seen_keys:
            seen_keys.add(k)
            unique_keys.append(p)
    with open(key_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=key_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(unique_keys)
    print(f"  Saved {len(unique_keys)} unique key bus stops to {key_file}")

    # --- 汇总 ---
    print("\n" + "=" * 60)
    print("采集完成汇总")
    print(f"  住宅POI (去重):  {len(set((p['location'], p['name']) for p in all_residential))}")
    print(f"  公交站 (去重):   {len(set((p['location'], p['name']) for p in all_bus_stops))}")
    print(f"  国贸公交站:      {len(guomao_bus)}")
    print(f"  关键字站点:      {len(unique_keys)}")
    print(f"  数据目录:        {OUT_DIR}")


if __name__ == "__main__":
    main()
