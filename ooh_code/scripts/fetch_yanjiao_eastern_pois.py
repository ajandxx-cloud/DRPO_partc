"""Supplement residential POIs in the eastern gap area (lon 117.00~117.06).

The original fetch already covers up to 117.05, but only found 2 residential
POIs east of 117.00. This script tries additional POI types and keyword
searches to find more residential anchors in the Dachang/Xiadian area.
"""
import csv
import time
from pathlib import Path

import requests

API_KEY = "bddf037ee91c85105519ff28b31ba0b5"
POLYGON_URL = "https://restapi.amap.com/v3/place/polygon"
TEXT_URL = "https://restapi.amap.com/v3/place/text"

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "yanjiao"

# Eastern gap sub-regions (split lat to avoid page limits)
EAST_REGIONS = [
    ("东-N", "117.00,39.94|117.07,39.98"),
    ("东-M", "117.00,39.90|117.07,39.94"),
]

# Extra POI types beyond residential (120300-120304)
EXTRA_TYPES = [
    ("120300", "住宅区（重试东部）"),
    ("190301", "村庄"),
    ("190302", "乡镇"),
]

# Keyword searches for known eastern communities
KEYWORDS_EAST = [
    "夏垫", "段甲岭", "黄土庄", "三河",
    "大厂", "邵府", "陈府", "祁各庄",
    "谭台", "小厂", "亮甲台",
]


def fetch_polygon(polygon: str, types: str, page: int = 1) -> tuple[list[dict], int]:
    params = {
        "key": API_KEY,
        "polygon": polygon,
        "types": types,
        "offset": 25,
        "page": page,
        "extensions": "all",
    }
    resp = requests.get(POLYGON_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "1":
        raise RuntimeError(f"API error: {data.get('info')}")
    return data.get("pois", []), int(data.get("count", 0))


def fetch_keyword(keyword: str, city: str = "131000") -> list[dict]:
    params = {
        "key": API_KEY,
        "keywords": keyword,
        "city": city,
        "citylimit": "true",
        "extensions": "all",
    }
    resp = requests.get(TEXT_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "1":
        return []
    return data.get("pois", [])


def main() -> None:
    all_new: list[dict] = []
    csv_fields = [
        "id", "name", "location", "type", "typecode", "address",
        "adcode", "pname", "cityname", "adname", "business_area",
    ]

    # 1. Polygon search with extra types
    for typecode, typename in EXTRA_TYPES:
        for region_name, polygon in EAST_REGIONS:
            print(f"  {typename} / {region_name}")
            page = 1
            while True:
                try:
                    pois, total = fetch_polygon(polygon, typecode, page)
                except Exception as e:
                    print(f"    ERROR page={page}: {e}")
                    break
                if not pois:
                    break
                # Only keep POIs with lon >= 117.00
                for p in pois:
                    loc = p.get("location", "")
                    if "," in loc:
                        lon = float(loc.split(",")[0])
                        if lon >= 117.00:
                            all_new.append(p)
                print(f"    page={page}: +{len(pois)} fetched, cumul east={len(all_new)}")
                if len(pois) < 25 or page * 25 >= total:
                    break
                page += 1
                time.sleep(0.25)
            time.sleep(0.3)

    # 2. Keyword search
    print("\n  Keyword searches:")
    for kw in KEYWORDS_EAST:
        print(f"    {kw}")
        pois = fetch_keyword(kw)
        for p in pois:
            loc = p.get("location", "")
            if "," in loc:
                lon = float(loc.split(",")[0])
                if 117.00 <= lon <= 117.07:
                    all_new.append(p)
        time.sleep(0.2)

    # 3. Deduplicate
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for p in all_new:
        key = (p.get("location", ""), p.get("name", ""))
        if key not in seen:
            seen.add(key)
            unique.append(p)

    print(f"\n  Total new POIs (deduped): {len(unique)}")
    for p in unique:
        loc = p.get("location", "")
        print(f"    {loc}  {p.get('name', '')}  [{p.get('typecode', '')}]")

    # 4. Append to existing CSV (avoid duplicates with existing entries)
    existing_keys: set[tuple[str, str]] = set()
    res_file = DATA_DIR / "residential_pois.csv"
    if res_file.exists():
        with res_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_keys.add((row.get("location", ""), row.get("name", "")))

    truly_new = [p for p in unique if (p.get("location", ""), p.get("name", "")) not in existing_keys]
    print(f"  Truly new (not in existing CSV): {len(truly_new)}")

    if truly_new:
        # Read existing rows
        existing_rows: list[dict] = []
        if res_file.exists():
            with res_file.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_rows.append(row)

        # Append
        with res_file.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            for p in truly_new:
                writer.writerow(p)
        print(f"  Appended {len(truly_new)} POIs to {res_file}")
    else:
        print("  No new POIs to add.")


if __name__ == "__main__":
    main()
