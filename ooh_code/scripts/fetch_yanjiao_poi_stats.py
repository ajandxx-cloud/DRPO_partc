"""
Step 1: 探测燕郊区域POI规模，确定数据生成参数。

使用高德POI 2.0 polygon搜索API，获取燕郊住宅区内：
  - 住宅POI（120300/120302/120303/120304）
  - 公交站POI（150700/150702）

输出：各类POI数量 → 据此确定 home locations 和 meeting points 规模。
"""
import requests
import time
import json
import sys
from pathlib import Path

API_KEY = "bddf037ee91c85105519ff28b31ba0b5"
BASE_URL = "https://restapi.amap.com/v3/place/polygon"

# 燕郊住宅区 polygon（矩形近似）: 燕顺路→迎宾路→行宫大街→天洋城一带
# 矩形: (116.78,39.90) 到 (117.05,39.98)
YANJIAO_POLYGON = "116.78,39.90|117.05,39.98"

# 国贸区域（用于目的地和北京侧上车点参考）
GUOMAO_POLYGON = "116.44,39.89|116.47,39.92"

# 住宅相关POI类型
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

# 定制快巴/818路关键站点名称（用于关键字搜索验证）
KEY_BUS_STOPS = [
    "天洋城", "燕灵路口", "兴达广场", "迎宾路口", "行宫花园",
    "上上城", "潮白人家", "燕郊学院街", "郎家园", "大北窑",
    "中赵甫村", "星月云河", "紫竹园", "忆江南", "盛恒时代",
    "御东瑞璟", "金谷爱舒荷", "建外",
]


def search_poi_polygon(polygon: str, types: str, page: int = 1, offset: int = 25) -> dict:
    """调用高德POI 2.0 polygon搜索API。"""
    params = {
        "key": API_KEY,
        "polygon": polygon,
        "types": types,
        "offset": offset,
        "page": page,
        "extensions": "all",
    }
    resp = requests.get(BASE_URL, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def count_all_pages(polygon: str, types: str, label: str) -> list[dict]:
    """翻页获取全部结果，返回所有POI列表。"""
    all_pois = []
    page = 1
    while True:
        try:
            data = search_poi_polygon(polygon, types, page=page)
        except Exception as e:
            print(f"  [ERROR] page={page}: {e}")
            break

        if data.get("status") != "1":
            print(f"  [ERROR] API error: {data.get('info', 'unknown')}")
            break

        pois = data.get("pois", [])
        if not pois:
            break

        all_pois.extend(pois)
        total = int(data.get("count", 0))
        print(f"  {label} page={page}: fetched {len(pois)}, cumulative={len(all_pois)}, total_available={total}")

        if len(all_pois) >= total or len(pois) < 25:
            break

        page += 1
        time.sleep(0.3)  # 避免QPS限流

    return all_pois


def main():
    print("=" * 60)
    print("Step 1: 燕郊POI规模探测")
    print("=" * 60)

    all_stats = {}

    # --- 住宅POI ---
    print("\n[1] 住宅POI探测 (燕郊范围)")
    residential_pois = {}
    for typecode, typename in RESIDENTIAL_TYPES:
        print(f"\n  查询: {typename} (typecode={typecode})")
        pois = count_all_pages(YANJIAO_POLYGON, typecode, typename)
        residential_pois[typecode] = pois
        print(f"  => {typename}: {len(pois)} 条")

    total_residential = sum(len(v) for v in residential_pois.values())
    print(f"\n  住宅POI总计: {total_residential}")

    # --- 公交站POI ---
    print("\n[2] 公交站POI探测 (燕郊范围)")
    bus_stop_pois = {}
    for typecode, typename in BUS_STOP_TYPES:
        print(f"\n  查询: {typename} (typecode={typecode})")
        pois = count_all_pages(YANJIAO_POLYGON, typecode, typename)
        bus_stop_pois[typecode] = pois
        print(f"  => {typename}: {len(pois)} 条")

    total_bus_stops = sum(len(v) for v in bus_stop_pois.values())
    print(f"\n  公交站POI总计: {total_bus_stops}")

    # --- 国贸公交站参考 ---
    print("\n[3] 国贸区域公交站 (destination参考)")
    guomao_pois = {}
    for typecode, typename in BUS_STOP_TYPES:
        pois = count_all_pages(GUOMAO_POLYGON, typecode, typename)
        guomao_pois[typecode] = pois
        print(f"  {typename}: {len(pois)} 条")

    # --- 汇总报告 ---
    print("\n" + "=" * 60)
    print("探测结果汇总")
    print("=" * 60)
    print(f"  燕郊住宅POI总数:    {total_residential}")
    print(f"  燕郊公交站POI总数:   {total_bus_stops}")
    print(f"  国贸区域公交站:      {sum(len(v) for v in guomao_pois.values())}")

    # --- 规模建议 ---
    print("\n[4] 数据规模建议")
    # meeting points 取公交站的50%-70%（去重、筛选后）
    mp_recommended = min(total_bus_stops, max(80, int(total_bus_stops * 0.6)))
    # home locations 为 meeting points 的 2-3 倍
    home_recommended_low = mp_recommended * 2
    home_recommended_high = min(total_residential, mp_recommended * 3)

    print(f"  推荐 meeting points: {mp_recommended} (公交站去重筛选后)")
    print(f"  推荐 home locations: {home_recommended_low} ~ {home_recommended_high}")
    print(f"  (对比: RC=90人/10点, Beijing_bus=240人/169点)")

    # 敏感性分析场景
    print(f"\n  敏感性分析场景:")
    print(f"    Low demand:    ~{max(60, home_recommended_low // 2)}人 / ~{mp_recommended}点")
    print(f"    Main case:     ~{home_recommended_low}人 / ~{mp_recommended}点")
    print(f"    High demand:   ~{home_recommended_high}人 / ~{mp_recommended}点")
    print(f"    Full scale:    ~{min(total_residential, home_recommended_high * 2)}人 / ~{mp_recommended}点")

    # 保存统计结果
    stats = {
        "yanjiao_residential_poi_count": total_residential,
        "yanjiao_bus_stop_count": total_bus_stops,
        "guomao_bus_stop_count": sum(len(v) for v in guomao_pois.values()),
        "recommended_meeting_points": mp_recommended,
        "recommended_home_locations_low": home_recommended_low,
        "recommended_home_locations_high": home_recommended_high,
        "residential_breakdown": {k: len(v) for k, v in residential_pois.items()},
        "bus_stop_breakdown": {k: len(v) for k, v in bus_stop_pois.items()},
        "search_polygon_yanjiao": YANJIAO_POLYGON,
        "search_polygon_guomao": GUOMAO_POLYGON,
    }

    out_dir = Path(__file__).parent.parent / "data" / "yanjiao"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_file = out_dir / "poi_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n  统计结果已保存至: {stats_file}")


if __name__ == "__main__":
    main()
