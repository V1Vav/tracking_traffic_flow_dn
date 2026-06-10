from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

Point = Tuple[float, float]
Region = Dict[str, Any]


def point_in_poly(x: float, y: float, poly: Sequence[Point]) -> bool:
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi)
        if intersect:
            inside = not inside
        j = i
    return inside


def parse_points_from_row(row: Dict[str, str]) -> List[Point]:
    # Supported formats:
    # name,x1,y1,x2,y2,... OR name,points="[[x,y],...]" OR separate numeric columns.
    if "points" in row and row["points"]:
        try:
            pts = json.loads(row["points"])
            return [(float(p[0]), float(p[1])) for p in pts]
        except Exception:
            pass
    numeric = []
    for k, v in row.items():
        if k.lower() in {"name", "region", "id", "label", "class"}:
            continue
        try:
            numeric.append(float(v))
        except Exception:
            pass
    pts = []
    for i in range(0, len(numeric) - 1, 2):
        pts.append((numeric[i], numeric[i + 1]))
    return pts


def load_regions(path: str | Path | None, frame_w: int = 1280, frame_h: int = 720) -> List[Region]:
    if not path:
        return default_regions(frame_w, frame_h)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    regions: List[Region] = []
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            items = data.get("regions", data.get("lanes", data))
            if isinstance(items, dict):
                items = [{"name": k, "points": v} for k, v in items.items()]
        else:
            items = data
        for item in items:
            name = str(item.get("name", item.get("id", item.get("region", f"region_{len(regions)}"))))
            pts_raw = item.get("points", item.get("polygon", item.get("coords", [])))
            pts = [(float(a), float(b)) for a, b in pts_raw]
            if len(pts) >= 3:
                regions.append({"name": name, "points": pts})
    else:
        with p.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                for row in reader:
                    name = row.get("name") or row.get("region") or row.get("id") or f"region_{len(regions)}"
                    pts = parse_points_from_row(row)
                    if len(pts) >= 3:
                        regions.append({"name": str(name), "points": pts})
            else:
                # Fallback for no header: name,x1,y1,x2,y2...
                f.seek(0)
                raw = csv.reader(f)
                for row in raw:
                    if len(row) < 7:
                        continue
                    name = row[0]
                    nums = [float(x) for x in row[1:] if x.strip()]
                    pts = [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
                    regions.append({"name": name, "points": pts})
    return regions or default_regions(frame_w, frame_h)


def default_regions(w: int, h: int) -> List[Region]:
    # Approximate 8 lanes + center. Used only when no template is provided.
    cx1, cy1, cx2, cy2 = w * 0.36, h * 0.30, w * 0.64, h * 0.70
    return [
        {"name": "t1", "points": [(cx1, 0), (w * 0.50, 0), (w * 0.50, cy1), (cx1, cy1)]},
        {"name": "t2", "points": [(w * 0.50, 0), (cx2, 0), (cx2, cy1), (w * 0.50, cy1)]},
        {"name": "b1", "points": [(w * 0.50, cy2), (cx2, cy2), (cx2, h), (w * 0.50, h)]},
        {"name": "b2", "points": [(cx1, cy2), (w * 0.50, cy2), (w * 0.50, h), (cx1, h)]},
        {"name": "l1", "points": [(0, cy1), (cx1, cy1), (cx1, h * 0.50), (0, h * 0.50)]},
        {"name": "l2", "points": [(0, h * 0.50), (cx1, h * 0.50), (cx1, cy2), (0, cy2)]},
        {"name": "r1", "points": [(cx2, h * 0.50), (w, h * 0.50), (w, cy2), (cx2, cy2)]},
        {"name": "r2", "points": [(cx2, cy1), (w, cy1), (w, h * 0.50), (cx2, h * 0.50)]},
        {"name": "center", "points": [(cx1, cy1), (cx2, cy1), (cx2, cy2), (cx1, cy2)]},
    ]


def classify_region(cx: float, cy: float, regions: Sequence[Region]) -> Optional[str]:
    # center has lower priority: if point is also in a lane, return lane first.
    center_hit: Optional[str] = None
    for r in regions:
        name = str(r["name"])
        if point_in_poly(cx, cy, r["points"]):
            if name.lower() == "center":
                center_hit = name
            else:
                return name
    return center_hit


def bbox_center(row: Dict[str, Any]) -> Tuple[float, float]:
    x1, y1, x2, y2 = float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0
