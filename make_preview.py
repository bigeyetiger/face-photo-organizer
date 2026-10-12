# -*- coding: utf-8 -*-
"""从 results.csv 重建 preview.html（流式写盘，内存友好）。"""
import base64
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

WORKDIR = Path(__file__).parent
THUMB_SIDE = 240
QUALITY = 65
MAX_PER_PERSON = 600


def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def to_b64(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def main():
    rows = list(csv.reader(open(WORKDIR / "results.csv", encoding="utf-8-sig")))[1:]
    by_person = {}
    for r in rows:
        if r[4] in ("已复制", "仅记录"):
            by_person.setdefault(r[2], []).append(r)

    parts = ["""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>识别结果核对</title><style>
body{font-family:"Microsoft YaHei",sans-serif;background:#fafafa;margin:24px}
h2{margin:28px 0 8px;border-left:4px solid #1976d2;padding-left:8px}
.row{display:flex;flex-wrap:wrap;gap:10px}
figure{margin:0;width:160px;background:#fff;border:3px solid #ddd;border-radius:8px;padding:6px;text-align:center}
figure img{width:100%;border-radius:4px}
figcaption{font-size:13px;margin-top:4px;line-height:1.4;word-break:break-all}
small{color:#888;font-weight:normal}
</style></head><body><h1>识别结果核对</h1>"""]

    for name, items in sorted(by_person.items()):
        items.sort(key=lambda r: -float(r[3]))
        parts.append(f"<h2>{name}（{len(items)} 张）</h2><div class='row'>")
        n = 0
        for r in items:
            if n >= MAX_PER_PERSON:
                break
            img = imread_unicode(r[0])
            if img is None:
                continue
            h, w = img.shape[:2]
            sc = THUMB_SIDE / max(h, w)
            thumb = cv2.resize(img, (int(w * sc), int(h * sc))) if sc < 1 else img
            s = float(r[3])
            color = "#2e7d32" if s >= 0.55 else ("#f9a825" if s >= 0.40 else "#9e9e9e")
            label = "确定" if s >= 0.55 else ("待复核" if s >= 0.40 else "边缘")
            figure = (f"<figure style='border-color:{color}'>"
                      f"<img src='data:image/jpeg;base64,{to_b64(thumb)}'>"
                      f"<figcaption style='color:{color}'>{s:.2f} {label}<br>"
                      f"<small>{r[1]}</small></figcaption></figure>")
            parts.append(figure)
            n += 1
        parts.append("</div>")
    parts.append("<p style='color:#aaa'>相似度≥0.55 确定 / 0.40–0.55 待复核 / 0.20–0.40 边缘（仅记录未复制）</p></body></html>")

    out = WORKDIR / "preview.html"
    out.write_text("".join(parts), encoding="utf-8")
    print(f"已生成 {out}（{out.stat().st_size/1e6:.1f} MB）", flush=True)


if __name__ == "__main__":
    main()
