# -*- coding: utf-8 -*-
"""
按人物筛选照片（本地离线，照片不上传）
用法：
  1) 建人物档案：python find_person.py build --ref F:/PURA70/sample
  2) 扫描归档：  python find_person.py scan --src F:/PURA70 --out F:/PURA70_按人整理
结果：每人一个文件夹 + results.csv + preview.html（浏览器打开核对）
"""
import argparse
import base64
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DET_SIZE = (640, 640)
MAX_SIDE = 1600      # 检测前把长边缩到该值（提速，精度足够）
THUMB_SIDE = 280     # 预览页缩略图长边
SIM_CLUSTER = 0.45   # 参考照聚类：≥此值判为同一人
SIM_COPY = 0.40      # 扫描：≥此值复制归档
SIM_LOG = 0.20       # 扫描：≥此值写入报告供人工复核


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def imread_unicode(path):
    """Windows 中文路径安全的读图。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img, params=None):
    ext = Path(path).suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img, params or [])
    if ok:
        buf.tofile(str(path))
    return ok


def make_app():
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=DET_SIZE)
    return app


def load_and_detect(app, path):
    """读图（长边≤1600）并检测人脸。返回 (图像, 人脸列表)，读图失败返回 (None, [])。"""
    img = imread_unicode(path)
    if img is None:
        return None, []
    h, w = img.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img, app.get(img)


# ---------------- build 模式：从参考照建人物档案 ----------------

def cmd_build(args):
    ref_dir = Path(args.ref)
    files = sorted(p for p in ref_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    if not files:
        sys.exit(f"参考目录里没有图片：{ref_dir}")
    app = make_app()

    faces_all = []  # (file, face_idx, face)
    for p in files:
        img, faces = load_and_detect(app, p)
        if img is not None:
            _REF_IMG_CACHE[p.name] = img  # 缓存缩放后原图，bbox 与其同坐标系
        log(f"参考照 {p.name}: 检出 {len(faces)} 张脸")
        for j, f in enumerate(faces):
            faces_all.append((p.name, j, f))

    # 贪心聚类：与已有档案最大相似度≥阈值则并入，否则新建档案
    persons = []  # dict(name, emb, files=参考文件名集合, keys=(文件名,脸序号)集合)
    for fname, fidx, face in faces_all:
        emb = face.normed_embedding
        best_i, best_s = -1, -1.0
        for i, per in enumerate(persons):
            s = float(np.max(per["emb"] @ emb))
            if s > best_s:
                best_i, best_s = i, s
        if best_s >= SIM_CLUSTER:
            persons[best_i]["emb"] = np.vstack([persons[best_i]["emb"], emb])
            persons[best_i]["files"].add(fname)
            persons[best_i]["keys"].add((fname, fidx))
        else:
            persons.append({"emb": emb[None, :], "files": {fname},
                            "keys": {(fname, fidx)}})

    persons.sort(key=lambda x: -len(x["emb"]))
    for i, per in enumerate(persons):
        per["name"] = f"人物{i+1}"

    out = {
        "ref_dir": str(ref_dir),
        "persons": [
            {"name": per["name"], "files": sorted(per["files"]),
             "emb": per["emb"].mean(axis=0).tolist()}
            for per in persons
        ],
    }
    ref_json = Path(args.workdir) / "refs.json"
    ref_json.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"共 {len(faces_all)} 张参考脸 → 聚类为 {len(persons)} 个人物，档案存 {ref_json}")
    for per in persons:
        log(f"  {per['name']}: 参考照 {sorted(per['files'])}")
    build_refs_preview(args.workdir, persons, faces_all)
    log("参考确认页已生成，请打开 refs_preview.html 核对每个人物是谁")


def build_refs_preview(workdir, persons, faces_all):
    """把每个档案的参考脸截图拼成确认页（原图来自 _REF_IMG_CACHE）。"""
    rows = []
    for per in persons:
        cells = []
        for fname, fidx, face in faces_all:
            if (fname, fidx) not in per["keys"]:
                continue
            img = _REF_IMG_CACHE.get(fname)
            if img is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            h, w = img.shape[:2]
            pad = int(0.25 * max(x2 - x1, y2 - y1))
            crop = img[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
            if crop.size:
                save_dir = Path(workdir) / "refs_faces"
                save_dir.mkdir(exist_ok=True)
                imwrite_unicode(save_dir / f'{per["name"]}_{fname.replace(".jpg", "")}_{len(cells)+1}.jpg', crop)
                cells.append((fname, crop))
        rows.append((per, cells))

    cards = []
    for per, cells in rows:
        imgs_html = []
        for fname, crop in cells:
            b64 = to_b64(crop)
            imgs_html.append(
                f'<figure><img src="data:image/jpeg;base64,{b64}">'
                f"<figcaption>{fname}</figcaption></figure>")
        cards.append(
            f'<div class="card"><h2>{per["name"]}'
            f'<small>（{len(per["emb"])} 张参考脸）</small></h2>'
            f'<div class="row">{"".join(imgs_html)}</div></div>')
    html = HTML_TMPL.replace("__TITLE__", "人物档案确认").replace(
        "__BODY__", "".join(cards))
    (Path(workdir) / "refs_preview.html").write_text(html, encoding="utf-8")


_REF_IMG_CACHE = {}


def to_b64(img, quality=80):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


# ---------------- scan 模式：扫描照片库并归档 ----------------

def cmd_scan(args):
    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ref_json = Path(args.workdir) / "refs.json"
    refs = json.loads(ref_json.read_text(encoding="utf-8"))
    persons = [{"name": p["name"], "emb": np.array(p["emb"], dtype=np.float32)}
               for p in refs["persons"]]
    for p in persons:
        n = np.linalg.norm(p["emb"])
        p["emb"] = p["emb"] / max(n, 1e-9)
    log(f"载入 {len(persons)} 个人物档案，开始扫描 {src}")

    # 排除：参考目录、输出目录
    exclude = {Path(refs["ref_dir"]).resolve(), out.resolve()}
    files = [p for p in sorted(src.rglob("*"))
             if p.is_file() and p.suffix.lower() in IMG_EXT
             and not any(p.resolve().is_relative_to(e) for e in exclude)]
    log(f"待扫描图片 {len(files)} 张（阈值：复制≥{args.threshold}，报告≥{SIM_LOG}）")

    app = make_app()
    rows = []  # csv 行
    t0, copied = time.time(), 0
    for idx, p in enumerate(files, 1):
        img, faces = load_and_detect(app, p)
        if img is None:
            rows.append([str(p), p.name, "", "", "读图失败", 0])
            continue
        if not faces:
            rows.append([str(p), p.name, "", "", "无人脸", 0])
        hits = {}
        best_overall = 0.0
        for face in faces:
            emb = face.normed_embedding
            for per in persons:
                s = float(per["emb"] @ emb)
                if s > hits.get(per["name"], 0.0):
                    hits[per["name"]] = s
                best_overall = max(best_overall, s)
        for name, s in sorted(hits.items(), key=lambda kv: -kv[1]):
            if s >= args.threshold:
                dest_dir = out / name
                dest_dir.mkdir(exist_ok=True)
                shutil.copy2(p, dest_dir / p.name)
                copied += 1
                rows.append([str(p), p.name, name, f"{s:.3f}", "已复制", len(faces)])
            elif s >= SIM_LOG:
                rows.append([str(p), p.name, name, f"{s:.3f}", "仅记录", len(faces)])
        if idx % 50 == 0 or idx == len(files):
            rate = idx / (time.time() - t0)
            log(f"进度 {idx}/{len(files)}（{rate:.1f} 张/秒），已复制 {copied} 张")
    write_results(args.workdir, rows, persons, src)
    log(f"完成：扫描 {len(files)} 张，复制 {copied} 张 → {out}")


def write_results(workdir, rows, persons, src):
    csv_path = Path(workdir) / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["照片路径", "文件名", "人物", "相似度", "动作", "照片内人脸数"])
        w.writerows(rows)

    # 预览页：命中行带缩略图与分数
    by_person = {p["name"]: [] for p in persons}
    for r in rows:
        if r[4] in ("已复制", "仅记录"):
            by_person[r[2]].append(r)
    cards = []
    for name, items in by_person.items():
        if not items:
            continue
        items.sort(key=lambda r: -float(r[3]))
        cells = []
        for r in items[:400]:  # 防止页面过大
            img = imread_unicode(r[0])
            if img is None:
                continue
            h, w = img.shape[:2]
            scale = THUMB_SIDE / max(h, w)
            thumb = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1 else img
            s = float(r[3])
            color = "#2e7d32" if s >= 0.55 else ("#f9a825" if s >= 0.40 else "#9e9e9e")
            certainty = "确定" if s >= 0.55 else ("待复核" if s >= 0.40 else "边缘")
            cells.append(
                f'<figure class="hit" style="border-color:{color}">'
                f'<img src="data:image/jpeg;base64,{to_b64(thumb, 70)}">'
                f'<figcaption style="color:{color}">{s:.2f} {certainty}<br>'
                f'<small>{r[1]}</small></figcaption></figure>')
        cards.append(f'<h2>{name}（{len(items)} 张）</h2>'
                     f'<div class="row">{"".join(cells)}</div>')
    html = HTML_TMPL.replace("__TITLE__", "识别结果核对").replace(
        "__BODY__", "".join(cards))
    (Path(workdir) / "preview.html").write_text(html, encoding="utf-8")
    log(f"报告已生成：{csv_path} 与 preview.html")


HTML_TMPL = """<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>__TITLE__</title><style>
body{font-family:"Microsoft YaHei",sans-serif;background:#fafafa;margin:24px}
h2{margin:28px 0 8px;border-left:4px solid #1976d2;padding-left:8px}
.row{display:flex;flex-wrap:wrap;gap:10px}
figure{margin:0;width:170px;background:#fff;border:3px solid #ddd;border-radius:8px;
padding:6px;text-align:center}
figure img{width:100%;border-radius:4px}
figcaption{font-size:13px;margin-top:4px;line-height:1.4;word-break:break-all}
small{color:#888;font-weight:normal}
</style></head><body><h1>__TITLE__</h1>__BODY__
<p style="color:#aaa;margin-top:40px">本地离线生成 · 相似度≥0.55 确定 / 0.40–0.55 待复核 / 0.20–0.40 仅记录</p>
</body></html>"""


def main():
    ap = argparse.ArgumentParser(description="按人物筛选照片（InsightFace 本地识别）")
    sub = ap.add_subparsers(dest="mode", required=True)

    b = sub.add_parser("build", help="从参考照建人物档案")
    b.add_argument("--ref", required=True, help="参考照片目录")
    b.add_argument("--workdir", default=str(Path(__file__).parent))
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("scan", help="扫描照片库并归档")
    s.add_argument("--src", required=True, help="照片库目录")
    s.add_argument("--out", required=True, help="输出目录（每人一个子文件夹）")
    s.add_argument("--threshold", type=float, default=SIM_COPY, help="复制阈值，默认 0.40")
    s.add_argument("--workdir", default=str(Path(__file__).parent))
    s.set_defaults(func=cmd_scan)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
