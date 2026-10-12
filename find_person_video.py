# -*- coding: utf-8 -*-
"""
视频版：按人物筛选视频（本地离线）
策略：
  长视频 → ffmpeg 只解码关键帧（速度比全帧快约 7 倍，时间点精度放宽到秒级）
  短视频(≤15s) → OpenCV 每秒抽 1 帧全帧解码
命中判定：单帧相似度≥0.40 记为出现，累计时长≥阈值才算命中。
用法：python find_person_video.py scan --src F:/PURA70 --out E:/家人合集
结果：命中视频复制到 <out>/视频_人物N/ + videos_report.csv + videos_preview.html + hit_frames/
"""
import argparse
import base64
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

VID_EXT = {".mp4", ".mov", ".mkv", ".avi", ".wmv", ".flv", ".3gp"}
DET_SIZE = (640, 640)
THUMB_SIDE = 320
SIM_HIT = 0.40
DEFAULT_MIN_SEC = 1.0
SHORT_VIDEO_SEC = 15  # 短于该值用全帧抽帧


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def imwrite_unicode(path, img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if ok:
        buf.tofile(str(path))
    return ok


def to_b64(img, quality=70):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def fmt_ts(sec):
    sec = max(0, sec)
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def probe_duration(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=60)
    try:
        return float(r.stdout.strip())
    except (ValueError, TypeError):
        return None


def dump_keyframes(path, tmpdir):
    """ffmpeg 只解码关键帧，缩放到长边1280 存为 jpg 序列。返回文件列表。"""
    tmpdir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-skip_frame", "nokey",
         "-i", str(path), "-vf", "scale='min(1280,iw)':-2", "-fps_mode", "vfr",
         "-q:v", "3", str(tmpdir / "%05d.jpg")],
        check=True, timeout=7200)
    return sorted(tmpdir.glob("*.jpg"))


def frames_every_second(path):
    """短视频用：OpenCV 每秒抽 1 帧。yield (秒, frame)。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(int(round(fps)), 1)
    fi = 0
    while True:
        if not cap.grab():
            break
        if fi % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            h, w = frame.shape[:2]
            sc = 1280 / max(h, w)
            if sc < 1:
                frame = cv2.resize(frame, (int(w * sc), int(h * sc)))
            yield fi / fps, frame
        fi += 1
    cap.release()


def keyframe_samples(kfs, interval):
    """流式产出关键帧：yield (秒, frame)，不整体驻留内存。"""
    for i, k in enumerate(kfs):
        img = imread_unicode(k)
        if img is not None:
            yield i * interval, img


def recognize_frames(app, persons, samples):
    """samples: iterable of (sec, frame)。
    返回 hits: name -> dict(times, best_s, best_sec, best_frame)，内存中只保留每人一帧最佳画面。"""
    hits = {p["name"]: {"times": [], "best_s": -1.0, "best_sec": 0.0, "best_frame": None}
            for p in persons}
    for sec, frame in samples:
        for face in app.get(frame):
            emb = face.normed_embedding
            for per in persons:
                s = float(per["emb"] @ emb)
                if s >= SIM_HIT:
                    h = hits[per["name"]]
                    h["times"].append(sec)
                    if s > h["best_s"]:
                        h["best_s"] = s
                        h["best_sec"] = sec
                        h["best_frame"] = frame.copy()
    return hits


def cmd_scan(args):
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    workdir = Path(args.workdir)
    refs = json.loads((workdir / "refs.json").read_text(encoding="utf-8"))
    persons = [{"name": p["name"], "emb": np.array(p["emb"], dtype=np.float32)}
               for p in refs["persons"]]
    for p in persons:
        p["emb"] /= max(np.linalg.norm(p["emb"]), 1e-9)

    exclude = {out.resolve()}
    files = sorted(p for p in src.rglob("*")
                   if p.is_file() and p.suffix.lower() in VID_EXT
                   and not any(p.resolve().is_relative_to(e) for e in exclude))
    total_gb = sum(f.stat().st_size for f in files) / 1e9
    log(f"共 {len(files)} 个视频（{total_gb:.1f} GB）。命中：单帧≥{SIM_HIT} 且累计≥{args.min_sec} 秒")

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=DET_SIZE)

    kf_tmp = workdir / "kf_tmp"
    rows, thumbs = [], {}
    t0 = time.time()
    for vi, vf in enumerate(files, 1):
        dur = probe_duration(vf)
        interval = 1.0
        use_keyframe = dur and dur > SHORT_VIDEO_SEC
        kfs = []
        if use_keyframe:
            try:
                kfs = dump_keyframes(vf, kf_tmp / vf.stem)
                if kfs:
                    interval = dur / len(kfs)
                    # 关键帧太稀疏（大 GOP）说明采样不足，退回全帧
                    if interval > 2.0:
                        kfs = []
                        interval = 1.0
            except (subprocess.SubprocessError, OSError) as e:
                log(f"  关键帧抽取失败({e.__class__.__name__})，改用全帧")

        if kfs:
            samples = keyframe_samples(kfs, interval)
        else:
            samples = frames_every_second(vf)
        hits = recognize_frames(app, persons, samples)
        if kfs:
            shutil.rmtree(kf_tmp / vf.stem, ignore_errors=True)

        hit_any = False
        for name, hres in hits.items():
            if not hres["times"]:
                continue
            total_sec = len(hres["times"]) * interval
            times = hres["times"]
            best = (hres["best_sec"], hres["best_frame"])
            hit = total_sec >= args.min_sec
            dest = out / f"视频_{name}"
            if hit:
                hit_any = True
                dest.mkdir(exist_ok=True)
                if not (dest / vf.name).exists():
                    shutil.copy2(vf, dest / vf.name)
                frame = best[1]
                h, w = frame.shape[:2]
                sc = THUMB_SIDE / max(h, w)
                thumbs[(vf.name, name)] = (
                    cv2.resize(frame, (int(w * sc), int(h * sc))) if sc < 1 else frame)
                fdst = (workdir / "hit_frames" /
                        f"{name}_{vf.stem}_{fmt_ts(best[0]).replace(':', '_')}.jpg")
                fdst.parent.mkdir(exist_ok=True)
                imwrite_unicode(fdst, frame)
            shown = times if len(times) <= 40 else times[:40] + ["…"]
            rows.append([
                str(vf), vf.name, name,
                f"{total_sec:.0f}/{(dur or 0):.0f}秒",
                f"{hres['best_s']:.3f}",
                "已复制" if hit else "未达时长阈值",
                " ".join(fmt_ts(t) for t in shown),
            ])
        el = time.time() - t0
        log(f"视频 {vi}/{len(files)} {vf.name}"
            f"（{'关键帧' if use_keyframe else '全帧'}，{dur or '?'}秒）"
            f"{' ✔命中' if hit_any else ''}，累计 {el/60:.0f} 分钟")

    with open(workdir / "videos_report.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["视频路径", "文件名", "人物", "出现/总时长", "最高相似度", "动作", "出现时间点"])
        w.writerows(rows)

    by_person = {}
    for r in rows:
        if r[5] == "已复制":
            by_person.setdefault(r[2], []).append(r)
    cards = []
    for name, items in sorted(by_person.items()):
        cells = []
        for r in sorted(items, key=lambda x: x[0]):
            tb = thumbs.get((r[1], r[2]))
            if tb is None:
                continue
            cells.append(
                f"<figure><img src='data:image/jpeg;base64,{to_b64(tb)}'>"
                f"<figcaption>{r[1]}<br><small>相似度 {r[4]} · 出现 {r[3]}：{r[6]}</small></figcaption></figure>")
        cards.append(f"<h2>{name}（{len(items)} 个视频）</h2><div class='row'>{''.join(cells)}</div>")
    html = HTML_TMPL.replace("__TITLE__", "视频识别结果").replace("__BODY__", "".join(cards))
    (workdir / "videos_preview.html").write_text(html, encoding="utf-8")
    log(f"完成：videos_report.csv 与 videos_preview.html 已生成，命中视频在 {out}")


HTML_TMPL = """<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>__TITLE__</title><style>
body{font-family:"Microsoft YaHei",sans-serif;background:#fafafa;margin:24px}
h2{margin:28px 0 8px;border-left:4px solid #1976d2;padding-left:8px}
.row{display:flex;flex-wrap:wrap;gap:10px}
figure{margin:0;width:300px;background:#fff;border:3px solid #2e7d32;border-radius:8px;padding:6px}
figure img{width:100%;border-radius:4px}
figcaption{font-size:12px;margin-top:4px;line-height:1.5;word-break:break-all}
small{color:#888}
</style></head><body><h1>__TITLE__</h1>__BODY__
<p style="color:#aaa">长视频按关键帧采样（约每秒1帧）· 单帧相似度≥0.40 记为出现 · 累计≥1秒才算命中</p>
</body></html>"""


def main():
    ap = argparse.ArgumentParser(description="按人物筛选视频（InsightFace 本地识别）")
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("scan", help="扫描视频库并归档")
    s.add_argument("--src", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--min-sec", type=float, default=DEFAULT_MIN_SEC)
    s.add_argument("--workdir", default=str(Path(__file__).parent))
    s.set_defaults(func=cmd_scan)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
