# -*- coding: utf-8 -*-
"""
把视频归档去重重组（在同盘内移动，秒级完成）：
- 只出现一位目标人物的视频 → 视频_人物N/
- 多人同框 → 视频_合影/
- 每个视频只保留一份，多余副本删除
读 find_person_video.py 扫描产生的 videos_report.csv。
注意：本工具假设 scan 已经把视频复制到了 <out>/视频_人物N/ 下（同盘移动才快）。
用法：python organize_videos.py [--workdir 扫描结果目录] [--out 视频归档目录]
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="按人物归档结果重组去重（视频）")
    ap.add_argument("--workdir", default=str(Path(__file__).parent),
                    help="扫描结果目录（含 videos_report.csv），默认脚本所在目录")
    ap.add_argument("--out", default="家人合集", help="视频归档目录（scan 的 --out），默认 ./家人合集")
    args = ap.parse_args()
    workdir = Path(args.workdir)
    out = Path(args.out)

    rows = list(csv.reader(open(workdir / "videos_report.csv", encoding="utf-8-sig")))[1:]
    by_video = {}  # 文件名 -> 人物集合
    for r in rows:
        if r[5] == "已复制":
            by_video.setdefault(r[1], set()).add(r[2])
    if not by_video:
        sys.exit(f"{workdir / 'videos_report.csv'} 里没有命中记录，请先运行 find_person_video.py scan")

    stat, removed = {}, 0
    for fname, persons in sorted(by_video.items()):
        target_dir = out / (f"视频_{sorted(persons)[0]}" if len(persons) == 1 else "视频_合影")
        target_dir.mkdir(exist_ok=True)
        target = target_dir / fname
        # 从现有归档副本里找源（同盘内移动）
        copies = [out / f"视频_{p}" / fname for p in sorted(persons)]
        existing = [c for c in copies if c.exists()]
        if target.exists():
            existing = [c for c in existing if c.resolve() != target.resolve()]
        if not existing and not target.exists():
            print(f"!! 找不到副本，跳过：{fname}")
            continue
        if not target.exists() and existing:
            shutil.move(str(existing[0]), str(target))
        for c in existing:
            if c.exists() and c.resolve() != target.resolve():
                c.unlink()
                removed += 1
        stat[target_dir.name] = stat.get(target_dir.name, 0) + 1

    # 清理空文件夹
    for d in sorted(out.glob("视频_*")):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()

    print(f"共 {len(by_video)} 个命中视频 → 去重后：")
    for k, v in sorted(stat.items()):
        print(f"  {k}: {v} 个")
    print(f"删除多余副本 {removed} 个")


if __name__ == "__main__":
    main()
