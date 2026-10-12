# -*- coding: utf-8 -*-
"""
把按人物归档的结果重组为去重合集：每张照片只出现一次。
- 单人照 → 人物N/
- 多人同框 → 合影/
- 内容完全相同的副本（同大小才比对 md5）只保留一张，记入 duplicates.csv
不重新识别，直接读 find_person.py 扫描产生的 results.csv。
用法：python organize_family.py [--workdir 扫描结果目录] [--out 输出目录]
"""
import argparse
import csv
import hashlib
import shutil
import sys
from collections import defaultdict
from pathlib import Path


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="按人物归档结果重组去重（图片）")
    ap.add_argument("--workdir", default=str(Path(__file__).parent),
                    help="扫描结果目录（含 results.csv），默认脚本所在目录")
    ap.add_argument("--out", default="家人合集", help="输出目录，默认 ./家人合集")
    args = ap.parse_args()
    workdir = Path(args.workdir)
    out = Path(args.out)

    rows = list(csv.reader(open(workdir / "results.csv", encoding="utf-8-sig")))[1:]
    photo_persons = {}  # 照片路径 -> 命中人物列表
    for r in rows:
        if r[4] == "已复制":
            photo_persons.setdefault(r[0], []).append(r[2])
    if not photo_persons:
        sys.exit(f"{workdir / 'results.csv'} 里没有命中记录，请先运行 find_person.py scan")

    if out.exists():
        sys.exit(f"输出目录已存在，先处理它再跑：{out}")

    # 只有「文件大小相同」的才需要读全文算 md5 确认是否副本（省大量慢盘读取）
    entries = []
    for path_str, persons in sorted(photo_persons.items()):
        p = Path(path_str)
        if p.exists():
            entries.append((p, p.stat().st_size, tuple(persons)))
    by_size = defaultdict(int)
    for _, size, _ in entries:
        by_size[size] += 1
    suspicious_sizes = {s for s, n in by_size.items() if n > 1}

    seen_md5 = set()
    dups = []      # (重复文件, 与谁重复)
    stat = {}      # 文件夹 -> 张数
    for p, size, persons in entries:
        if size in suspicious_sizes:
            digest = md5_of(p)
            if digest in seen_md5:
                dups.append((p.name, "同内容副本"))
                continue
            seen_md5.add(digest)
        dest_dir = out / persons[0] if len(persons) == 1 else out / "合影"
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest_dir / p.name)
        stat[dest_dir.name] = stat.get(dest_dir.name, 0) + 1

    with open(workdir / "family_collection.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["文件名", "所在文件夹", "出现人物"])
        for path_str, persons in sorted(photo_persons.items()):
            p = Path(path_str)
            folder = persons[0] if len(persons) == 1 else "合影"
            w.writerow([p.name, folder, "+".join(sorted(persons))])

    print(f"共 {len(photo_persons)} 张照片含目标人物 → 去重后 {sum(stat.values())} 张：")
    for k, v in sorted(stat.items()):
        print(f"  {k}: {v} 张")
    print(f"内容完全相同的副本剔除 {len(dups)} 张（清单见 duplicates.csv）")
    with open(workdir / "duplicates.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["被剔除的副本", "与哪张重复"])
        w.writerows(dups)


if __name__ == "__main__":
    main()
