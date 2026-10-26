"""
Gộp mọi file kết quả JSON (train.py, baselines_ml.py) theo (tag, phương pháp):
trung bình ± độ lệch chuẩn qua các seed, xuất bảng Markdown + CSV.

  python collect_results.py --results results --out summary
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="summary")
    a = ap.parse_args()

    groups = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(a.results, "*.json"))):
        try:
            d = json.load(open(path, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "tag" not in d or "runs" not in d:
            continue
        for r in d["runs"]:
            groups[(d["tag"], d.get("method", "?"))].append(r)

    keys = ["acc", "precision", "recall", "f1"]
    rows = []
    for (tag, method), runs in sorted(groups.items()):
        seen, uniq = set(), []                      # một seed chỉ tính một lần
        for r in runs:
            if r.get("seed") not in seen:
                seen.add(r.get("seed"))
                uniq.append(r)
        m = {k: 100 * np.array([r["test"][k] for r in uniq]) for k in keys}
        rows.append(dict(tag=tag, method=method, n=len(uniq),
                         **{k: f"{m[k].mean():.2f} ± {m[k].std():.2f}" for k in keys},
                         time_s=f"{np.mean([r.get('train_time_s', 0) for r in uniq]):.0f}",
                         vram_gib=f"{max(r.get('peak_vram_gib', 0) for r in uniq):.2f}"))

    if not rows:
        print("Chưa có kết quả.")
        return
    cols = ["tag", "method", "n", *keys, "time_s", "vram_gib"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    with open(os.path.join(a.out, "summary.csv"), "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(f'"{r[c]}"' for c in cols) + "\n")
    print("\n".join(md))
    print(f"\nĐã ghi {a.out}/summary.md và summary.csv")


if __name__ == "__main__":
    main()
