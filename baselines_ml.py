"""
Baseline học máy truyền thống (Bảng III của bài báo) trên CÙNG split few-shot.
Chỉ dùng đặc trưng flow, không dùng đồ thị; huấn luyện trên các nút có nhãn.

  python baselines_ml.py --data data/NF-ToN-IoT-v2.csv --task multi --runs 3
"""
import argparse
import json
import os
import time

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src.prepare import add_data_args, prepare


def models(seed, n_jobs, xgb_gpu):
    m = {
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=n_jobs),
        "AdaBoost": AdaBoostClassifier(random_state=seed),
        "DecisionTree": DecisionTreeClassifier(random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=n_jobs, random_state=seed),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=100, n_jobs=n_jobs, random_state=seed),
    }
    try:
        from xgboost import XGBClassifier
        kw = dict(device="cuda") if xgb_gpu else {}
        m["XGBoost"] = XGBClassifier(n_estimators=200, tree_method="hist", random_state=seed, n_jobs=n_jobs, **kw)
    except ImportError:
        print("(chưa cài xgboost -> bỏ qua XGBoost)")
    return m


def main():
    p = add_data_args(argparse.ArgumentParser())
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--out", default="results")
    p.add_argument("--tag", default=None)
    p.add_argument("--methods", nargs="+", default=None, help="chỉ chạy các phương pháp này")
    p.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 0)) or -1,
                   help="n_jobs cho sklearn/xgboost (mặc định: OMP_NUM_THREADS)")
    p.add_argument("--xgb-gpu", action="store_true", help="XGBoost chạy trên GPU (xgboost >= 2.0)")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    base, table = args.seed, {}
    for r in range(args.runs):
        args.seed = base + r
        g = prepare(args, verbose=(r == 0))
        C = g.num_classes
        Xtr, ytr = g.x[g.train_idx], g.y[g.train_idx]
        Xte, yte = g.x[g.test_idx], g.y[g.test_idx]
        # XGBoost cần nhãn liên tục 0..k-1 trên tập train
        present = np.unique(ytr)
        remap = {c: i for i, c in enumerate(present)}
        for name, clf in models(args.seed, args.threads, args.xgb_gpu).items():
            if args.methods and name not in args.methods:
                continue
            t0 = time.time()
            if name == "XGBoost":
                clf.fit(Xtr, np.vectorize(remap.get)(ytr))
                pred = present[clf.predict(Xte)]
            else:
                clf.fit(Xtr, ytr)
                pred = clf.predict(Xte)
            P, R, F, _ = precision_recall_fscore_support(yte, pred, labels=range(C), average="macro", zero_division=0)
            acc = accuracy_score(yte, pred)
            table.setdefault(name, []).append(dict(seed=args.seed, test=dict(acc=acc, precision=P, recall=R, f1=F),
                                                   train_time_s=time.time() - t0))
            print(f"[run {r}] {name:<13s} acc {acc:.4f}  F1 {F:.4f}  ({time.time() - t0:.1f}s)")
    args.seed = base

    print("\n%-13s %8s %8s %8s %8s" % ("Method", "Acc", "Prec", "Recall", "F1"))
    tag = args.tag or (f"{args.task}_ml_" + (f"k{args.k_shot}" if args.k_shot else f"r{args.label_ratio}"))
    stamp = f"{time.strftime('%Y%m%d-%H%M%S')}_{os.getpid()}"
    for name, runs in table.items():
        keys = ["acc", "precision", "recall", "f1"]
        summary = {k: (float(np.mean([r["test"][k] for r in runs])), float(np.std([r["test"][k] for r in runs])))
                   for k in keys}
        print("%-13s %8.2f %8.2f %8.2f %8.2f" % (name, *[100 * summary[k][0] for k in keys]))
        seeds = "-".join(str(r["seed"]) for r in runs)
        path = os.path.join(args.out, f"{tag}__{name}__s{seeds}__{stamp}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dict(tag=tag, method=name, config=vars(args), summary=summary, runs=runs),
                      f, indent=2, ensure_ascii=False, default=float)
    print(f"Đã lưu kết quả vào {args.out}/")


if __name__ == "__main__":
    main()
