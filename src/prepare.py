"""
Ghép toàn bộ pipeline: CSV -> GraphData (đặc trưng, nhãn, line graph, split).

Cache 2 tầng (dùng chung giữa mọi lần chạy, kể cả khi chạy song song nhiều GPU):
  tầng 1  flows_<key>.pkl : các flow sau downsample   (key = file CSV + frac + data_seed)
  tầng 2  lg_<key>.npy    : cạnh line graph (int32)  (key = tầng 1 + host_key + max_clique + k_sample)
Phần còn lại (nhãn, split, target encoding, chuẩn hoá) phụ thuộc seed/label_ratio
nhưng chạy chỉ vài giây nên tính lại mỗi lần, không cache.

-> Đọc CSV 10+ GB và dựng line graph chỉ MỘT lần cho cả trăm lần chạy.
Ghi cache theo kiểu nguyên tử (ghi file tạm rồi os.replace) nên nhiều tiến trình
cùng đọc/ghi không làm hỏng file.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

from .data import (DEFAULT_CATEGORICAL, LABEL_COL, GraphData, build_line_graph,
                   feature_matrix, host_keys, load_netflow, make_labels,
                   standardize, target_encode)
from .splits import make_splits


def _hash(obj) -> str:
    return hashlib.md5(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def _file_sig(p: str):
    st = os.stat(p)
    return [os.path.abspath(p), st.st_size, int(st.st_mtime)]


def _atomic(path: str, write_fn):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        write_fn(f)
    os.replace(tmp, path)


def flows_key(args) -> str:
    return _hash(dict(files=[_file_sig(p) for p in args.data], frac=args.sample_frac, dseed=args.data_seed))


def graph_key(args) -> str:
    return _hash(dict(f=flows_key(args), host=args.host_key, mc=args.max_clique, ks=args.k_sample))


def load_flows(args, verbose=True) -> pd.DataFrame:
    path = os.path.join(args.cache_dir, f"flows_{flows_key(args)}.pkl") if args.cache_dir else None
    if path and os.path.exists(path):
        if verbose:
            print(f"[cache] flows <- {path}")
        return pd.read_pickle(path)
    t0 = time.time()
    df = load_netflow(args.data, sample_frac=args.sample_frac, seed=args.data_seed)
    if verbose:
        print(f"[data] đọc + downsample {len(df):,} flow trong {time.time() - t0:.0f}s")
    if path:
        _atomic(path, lambda f: df.to_pickle(f))
    return df


def load_line_graph(args, df, verbose=True) -> np.ndarray:
    path = os.path.join(args.cache_dir, f"lg_{graph_key(args)}.npy") if args.cache_dir else None
    if path and os.path.exists(path):
        if verbose:
            print(f"[cache] line graph <- {path}")
        return np.load(path).astype(np.int64)
    t0 = time.time()
    src, dst = host_keys(df, args.host_key)
    ei = build_line_graph(src, dst, max_clique=args.max_clique, k_sample=args.k_sample, seed=args.data_seed)
    if verbose:
        print(f"[graph] dựng line graph trong {time.time() - t0:.0f}s")
    if path:
        _atomic(path, lambda f: np.save(f, ei.astype(np.int32)))
    return ei


def prepare(args, verbose: bool = True) -> GraphData:
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
    df = load_flows(args, verbose)
    edge_index = load_line_graph(args, df, verbose)

    y, class_names = make_labels(df, args.task)
    y_bin = df[LABEL_COL].astype(int).to_numpy()
    # Split TRƯỚC khi target encoding để encoder chỉ thấy nhãn huấn luyện
    train_idx, val_idx, test_idx = make_splits(y, label_ratio=args.label_ratio, k_shot=args.k_shot,
                                               val_ratio=args.val_ratio, seed=args.seed)
    feats, names = feature_matrix(df)
    if args.target_encoding:
        feats = target_encode(feats, y_bin, train_idx, DEFAULT_CATEGORICAL, seed=args.seed)
    # Scaler là biến đổi không giám sát -> fit trên toàn bộ nút (transductive)
    x = standardize(feats.to_numpy(), fit_idx=None, log1p=args.log1p)

    if verbose:
        n, e = len(y), edge_index.shape[1] // 2
        iso = n - len(np.unique(edge_index[0])) if e else n
        print(f"[data] {n:,} flow | {len(class_names)} lớp: " +
              ", ".join(f"{c}={np.sum(y == i):,}" for i, c in enumerate(class_names)))
        print(f"[graph] {n:,} nút, {e:,} cạnh vô hướng, bậc TB {2 * e / max(n, 1):.1f}, "
              f"{iso:,} nút cô lập ({100 * iso / n:.1f}%)")
        print(f"[split] train(có nhãn)={len(train_idx):,}  val={len(val_idx):,}  test={len(test_idx):,}")

    return GraphData(x=x, y=y, edge_index=edge_index, class_names=class_names, feature_names=names,
                     train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


def main():
    """Dựng sẵn cache (không cần torch):  python -m src.prepare --data X.csv"""
    import argparse
    a = add_data_args(argparse.ArgumentParser()).parse_args()
    prepare(a)


def add_data_args(p):
    g = p.add_argument_group("dữ liệu")
    g.add_argument("--data", nargs="+", required=True, help="đường dẫn CSV NetFlow v2")
    g.add_argument("--task", choices=["binary", "multi"], default="multi")
    g.add_argument("--sample-frac", type=float, default=0.02, help="tỉ lệ downsample (bài báo: 0.02)")
    g.add_argument("--host-key", choices=["ip_port", "ip"], default="ip_port",
                   help="cách định danh host khi dựng line graph (bài báo: ip_port)")
    g.add_argument("--max-clique", type=int, default=100,
                   help="host có <= số flow này thì nối đầy đủ; lớn hơn thì lấy mẫu")
    g.add_argument("--k-sample", type=int, default=10, help="số láng giềng lấy mẫu cho host hub")
    g.add_argument("--label-ratio", type=float, default=0.3, help="tỉ lệ nút có nhãn (bài báo: 0.3)")
    g.add_argument("--k-shot", type=int, default=None, help="nếu đặt: K nút có nhãn mỗi lớp")
    g.add_argument("--val-ratio", type=float, default=0.1)
    g.add_argument("--no-target-encoding", dest="target_encoding", action="store_false")
    g.add_argument("--log1p", action="store_true", help="log1p trước khi chuẩn hoá (không có trong bài báo)")
    g.add_argument("--cache-dir", default="cache", help="'' để tắt cache; trên Kubeflow nên trỏ vào PVC dùng chung")
    g.add_argument("--seed", type=int, default=0, help="seed cho split + khởi tạo mô hình")
    g.add_argument("--data-seed", type=int, default=0, help="seed cho downsample + lấy mẫu line graph")
    return p


if __name__ == "__main__":
    main()
