"""
Data pipeline cho FeCoGraph (bản tập trung, không federated).

Các bước (theo mục V-A.2 của bài báo):
  1. Đọc NetFlow v2 CSV (NF-BoT-IoT-v2, NF-ToN-IoT-v2, NF-CSE-CIC-IDS2018-v2).
  2. Downsample phân tầng theo nhãn tấn công (bài báo: 2%).
  3. Gộp IP + port thành "host key" (bài báo: concat IP và port).
  4. Điền NaN / inf bằng 0.
  5. Target encoding cho các cột phân loại (chỉ fit trên tập có nhãn huấn luyện
     để tránh rò rỉ nhãn).
  6. StandardScaler.
  7. Xây line graph: mỗi flow = 1 nút, 2 flow chung host thì nối cạnh.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Tên cột chuẩn của bộ NetFlow v2 (Sarhan et al., 43 đặc trưng + Label + Attack)
# ----------------------------------------------------------------------------
SRC_IP, SRC_PORT = "IPV4_SRC_ADDR", "L4_SRC_PORT"
DST_IP, DST_PORT = "IPV4_DST_ADDR", "L4_DST_PORT"
LABEL_COL, ATTACK_COL = "Label", "Attack"
ID_COLS = [SRC_IP, SRC_PORT, DST_IP, DST_PORT]

# Cột mang tính phân loại (mã số nhưng không có thứ tự) -> target encoding
DEFAULT_CATEGORICAL = [
    "PROTOCOL", "L7_PROTO", "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
    "ICMP_TYPE", "ICMP_IPV4_TYPE", "DNS_QUERY_TYPE", "FTP_COMMAND_RET_CODE",
]


@dataclass
class GraphData:
    """Toàn bộ dữ liệu dạng numpy; train.py sẽ chuyển sang torch."""
    x: np.ndarray                 # [N, F] đặc trưng flow (đã chuẩn hoá)
    y: np.ndarray                 # [N] nhãn số nguyên
    edge_index: np.ndarray        # [2, E] cạnh vô hướng của line graph (cả 2 chiều)
    class_names: list[str]
    feature_names: list[str]
    train_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    val_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    test_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))

    @property
    def num_nodes(self) -> int:
        return self.x.shape[0]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


# ----------------------------------------------------------------------------
# 1-2. Đọc và downsample
# ----------------------------------------------------------------------------
def load_netflow(paths: list[str], sample_frac: float = 0.02, seed: int = 0,
                 chunksize: int | None = 2_000_000) -> pd.DataFrame:
    """Đọc một hay nhiều CSV, downsample phân tầng theo cột Attack.

    Đọc theo chunk để không cần nạp toàn bộ file 10+ GB vào RAM:
    mỗi chunk được lấy mẫu `sample_frac` theo từng lớp.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for p in paths:
        reader = pd.read_csv(p, chunksize=chunksize) if chunksize else [pd.read_csv(p)]
        for chunk in reader:
            if sample_frac < 1.0:
                # Bernoulli từng dòng: kỳ vọng đúng tỉ lệ mỗi lớp trên toàn file,
                # kể cả lớp hiếm bị chia nhỏ qua nhiều chunk (groupby.sample sẽ làm tròn về 0).
                chunk = chunk[rng.random(len(chunk)) < sample_frac]
            parts.append(chunk)
    df = pd.concat(parts, ignore_index=True)
    # Một số bản phát hành có khoảng trắng thừa trong tên lớp
    df[ATTACK_COL] = df[ATTACK_COL].astype(str).str.strip()
    return df


# ----------------------------------------------------------------------------
# Nhãn
# ----------------------------------------------------------------------------
def make_labels(df: pd.DataFrame, task: str) -> tuple[np.ndarray, list[str]]:
    if task == "binary":
        y = df[LABEL_COL].astype(int).to_numpy()
        return y, ["Benign", "Malicious"]
    if task == "multi":
        names = sorted(df[ATTACK_COL].unique(), key=lambda s: (s.lower() != "benign", s))
        mapping = {n: i for i, n in enumerate(names)}
        return df[ATTACK_COL].map(mapping).to_numpy().astype(np.int64), list(names)
    raise ValueError(f"task không hợp lệ: {task}")


# ----------------------------------------------------------------------------
# 3. Host key và line graph
# ----------------------------------------------------------------------------
def host_keys(df: pd.DataFrame, mode: str = "ip_port") -> tuple[np.ndarray, np.ndarray]:
    """mode='ip_port' (theo bài báo) hoặc 'ip' (chỉ IP)."""
    if mode == "ip_port":
        src = df[SRC_IP].astype(str) + ":" + df[SRC_PORT].astype(str)
        dst = df[DST_IP].astype(str) + ":" + df[DST_PORT].astype(str)
    elif mode == "ip":
        src, dst = df[SRC_IP].astype(str), df[DST_IP].astype(str)
    else:
        raise ValueError(mode)
    return src.to_numpy(), dst.to_numpy()


def build_line_graph(src: np.ndarray, dst: np.ndarray, max_clique: int = 100,
                     k_sample: int = 10, seed: int = 0) -> np.ndarray:
    """Xây line graph L(G).

    Trong đồ thị gốc G: host là nút, flow là cạnh.
    Trong L(G): flow là nút; hai flow nối nhau nếu chung ít nhất một host.

    Line graph chính xác có d(d-1)/2 cạnh cho mỗi host bậc d -> với host
    "hub" (vd. victim:80 nhận 100k flow) là ~5e9 cạnh, không thể lưu.
    Vì vậy:
      - host có <= max_clique flow: nối đầy đủ (clique) như định nghĩa.
      - host có > max_clique flow: mỗi flow nối với k_sample flow ngẫu nhiên
        cùng host (xấp xỉ thưa của clique).
    Đặt max_clique rất lớn để có line graph chính xác (nếu RAM cho phép).

    Trả về edge_index [2, E] vô hướng (đã có cả 2 chiều), không self-loop.
    """
    rng = np.random.default_rng(seed)
    n = len(src)
    host_id, _ = pd.factorize(np.concatenate([src, dst]))
    flow_id = np.concatenate([np.arange(n), np.arange(n)])

    # Bỏ trùng (host, flow) khi src == dst
    key = host_id.astype(np.int64) * n + flow_id
    key = np.unique(key)
    host_id, flow_id = key // n, key % n          # đã sắp theo host

    _, starts, counts = np.unique(host_id, return_index=True, return_counts=True)
    us, vs = [], []
    # (a) host nhỏ: vector hoá theo KÍCH THƯỚC nhóm thay vì lặp từng host
    #     (với dữ liệu đầy đủ có hàng triệu host -> nhanh hơn ~100 lần)
    small = (counts >= 2) & (counts <= max_clique)
    for c in np.unique(counts[small]):
        st = starts[small & (counts == c)]
        members = flow_id[st[:, None] + np.arange(c)[None, :]]      # [số nhóm, c]
        iu, ju = np.triu_indices(c, k=1)
        us.append(members[:, iu].ravel()); vs.append(members[:, ju].ravel())
    # (b) host hub: lấy mẫu k láng giềng mỗi flow (số hub ít nên lặp được)
    for st, c in zip(starts[counts > max_clique], counts[counts > max_clique]):
        members = flow_id[st:st + c]
        partners = members[rng.integers(0, c, size=(c, k_sample))]
        us.append(np.repeat(members, k_sample)); vs.append(partners.ravel())

    if not us:
        return np.zeros((2, 0), dtype=np.int64)
    u = np.concatenate(us).astype(np.int64)
    v = np.concatenate(vs).astype(np.int64)
    keep = u != v
    u, v = u[keep], v[keep]
    lo, hi = np.minimum(u, v), np.maximum(u, v)
    und = np.unique(lo * n + hi)
    lo, hi = und // n, und % n
    return np.stack([np.concatenate([lo, hi]), np.concatenate([hi, lo])])


# ----------------------------------------------------------------------------
# 5-6. Đặc trưng
# ----------------------------------------------------------------------------
def feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    drop = set(ID_COLS + [LABEL_COL, ATTACK_COL])
    cols = [c for c in df.columns if c not in drop]
    feats = df[cols].apply(pd.to_numeric, errors="coerce")
    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float64)
    return feats, cols


def _te_map(values: np.ndarray, target: np.ndarray, prior: float, smoothing: float) -> pd.Series:
    stats = pd.DataFrame({"v": values, "t": target}).groupby("v")["t"].agg(["mean", "count"])
    return (stats["mean"] * stats["count"] + prior * smoothing) / (stats["count"] + smoothing)


def target_encode(feats: pd.DataFrame, y_bin: np.ndarray, fit_idx: np.ndarray,
                  cols: list[str], smoothing: float = 10.0, n_folds: int = 5,
                  max_card: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Target encoding có làm trơn, chỉ dùng nhãn của các dòng fit_idx (tập có nhãn).

    Mã hoá theo tỉ lệ độc hại (nhãn nhị phân) cho cả bài toán nhị phân và đa lớp.

    Chống rò rỉ nhãn:
      - Dòng thuộc fit_idx được mã hoá OUT-OF-FOLD (K-fold): giá trị của một dòng
        không bao giờ được tính từ chính nhãn của nó. Nếu không làm vậy, cột có
        nhiều giá trị khác nhau sẽ "chứa" nhãn train -> train acc 100%, test sụp.
      - Các dòng còn lại (val/test, không nhãn) dùng thống kê trên toàn fit_idx.
      - Cột có > max_card giá trị khác nhau trên tập fit được giữ nguyên (không encode).
    Giá trị chưa thấy -> prior (tỉ lệ độc hại trung bình).
    """
    feats = feats.copy()
    rng = np.random.default_rng(seed)
    prior = float(y_bin[fit_idx].mean())
    folds = np.array_split(rng.permutation(fit_idx), n_folds)
    for c in cols:
        if c not in feats.columns:
            continue
        v_all = feats[c].to_numpy()
        if len(np.unique(v_all[fit_idx])) > max_card:
            continue
        enc_full = _te_map(v_all[fit_idx], y_bin[fit_idx], prior, smoothing)
        out = pd.Series(v_all).map(enc_full).fillna(prior).to_numpy(dtype=np.float64, copy=True)
        for k in range(n_folds):
            hold = folds[k]
            rest = np.concatenate([folds[j] for j in range(n_folds) if j != k])
            enc_k = _te_map(v_all[rest], y_bin[rest], prior, smoothing)
            out[hold] = pd.Series(v_all[hold]).map(enc_k).fillna(prior).to_numpy(dtype=np.float64)
        feats[c] = out
    return feats


def standardize(x: np.ndarray, fit_idx: np.ndarray | None = None, log1p: bool = False) -> np.ndarray:
    if log1p:  # tuỳ chọn, không có trong bài báo; giúp các cột byte/packet đuôi dài
        x = np.sign(x) * np.log1p(np.abs(x))
    ref = x if fit_idx is None else x[fit_idx]
    mu, sd = ref.mean(0), ref.std(0)
    sd[sd < 1e-12] = 1.0
    return ((x - mu) / sd).astype(np.float32)
