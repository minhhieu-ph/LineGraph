"""
Chia tập theo kịch bản few-shot (transductive: tất cả nút nằm trong cùng một
line graph, chỉ một phần nút có nhãn khi huấn luyện).

Hai chế độ:
  - label_ratio: lấy r% nút mỗi lớp làm tập có nhãn (bài báo: r = 30%).
  - k_shot     : lấy đúng K nút mỗi lớp (few-shot "đúng nghĩa", bài báo không làm).
Phần còn lại chia thành val (để chọn epoch / siêu tham số) và test.
"""
from __future__ import annotations

import numpy as np


def _per_class_take(y: np.ndarray, pool: np.ndarray, rng, ratio=None, k=None, min_keep=1):
    taken = []
    for c in np.unique(y[pool]):
        idx = pool[y[pool] == c]
        idx = rng.permutation(idx)
        if k is not None:
            m = min(k, max(len(idx) - min_keep, 1))   # chừa lại ít nhất min_keep mẫu để test
        else:
            m = int(round(ratio * len(idx)))
            m = max(1, min(m, len(idx) - min_keep)) if len(idx) > 1 else len(idx)
        taken.append(idx[:m])
    return np.sort(np.concatenate(taken)) if taken else np.array([], dtype=np.int64)


def make_splits(y: np.ndarray, label_ratio: float = 0.3, k_shot: int | None = None,
                val_ratio: float = 0.1, seed: int = 0):
    """Trả về (train_idx, val_idx, test_idx), phân tầng theo lớp.

    val_ratio tính trên TỔNG số nút, lấy từ phần không có nhãn huấn luyện.
    Với lớp quá hiếm (vd. Theft trong BoT-IoT), mỗi tập vẫn cố có >= 1 mẫu.
    """
    rng = np.random.default_rng(seed)
    all_idx = np.arange(len(y))
    if k_shot is not None:
        train = _per_class_take(y, all_idx, rng, k=k_shot, min_keep=2)
    else:
        train = _per_class_take(y, all_idx, rng, ratio=label_ratio, min_keep=2)
    rest = np.setdiff1d(all_idx, train)
    if val_ratio > 0:
        val_frac_of_rest = min(val_ratio * len(y) / max(len(rest), 1), 0.9)
        val = _per_class_take(y, rest, rng, ratio=val_frac_of_rest, min_keep=1)
    else:
        val = np.array([], dtype=np.int64)
    test = np.setdiff1d(rest, val)
    return train.astype(np.int64), val.astype(np.int64), test.astype(np.int64)
