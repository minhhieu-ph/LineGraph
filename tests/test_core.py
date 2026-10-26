"""
Kiểm thử các thành phần PyTorch bằng cách so với cài đặt "ngây thơ" (vòng lặp).
Chạy:  python -m pytest -q tests      (hoặc: python tests/test_core.py)
"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.augment import Augmentor, degree, drop_edges, edge_drop_weights, undirected  # noqa: E402
from src.data import build_line_graph  # noqa: E402
from src.losses import grace_loss, supcon_loss  # noqa: E402
from src.model import FeCoGraph, build_adj  # noqa: E402

torch.manual_seed(0)


def naive_supcon(z, y, tau):
    m = len(z)
    total, cnt = 0.0, 0
    for i in range(m):
        denom = sum(math.exp(float(z[i] @ z[a]) / tau) for a in range(m) if a != i)
        pos = [p for p in range(m) if p != i and y[p] == y[i]]
        if not pos:
            continue
        total += -sum(math.log(math.exp(float(z[i] @ z[p]) / tau) / denom) for p in pos) / len(pos)
        cnt += 1
    return total / cnt


def naive_grace(z1, z2, tau):
    def side(a, b):
        out = []
        for i in range(len(a)):
            pos = math.exp(float(a[i] @ b[i]) / tau)
            neg = sum(math.exp(float(a[i] @ b[k]) / tau) for k in range(len(a)) if k != i) + \
                sum(math.exp(float(a[i] @ a[k]) / tau) for k in range(len(a)) if k != i)
            out.append(-math.log(pos / (pos + neg)))
        return out
    return 0.5 * (np.mean(side(z1, z2)) + np.mean(side(z2, z1)))


def test_supcon_matches_naive():
    z = torch.nn.functional.normalize(torch.randn(12, 5), dim=1)
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 0, 1, 3, 3, 2])
    assert abs(supcon_loss(z, y, 0.3).item() - naive_supcon(z, y, 0.3)) < 1e-4


def test_supcon_gradient_finite():
    z = torch.nn.functional.normalize(torch.randn(8, 4), dim=1).requires_grad_()
    y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 4])   # có lớp chỉ 1 mẫu -> anchor không có positive
    supcon_loss(z, y, 0.1).backward()
    assert torch.isfinite(z.grad).all()


def test_grace_matches_naive():
    z1 = torch.nn.functional.normalize(torch.randn(7, 4), dim=1)
    z2 = torch.nn.functional.normalize(torch.randn(7, 4), dim=1)
    assert abs(grace_loss(z1, z2, 0.5).item() - naive_grace(z1, z2, 0.5)) < 1e-4


def test_gcn_adj_matches_dense():
    ei = torch.tensor([[0, 1, 2, 0], [1, 2, 3, 2]])
    n = 5                                            # nút 4 cô lập
    adj = build_adj(ei, n, "gcn").to_dense()
    A = torch.zeros(n, n)
    A[ei[0], ei[1]] = 1
    A[ei[1], ei[0]] = 1
    A += torch.eye(n)
    d = A.sum(1).pow(-0.5)
    assert torch.allclose(adj, d[:, None] * A * d[None, :], atol=1e-6)
    sage = build_adj(ei, n, "sage").to_dense()
    assert torch.allclose(sage.sum(1)[:4], torch.ones(4)) and sage[4].sum() == 0


def test_adaptive_drop_prefers_low_centrality():
    # sao: nút 0 nối mọi nút -> cạnh qua nút trung tâm "quan trọng" hơn cạnh rìa
    edges = [(0, i) for i in range(1, 30)] + [(i, i + 1) for i in range(1, 29)]
    ei = torch.tensor(edges).t()
    und = undirected(torch.cat([ei, ei.flip(0)], 1))
    w = edge_drop_weights(und, degree(und, 30))
    hub = (und[0] == 0) | (und[1] == 0)
    assert w[hub].mean() < w[~hub].mean()
    kept = drop_edges(und, w, p=0.5, p_tau=0.7)
    assert kept.size(1) <= und.size(1)


def test_line_graph_small():
    # flow0: A->B, flow1: B->C, flow2: D->E   => chỉ (0,1) chung host B
    src = np.array(["A", "B", "D"]); dst = np.array(["B", "C", "E"])
    ei = build_line_graph(src, dst)
    assert set(map(tuple, ei.T.tolist())) == {(0, 1), (1, 0)}


def test_model_forward_backward():
    n, f, c = 50, 39, 4
    x = torch.randn(n, f)
    ei = torch.randint(0, n, (2, 200))
    ei = ei[:, ei[0] != ei[1]]
    und = undirected(torch.cat([ei, ei.flip(0)], 1)).unique(dim=1)
    for kind in ["gcn", "sage"]:
        m = FeCoGraph(f, c, gnn=kind)
        aug = Augmentor(x, und)
        x1, u1 = aug.view(x, 0)
        h = m.embed(x, build_adj(und, n, kind))
        z = m.project(m.embed(x1, build_adj(u1, n, kind)))
        y = torch.randint(0, c, (n,))
        loss = torch.nn.functional.cross_entropy(m.classify(h), y) + supcon_loss(torch.cat([z, z]), torch.cat([y, y]))
        loss.backward()
        assert h.shape == (n, 32) and torch.allclose(z.norm(dim=1), torch.ones(n), atol=1e-5)
        assert all(p.grad is not None for p in m.encoder.parameters())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK ", name)
