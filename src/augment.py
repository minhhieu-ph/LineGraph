"""
Adaptive graph augmentation (mục IV-B.2, Eq. 1-5), theo GCA (Zhu et al., WWW'21).

Ý tưởng: cạnh / chiều đặc trưng càng "quan trọng" (gắn với nút có độ trung tâm
cao) thì xác suất bị bỏ / che càng thấp.

  Cạnh (u,v):  w = (phi(u) + phi(v)) / 2,  s = log w
               p_uv = min( (s_max - s) / (s_max - mean(s)) * p_e , p_tau )      (Eq. 2)
  Đặc trưng i: w_i = sum_u |x_ui| * phi(u), s_i = log w_i
               p_i  = min( (s_max - s_i) / (s_max - mean(s)) * p_f , p_tau )    (Eq. 4-5)

Mọi hàm làm việc trên danh sách cạnh VÔ HƯỚNG `und` [2, E] (u < v) để việc bỏ
cạnh giữ được tính đối xứng.
"""
from __future__ import annotations

import torch

EPS = 1e-12


def undirected(edge_index: torch.Tensor) -> torch.Tensor:
    return edge_index[:, edge_index[0] < edge_index[1]]


def degree(und: torch.Tensor, n: int) -> torch.Tensor:
    ones = torch.ones(und.size(1), device=und.device)
    deg = torch.zeros(n, device=und.device)
    deg.index_add_(0, und[0], ones)
    deg.index_add_(0, und[1], ones)
    return deg


def pagerank(und: torch.Tensor, n: int, damping: float = 0.85, iters: int = 20) -> torch.Tensor:
    src = torch.cat([und[0], und[1]])
    dst = torch.cat([und[1], und[0]])
    deg_inv = 1.0 / degree(und, n).clamp(min=1.0)
    x = torch.ones(n, device=und.device)
    for _ in range(iters):
        agg = torch.zeros(n, device=und.device).index_add_(0, dst, x[src] * deg_inv[src])
        x = (1 - damping) + damping * agg
    return x


def centrality(und: torch.Tensor, n: int, kind: str = "degree") -> torch.Tensor:
    if kind == "degree":
        return degree(und, n)
    if kind == "pagerank":
        return pagerank(und, n)
    raise ValueError(kind)


def _normalize(s: torch.Tensor) -> torch.Tensor:
    return (s.max() - s) / (s.max() - s.mean()).clamp(min=EPS)


def edge_drop_weights(und: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    w = (phi[und[0]] + phi[und[1]]) / 2
    return _normalize(torch.log(w.clamp(min=EPS)))


def feature_mask_weights(x: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    w = x.abs().t() @ phi                         # [F]
    return _normalize(torch.log(w.clamp(min=EPS)))


def drop_edges(und: torch.Tensor, weights: torch.Tensor, p: float, p_tau: float) -> torch.Tensor:
    prob = (weights * p).clamp(max=p_tau)
    keep = torch.rand_like(prob) >= prob          # Eq. 1: giữ với xác suất 1 - p_uv
    return und[:, keep]


def mask_features(x: torch.Tensor, weights: torch.Tensor, p: float, p_tau: float) -> torch.Tensor:
    prob = (weights * p).clamp(max=p_tau)
    m = (torch.rand_like(prob) >= prob).to(x.dtype)  # m_i ~ Bern(1 - p_i), dùng chung cho mọi nút (Eq. 3)
    return x * m


class Augmentor:
    """Sinh hai view. mode='adaptive' (bài báo) hoặc 'uniform' (GRACE, để ablation)."""

    def __init__(self, x, und, mode="adaptive", centrality_kind="degree",
                 pe=(0.2, 0.4), pf=(0.1, 0.2), p_tau=0.7):
        self.und, self.pe, self.pf, self.p_tau = und, pe, pf, p_tau
        n = x.size(0)
        if mode == "adaptive":
            phi = centrality(und, n, centrality_kind)
            self.ew = edge_drop_weights(und, phi)
            self.fw = feature_mask_weights(x, phi)
        elif mode == "uniform":
            self.ew = torch.ones(und.size(1), device=x.device)
            self.fw = torch.ones(x.size(1), device=x.device)
        else:
            raise ValueError(mode)

    def view(self, x, i: int):
        und = drop_edges(self.und, self.ew, self.pe[i], self.p_tau)
        return mask_features(x, self.fw, self.pf[i], self.p_tau), und
