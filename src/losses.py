"""
Hàm mất mát.

supcon_loss : label-aware graph contrastive loss (Eq. 8-9), tức SupCon của
              Khosla et al. áp lên embedding nút của 2 view.
              Cặp dương = mọi mẫu CÙNG LỚP ở cả hai view (kể cả chính nút đó ở view kia).
grace_loss  : contrastive TỰ GIÁM SÁT (GRACE), cặp dương duy nhất = cùng nút ở view
              kia. Dùng cho ablation "SSLCon" ở Bảng VI.

Tổng (Eq. 12):  L = (1 - lambda) * L_con + lambda * L_ce
"""
from __future__ import annotations

import torch


def supcon_loss(z: torch.Tensor, labels: torch.Tensor, tau: float = 0.3) -> torch.Tensor:
    """z: [M, d] đã chuẩn hoá L2 (M = 2B, ghép 2 view); labels: [M]."""
    m = z.size(0)
    sim = z @ z.t() / tau
    self_mask = torch.eye(m, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, float("-inf"))           # loại anchor khỏi mẫu số
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (labels[:, None] == labels[None, :]) & ~self_mask
    n_pos = pos.sum(1)
    valid = n_pos > 0
    log_prob = log_prob.masked_fill(~pos, 0.0)                # tránh -inf * 0
    mean_log_prob_pos = log_prob.sum(1)[valid] / n_pos[valid]
    return -mean_log_prob_pos.mean()


def grace_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.3) -> torch.Tensor:
    """InfoNCE 2 chiều của GRACE; z1, z2: [B, d] đã chuẩn hoá, hàng i là cùng một nút."""
    def one_side(a, b):
        b_ = a.size(0)
        eye = torch.eye(b_, dtype=torch.bool, device=a.device)
        refl = (a @ a.t() / tau).masked_fill(eye, float("-inf"))   # negative trong cùng view
        betw = a @ b.t() / tau                                     # đường chéo = positive
        logits = torch.cat([betw, refl], dim=1)
        return -(betw.diagonal() - torch.logsumexp(logits, dim=1))
    return 0.5 * (one_side(z1, z2) + one_side(z2, z1)).mean()
