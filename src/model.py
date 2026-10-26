"""
Kiến trúc FeCoGraph (mục IV-B.3): Encoder GNN dùng chung -> {Projector, Classifier}.

GCN được viết bằng phép nhân ma trận thưa của PyTorch thuần (không cần PyG/DGL),
nên chỉ cần cài torch.

  Encoder   : 2 lớp GCN, 64 -> 32 (bài báo, mục V-A.5)
  Projector : MLP 2 lớp (ẩn 256), đầu ra chuẩn hoá L2 lên siêu cầu đơn vị;
              chỉ dùng khi huấn luyện, bỏ đi lúc suy luận.
  Classifier: MLP fully-connected -> logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_adj(und: torch.Tensor, n: int, kind: str = "gcn") -> torch.Tensor:
    """Ma trận kề thưa từ cạnh vô hướng.

    gcn : D^-1/2 (A + I) D^-1/2   (Kipf & Welling)
    sage: D^-1 A                  (trung bình láng giềng, không self-loop)
    """
    dev = und.device
    row = torch.cat([und[0], und[1]])
    col = torch.cat([und[1], und[0]])
    if kind == "gcn":
        loop = torch.arange(n, device=dev)
        row, col = torch.cat([row, loop]), torch.cat([col, loop])
    deg = torch.zeros(n, device=dev).index_add_(0, row, torch.ones_like(row, dtype=torch.float))
    if kind == "gcn":
        dis = deg.pow(-0.5)
        val = dis[row] * dis[col]
    else:
        val = 1.0 / deg.clamp(min=1.0)[row]
    return torch.sparse_coo_tensor(torch.stack([row, col]), val, (n, n)).coalesce()


class GCNLayer(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.lin = nn.Linear(i, o)

    def forward(self, x, adj):
        return torch.sparse.mm(adj, self.lin(x))


class SAGELayer(nn.Module):
    """h_v = W [h_v || mean_{u in N(v)} h_u]  (Eq. 6-7 của bài báo)."""

    def __init__(self, i, o):
        super().__init__()
        self.lin_self = nn.Linear(i, o)
        self.lin_neigh = nn.Linear(i, o, bias=False)

    def forward(self, x, adj):
        return self.lin_self(x) + self.lin_neigh(torch.sparse.mm(adj, x))


class Encoder(nn.Module):
    def __init__(self, in_dim, hidden=(64, 32), kind="gcn", dropout=0.0):
        super().__init__()
        Layer = {"gcn": GCNLayer, "sage": SAGELayer}[kind]
        dims = [in_dim, *hidden]
        self.layers = nn.ModuleList(Layer(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.acts = nn.ModuleList(nn.PReLU(b) for b in dims[1:])
        self.dropout = dropout
        self.out_dim = dims[-1]

    def forward(self, x, adj):
        for i, (layer, act) in enumerate(zip(self.layers, self.acts)):
            if i > 0:
                x = F.dropout(x, self.dropout, self.training)
            x = act(layer(x, adj))
        return x


class FeCoGraph(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=(64, 32), gnn="gcn",
                 proj_hidden=256, proj_dim=128, dropout=0.0):
        super().__init__()
        self.encoder = Encoder(in_dim, hidden, gnn, dropout)
        h = self.encoder.out_dim
        self.projector = nn.Sequential(nn.Linear(h, proj_hidden), nn.ELU(), nn.Linear(proj_hidden, proj_dim))
        self.classifier = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h, num_classes))

    def embed(self, x, adj):
        return self.encoder(x, adj)

    def project(self, h):
        return F.normalize(self.projector(h), dim=1)

    def classify(self, h):
        return self.classifier(h)
