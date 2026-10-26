"""
Huấn luyện FeCoGraph (bản tập trung, không federated) cho NIDS few-shot.

Ví dụ:
  # Đúng thiết lập bài báo: đa lớp, 30% nhãn, SupCon + CE, augmentation thích ứng
  python train.py --data data/NF-ToN-IoT-v2.csv --task multi

  # Baseline GCN thuần (chỉ CE)                       -> --loss ce
  # Ablation SSLCon (GRACE thay cho SupCon)           -> --loss sslcon
  # Ablation augmentation đồng đều (GRACE-style)      -> --aug uniform
  # Few-shot "đúng nghĩa": 10 mẫu có nhãn mỗi lớp     -> --k-shot 10
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

from src.augment import Augmentor, undirected
from src.losses import grace_loss, supcon_loss
from src.model import FeCoGraph, build_adj
from src.prepare import add_data_args, prepare


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(p)
    m = p.add_argument_group("mô hình")
    m.add_argument("--gnn", choices=["gcn", "sage"], default="gcn")
    m.add_argument("--hidden", type=int, nargs="+", default=[64, 32])
    m.add_argument("--proj-hidden", type=int, default=256)
    m.add_argument("--proj-dim", type=int, default=128)
    m.add_argument("--dropout", type=float, default=0.0)
    c = p.add_argument_group("contrastive")
    c.add_argument("--loss", choices=["supcon", "sslcon", "ce"], default="supcon")
    c.add_argument("--lam", type=float, default=0.5, help="trọng số CE trong Eq. 12")
    c.add_argument("--tau", type=float, default=0.3)
    c.add_argument("--aug", choices=["adaptive", "uniform"], default="adaptive")
    c.add_argument("--centrality", choices=["degree", "pagerank"], default="degree")
    c.add_argument("--pe", type=float, nargs=2, default=[0.2, 0.4], help="p_e của view 1, 2")
    c.add_argument("--pf", type=float, nargs=2, default=[0.1, 0.2], help="p_f của view 1, 2")
    c.add_argument("--p-tau", type=float, default=0.7)
    c.add_argument("--con-batch", type=int, default=2048, help="số nút / batch khi tính loss contrastive")
    c.add_argument("--balanced-batch", action="store_true",
                   help="lấy mẫu batch SupCon cân bằng theo lớp (không có trong bài báo)")
    t = p.add_argument_group("huấn luyện")
    t.add_argument("--epochs", type=int, default=2000)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--wd", type=float, default=0.0)
    t.add_argument("--class-weight", action="store_true", help="CE có trọng số nghịch tần suất lớp")
    t.add_argument("--eval-every", type=int, default=10)
    t.add_argument("--patience", type=int, default=0, help="dừng sớm sau N lần eval không cải thiện (0 = tắt)")
    t.add_argument("--runs", type=int, default=1, help="số lần chạy với seed khác nhau")
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    t.add_argument("--out", default="results")
    t.add_argument("--tag", default=None, help="tên nhóm kết quả (collect_results.py gộp theo tag)")
    t.add_argument("--save-emb", action="store_true", help="lưu embedding tập test để vẽ t-SNE")
    t.add_argument("--prepare-only", action="store_true", help="chỉ dựng cache dữ liệu rồi thoát")
    t.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 0)) or None,
                   help="số luồng CPU cho torch (mặc định: OMP_NUM_THREADS)")
    return p.parse_args()


def setup_perf(args):
    if args.threads:
        torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        # A100: TF32 cho matmul dày (Linear) nhanh hơn nhiều so với FP32, sai số không đáng kể
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


@torch.no_grad()
def evaluate(model, x, adj, y, idx, num_classes):
    model.eval()
    logits = model.classify(model.embed(x, adj))
    pred = logits[idx].argmax(1).cpu().numpy()
    true = y[idx].cpu().numpy()
    p, r, f, _ = precision_recall_fscore_support(true, pred, labels=range(num_classes),
                                                 average="macro", zero_division=0)
    return dict(acc=accuracy_score(true, pred), precision=p, recall=r, f1=f), pred, true


def sample_batch(idx, y, size, balanced, gen):
    if len(idx) <= size:
        return idx
    if not balanced:
        return idx[torch.randperm(len(idx), generator=gen, device="cpu")[:size].to(idx.device)]
    classes = y[idx].unique()
    per = max(size // len(classes), 1)
    out = []
    for c in classes:
        ci = idx[y[idx] == c]
        out.append(ci[torch.randint(len(ci), (per,), generator=gen, device="cpu").to(idx.device)])
    return torch.cat(out)


def run_once(args, run_id):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    g = prepare(args, verbose=(run_id == 0))
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    x = torch.from_numpy(g.x).to(dev)
    y = torch.from_numpy(g.y).long().to(dev)
    und = undirected(torch.from_numpy(g.edge_index).long().to(dev))
    n, C = g.num_nodes, g.num_classes
    adj = build_adj(und, n, args.gnn)
    tr, va, te = (torch.from_numpy(i).long().to(dev) for i in (g.train_idx, g.val_idx, g.test_idx))

    model = FeCoGraph(x.size(1), C, tuple(args.hidden), args.gnn, args.proj_hidden,
                      args.proj_dim, args.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    aug = Augmentor(x, und, args.aug, args.centrality, tuple(args.pe), tuple(args.pf), args.p_tau) \
        if args.loss != "ce" else None
    cw = None
    if args.class_weight:
        cnt = torch.bincount(y[tr], minlength=C).float().clamp(min=1)
        cw = (cnt.sum() / (C * cnt)).to(dev)
    gen = torch.Generator().manual_seed(args.seed)

    best = dict(f1=-1.0, epoch=0)
    best_state, bad, t0 = None, 0, time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        h = model.embed(x, adj)
        loss_ce = F.cross_entropy(model.classify(h)[tr], y[tr], weight=cw)
        loss_con = torch.zeros((), device=dev)
        if args.loss != "ce":
            x1, und1 = aug.view(x, 0)
            x2, und2 = aug.view(x, 1)
            z1 = model.project(model.embed(x1, build_adj(und1, n, args.gnn)))
            z2 = model.project(model.embed(x2, build_adj(und2, n, args.gnn)))
            if args.loss == "supcon":
                b = sample_batch(tr, y, args.con_batch, args.balanced_batch, gen)
                loss_con = supcon_loss(torch.cat([z1[b], z2[b]]), torch.cat([y[b], y[b]]), args.tau)
            else:  # sslcon: không dùng nhãn, lấy mẫu trên toàn bộ nút
                b = torch.randperm(n, generator=gen)[:args.con_batch].to(dev)
                loss_con = grace_loss(z1[b], z2[b], args.tau)
            loss = (1 - args.lam) * loss_con + args.lam * loss_ce
        else:
            loss = loss_ce
        loss.backward()
        opt.step()

        if ep % args.eval_every == 0 or ep == args.epochs:
            sel_idx = va if len(va) else tr
            m, _, _ = evaluate(model, x, adj, y, sel_idx, C)
            if m["f1"] > best["f1"]:
                best = dict(m, epoch=ep)
                best_state = copy.deepcopy(model.state_dict())
                bad = 0
            else:
                bad += 1
            if ep % (args.eval_every * 10) == 0 or ep == args.epochs:
                print(f"  ep {ep:5d} | loss {loss.item():.4f} (ce {loss_ce.item():.4f}, con {loss_con.item():.4f}) "
                      f"| val F1 {m['f1']:.4f} | best {best['f1']:.4f}@{best['epoch']} | {time.time() - t0:.0f}s")
            if args.patience and bad >= args.patience:
                print(f"  dừng sớm ở epoch {ep}")
                break

    model.load_state_dict(best_state)
    test, pred, true = evaluate(model, x, adj, y, te, C)
    train_time = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated(dev) / 2**30 if dev.type == "cuda" else 0.0
    print(f"  thời gian {train_time:.0f}s | VRAM đỉnh {peak_gb:.2f} GiB")
    report = classification_report(true, pred, labels=range(C), target_names=g.class_names,
                                   digits=4, zero_division=0, output_dict=True)
    print(classification_report(true, pred, labels=range(C), target_names=g.class_names,
                                digits=4, zero_division=0))
    if args.save_emb:
        model.eval()
        with torch.no_grad():
            emb = model.embed(x, adj)[te].cpu().numpy()
        np.savez_compressed(os.path.join(args.out, f"emb_run{run_id}.npz"), emb=emb, y=true,
                            raw=g.x[g.test_idx], class_names=np.array(g.class_names, dtype=object))
    return dict(test=test, best_val=best, per_class=report, seed=args.seed,
                train_time_s=train_time, peak_vram_gib=peak_gb,
                num_nodes=n, num_edges=int(und.size(1)))


def main():
    args = get_args()
    os.makedirs(args.out, exist_ok=True)
    setup_perf(args)
    if args.prepare_only:
        prepare(args)
        return
    print(f"[cfg] loss={args.loss} aug={args.aug} gnn={args.gnn} lam={args.lam} tau={args.tau} "
          f"task={args.task} label_ratio={args.label_ratio} k_shot={args.k_shot} device={args.device}")
    base_seed, results = args.seed, []
    for r in range(args.runs):
        args.seed = base_seed + r
        print(f"\n===== run {r + 1}/{args.runs} (seed {args.seed}) =====")
        results.append(run_once(args, r))
        t = results[-1]["test"]
        print(f"[test] acc {t['acc']:.4f} | P {t['precision']:.4f} | R {t['recall']:.4f} | F1 {t['f1']:.4f}")
    args.seed = base_seed

    keys = ["acc", "precision", "recall", "f1"]
    summary = {k: (float(np.mean([r["test"][k] for r in results])),
                   float(np.std([r["test"][k] for r in results]))) for k in keys}
    print("\n===== tổng kết (macro, test) =====")
    for k, (mu, sd) in summary.items():
        print(f"  {k:<9s} {100 * mu:6.2f} ± {100 * sd:.2f}")

    tag = args.tag or (f"{args.task}_{args.loss}_{args.aug}_{args.gnn}_" +
                       (f"k{args.k_shot}" if args.k_shot else f"r{args.label_ratio}"))
    seeds = "-".join(str(r["seed"]) for r in results)
    path = os.path.join(args.out, f"{tag}__s{seeds}__{time.strftime('%Y%m%d-%H%M%S')}_{os.getpid()}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(tag=tag, method=f"GNN-{args.loss}", config=vars(args), summary=summary, runs=results), f, indent=2,
                  ensure_ascii=False, default=float)
    print(f"Đã lưu {path}")


if __name__ == "__main__":
    main()
