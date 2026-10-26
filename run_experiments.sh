#!/usr/bin/env bash
# Tái hiện các thí nghiệm chính trên MỘT máy/pod nhiều GPU.
# Sửa đường dẫn dữ liệu; sweep.py tự phát hiện GPU, chia job, và có thể chạy lại để resume.
set -e
DATA=${DATA:-"data/NF-ToN-IoT-v2.csv data/NF-BoT-IoT-v2.csv data/NF-CSE-CIC-IDS2018-v2.csv"}
SEEDS=${SEEDS:-"0 1 2"}
JOBS_PER_GPU=${JOBS_PER_GPU:-3}

# 0. Kiểm thử nhanh (vài phút)
python -m pytest -q tests
python make_synthetic.py --rows 20000 --out data/synthetic_nf_v2.csv
python train.py --data data/synthetic_nf_v2.csv --sample-frac 1.0 --epochs 200 --task multi

# 1. Toàn bộ thí nghiệm, song song trên mọi GPU
#    paper       : Bảng III (ML + GCN-CE + FeCoGraph, nhị phân và đa lớp, 30% nhãn)
#    ablation    : Bảng VI (SSLCon) + augmentation đồng đều + encoder SAGE
#    label       : Fig. 7 (10/30/50/70% nhãn)
#    kshot       : few-shot đúng nghĩa (5/10/20/50 mẫu mỗi lớp)
#    sensitivity : Bảng IV-V (λ, τ)
python sweep.py --preset paper ablation label kshot sensitivity \
    --data $DATA --seeds $SEEDS --jobs-per-gpu $JOBS_PER_GPU --cpus-per-gpu 11 --xgb-gpu

# 2. Bảng tổng hợp (mean ± std qua seed) -> summary/summary.md, summary.csv
python collect_results.py

# 3. Fig. 6: t-SNE (một lần chạy, 1 GPU)
python train.py --data data/NF-CSE-CIC-IDS2018-v2.csv --task multi --save-emb --out results/tsne
python plot_tsne.py results/tsne/emb_run0.npz --out results/tsne/tsne.png
