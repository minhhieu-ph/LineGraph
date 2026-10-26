# FeCoGraph (bản tập trung): few-shot NIDS bằng GNN trên line graph + label-aware contrastive learning

Cài đặt lại bài báo **FeCoGraph** (Mao et al., IEEE TIFS 2025), **bỏ phần Federated Learning**.
Giữ nguyên: dựng line graph từ NetFlow, adaptive graph augmentation (GCA), encoder GCN 2 lớp,
projector + classifier, hàm mất mát `L = (1-λ)·L_supcon + λ·L_ce`, đánh giá macro.

GCN được viết bằng `torch.sparse` thuần, **không cần PyG hay DGL**.

## Cấu trúc

```
src/data.py      đọc NetFlow v2, downsample, target encoding (chống rò rỉ), scaler, dựng line graph
src/splits.py    chia few-shot: theo tỉ lệ nhãn (--label-ratio) hoặc K mẫu/lớp (--k-shot)
src/prepare.py   ghép pipeline, cache 2 tầng (flows đã downsample + line graph) dùng chung mọi lần chạy
src/augment.py   adaptive augmentation (Eq. 1-5), có chế độ uniform để ablation
src/model.py     Encoder GCN/SAGE, Projector, Classifier, ma trận kề chuẩn hoá
src/losses.py    SupCon (Eq. 8-9) và GRACE/SSLCon (ablation Bảng VI)
train.py         huấn luyện + chọn epoch theo val macro-F1 + báo cáo test
baselines_ml.py  KNN, AdaBoost, DT, RF, ExtraTrees, XGBoost trên cùng split
plot_tsne.py     tái hiện Fig. 6
make_synthetic.py  sinh CSV giả lập đúng schema NetFlow v2 để chạy thử
tests/           kiểm thử loss/adj/augment so với cài đặt vòng lặp
sweep.py         lập lịch chạy song song nhiều job trên nhiều GPU, resume, chia shard cho nhiều pod
collect_results.py  gộp kết quả theo tag: mean ± std qua seed -> summary/summary.md, .csv
run_experiments.sh  lệnh tái hiện Bảng III–VI, Fig. 6–7 và vài ablation bổ sung
k8s/sweep-job.yaml  ví dụ Indexed Job cho Kubeflow (nhiều pod)
```

## Chạy trên nhiều GPU (Kubeflow, A100 40 GB, 11 core + 128 GiB / GPU)

**Song song hoá theo job, không DDP.** Ở quy mô bài báo (2% ≈ 0,3–0,4 triệu nút) một lần chạy chỉ
dùng vài GiB VRAM; chia một lần chạy cho nhiều GPU chủ yếu tốn thời gian đồng bộ. Trong khi đó bộ
thí nghiệm đầy đủ có ~105 lần chạy độc lập cho mỗi bộ dữ liệu (3 seed). `sweep.py` phát mỗi lần chạy
thành một job trên một GPU, nhiều job trên một GPU, nên tăng tốc gần tuyến tính theo số GPU.

```bash
# Một pod / Notebook nhiều GPU: tự phát hiện GPU
python sweep.py --preset paper ablation label kshot sensitivity \
    --data data/NF-ToN-IoT-v2.csv --seeds 0 1 2 --jobs-per-gpu 3 --cpus-per-gpu 11 --xgb-gpu
python sweep.py ... --dry-run          # xem trước danh sách job
python sweep.py ...                    # chạy lại cùng lệnh = resume, bỏ qua job đã xong
python collect_results.py              # bảng tổng hợp

# Nhiều pod: xem k8s/sweep-job.yaml (Indexed Job + --num-shards, cache/kết quả trên PVC dùng chung)
```

Chọn `--jobs-per-gpu`: chạy thử vài job rồi xem cột `vram_gib` trong `summary/summary.md`
(VRAM đỉnh mỗi job). Giữ tổng VRAM ≤ ~35 GiB/GPU và RAM ≤ 128 GiB/GPU. Mỗi job nhận
`cpus_per_gpu / jobs_per_gpu` luồng CPU, nên mặc định 11 core được chia đều, không tranh chấp.

Các tối ưu khác đã có sẵn:
- **Cache 2 tầng.** CSV 10+ GB chỉ đọc một lần, line graph chỉ dựng một lần, rồi dùng chung cho mọi
  seed/tỉ lệ nhãn/ablation. `sweep.py` dựng cache trước rồi mới phát job.
- **Line graph vector hoá** theo kích thước nhóm host: 2 triệu flow dựng trong khoảng 10 giây.
- **TF32 trên A100** cho các lớp Linear, cùng giới hạn luồng CPU mỗi job.
- **XGBoost trên GPU** (`--xgb-gpu`).

Nếu muốn vượt quy mô bài báo (vd. 20–100% dữ liệu, hàng chục triệu nút), full-batch sẽ không còn
vừa 40 GB. Khi đó cần huấn luyện mini-batch với lấy mẫu láng giềng + DDP; phần này chưa có trong code.

## Cài đặt và chạy thử

```bash
pip install -r requirements.txt          # torch: chọn bản CUDA tại pytorch.org
python -m pytest -q tests                # phải pass hết trước khi chạy thật
python make_synthetic.py
python train.py --data data/synthetic_nf_v2.csv --sample-frac 1.0 --epochs 200
```

Dữ liệu thật: NF-BoT-IoT-v2, NF-ToN-IoT-v2, NF-CSE-CIC-IDS2018-v2 (Sarhan et al., Đại học Queensland).
Đặt CSV vào `data/`. Lần đầu sẽ đọc theo chunk và downsample 2% như bài báo; đồ thị được cache trong `cache/`.

```bash
python train.py --data data/NF-ToN-IoT-v2.csv --task multi --runs 3          # FeCoGraph
python train.py --data data/NF-ToN-IoT-v2.csv --task multi --loss ce         # baseline GCN
python train.py --data data/NF-ToN-IoT-v2.csv --task multi --k-shot 10       # 10-shot thật sự
python baselines_ml.py --data data/NF-ToN-IoT-v2.csv --task multi
```

## Ánh xạ bài báo → code

| Bài báo | Code |
|---|---|
| Gộp IP+port, điền NaN/inf, target encoding, StandardScaler (V-A.2) | `src/data.py` |
| Line graph: flow → nút, chung host → cạnh (IV-B.1) | `build_line_graph` |
| Adaptive augmentation, Eq. 1–5 | `src/augment.py` |
| Encoder 2 lớp GCN 64→32, projector ẩn 256, classifier MLP (IV-B.3, V-A.5) | `src/model.py` |
| Label-aware contrastive, Eq. 8–9 | `supcon_loss` |
| CE + loss tổng Eq. 10–12 | `train.py` |
| Adam, lr 0.001, 2000 epoch, 30% nhãn, macro metrics | tham số mặc định của `train.py` |
| SSLCon (Bảng VI) | `--loss sslcon` |

## Những chỗ bài báo không nói rõ, và lựa chọn ở đây

Cần ghi rõ các điểm này trong đề án, vì chúng có thể làm kết quả lệch so với bài báo.

1. **Line graph với host hub.** Line graph chính xác có d(d−1)/2 cạnh cho mỗi host bậc d. Host như
   `victim:80` với 100k flow sẽ cho ~5·10⁹ cạnh, không lưu nổi. Host có ≤ `--max-clique` (100) flow
   được nối đầy đủ; host lớn hơn thì mỗi flow nối với `--k-sample` (10) flow ngẫu nhiên cùng host.
   Bài báo không nói họ xử lý thế nào.
2. **Gộp IP+port.** Theo bài báo (`--host-key ip_port`). Port nguồn thường là ephemeral nên nhiều flow
   chỉ nối qua phía đích; thử `--host-key ip` để so sánh.
3. **Target encoding.** Mã hoá theo tỉ lệ độc hại, chỉ dùng nhãn của tập train, **out-of-fold 5 lần**,
   bỏ qua cột có >1000 giá trị. Nếu không làm OOF, cột có nhiều giá trị sẽ chứa luôn nhãn: train 100%,
   test sụp đổ (lỗi này đã gặp và sửa khi kiểm thử). Danh sách cột phân loại nằm trong `DEFAULT_CATEGORICAL`.
4. **Split.** Bài báo chia 30% train / 70% test, có nhắc "validation" khi chọn siêu tham số nhưng không mô tả.
   Ở đây: 30% có nhãn, 10% (tổng) làm val để chọn epoch, 60% còn lại là test. Một line graph duy nhất (transductive).
5. **Tham số augmentation** `p_e=(0.2, 0.4)`, `p_f=(0.1, 0.2)`, `p_τ=0.7`, độ trung tâm = bậc: lấy theo GCA,
   bài báo không công bố.
6. **SupCon chỉ tính trên nút có nhãn**, mỗi epoch lấy 2048 nút (`--con-batch`) vì bài báo nói tính theo batch
   để tránh hết bộ nhớ.
7. **Eq. 9 trong bài có lỗi chỉ số** (tổng lặp hai lần, `1[k≠i]` nhưng dùng `z_a`). Code cài theo SupCon gốc của Khosla et al.
8. **λ, τ mặc định 0.5 / 0.3** (tốt nhất trên ToN-IoT theo Bảng IV–V). Bộ khác nên quét lại.
9. **Downsample** dùng Bernoulli từng dòng (kỳ vọng đúng tỉ lệ mỗi lớp) để lớp cực hiếm như Theft không bị làm tròn về 0.

## Tuỳ chọn bổ sung (không có trong bài báo)

- `--k-shot K`: few-shot thật sự, K mẫu có nhãn mỗi lớp.
- `--aug uniform`: ablation augmentation thích ứng vs. đồng đều.
- `--balanced-batch`, `--class-weight`: hỗ trợ lớp hiếm.
- `--gnn sage`: encoder kiểu GraphSAGE (Eq. 6–7).
- `--log1p`: log-transform các cột đuôi dài trước khi chuẩn hoá.
