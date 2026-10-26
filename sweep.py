"""
Bộ lập lịch chạy song song nhiều thí nghiệm trên nhiều GPU (một pod Kubeflow hoặc nhiều pod).

Vì sao song song hoá theo JOB thay vì DDP cho từng lần chạy?
  Ở quy mô bài báo (2% dữ liệu, ~0.3-0.4 triệu nút) một lần chạy chỉ dùng vài GiB VRAM
  và GPU không bận hết; chia một lần chạy cho nhiều GPU tốn thời gian đồng bộ hơn là tính.
  Trong khi đó cả bộ thí nghiệm có ~100 lần chạy độc lập (seed x ablation x tỉ lệ nhãn x λ/τ).
  -> Mỗi job 1 GPU, nhiều job / GPU, các GPU chạy song song: tăng tốc gần tuyến tính.

Tính năng:
  - Tự phát hiện GPU, chạy --jobs-per-gpu job trên mỗi GPU (CUDA_VISIBLE_DEVICES riêng từng job).
  - Chia CPU: mỗi job nhận cpus_per_gpu / jobs_per_gpu luồng (OMP/MKL/torch).
  - Dựng cache dữ liệu MỘT lần trước khi phát job (đọc CSV + line graph dùng chung).
  - Resume: job xong được đánh dấu trong <state>/; chạy lại lệnh sẽ bỏ qua job đã xong
    (hữu ích khi pod Kubeflow bị khởi động lại).
  - --shard i --num-shards n: chia danh sách job cho nhiều pod (Indexed Job).

Ví dụ (1 pod, 4 x A100, 11 core / GPU):
  python sweep.py --preset paper ablation --data data/NF-ToN-IoT-v2.csv --seeds 0 1 2 \\
                  --jobs-per-gpu 3 --cpus-per-gpu 11
  python collect_results.py
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import signal
import subprocess
import sys
import time

PY = sys.executable


# ----------------------------------------------------------------------------
# Danh sách thí nghiệm
# ----------------------------------------------------------------------------
def build_jobs(a) -> list[tuple[str, list[str]]]:
    jobs = []
    extra_gnn = shlex.split(a.extra_gnn)
    extra_all = shlex.split(a.extra)
    for data in a.data:
        ds = os.path.splitext(os.path.basename(data))[0]
        base = ["--data", data, *extra_all]

        def gnn(tag, *opts, seed):
            jobs.append((f"{ds}__{tag}__s{seed}",
                         [PY, "train.py", *base, *opts, "--seed", str(seed), "--runs", "1",
                          "--tag", f"{ds}__{tag}", *extra_gnn]))

        def ml(tag, *opts, seed):
            jobs.append((f"{ds}__{tag}__s{seed}",
                         [PY, "baselines_ml.py", *base, *opts, "--seed", str(seed), "--runs", "1",
                          "--tag", f"{ds}__{tag}", *(["--xgb-gpu"] if a.xgb_gpu else [])]))

        for s in a.seeds:
            if "paper" in a.preset:                       # Bảng III
                for task in ["binary", "multi"]:
                    ml(f"{task}_ml_r0.3", "--task", task, seed=s)
                    gnn(f"{task}_gcn-ce_r0.3", "--task", task, "--loss", "ce", seed=s)
                    gnn(f"{task}_fecograph_r0.3", "--task", task, "--loss", "supcon", seed=s)
            if "ablation" in a.preset:                    # Bảng VI + ablation bổ sung
                gnn("multi_sslcon_r0.3", "--task", "multi", "--loss", "sslcon", seed=s)
                gnn("multi_fecograph-uniformaug_r0.3", "--task", "multi", "--aug", "uniform", seed=s)
                gnn("multi_fecograph-sage_r0.3", "--task", "multi", "--gnn", "sage", seed=s)
            if "label" in a.preset:                       # Fig. 7
                for r in [0.1, 0.5, 0.7] + ([] if "paper" in a.preset else [0.3]):
                    gnn(f"multi_fecograph_r{r}", "--task", "multi", "--label-ratio", str(r), seed=s)
                    gnn(f"multi_gcn-ce_r{r}", "--task", "multi", "--loss", "ce", "--label-ratio", str(r), seed=s)
            if "kshot" in a.preset:                       # few-shot đúng nghĩa
                for k in [5, 10, 20, 50]:
                    gnn(f"multi_fecograph_k{k}", "--task", "multi", "--k-shot", str(k), seed=s)
                    gnn(f"multi_gcn-ce_k{k}", "--task", "multi", "--loss", "ce", "--k-shot", str(k), seed=s)
                    ml(f"multi_ml_k{k}", "--task", "multi", "--k-shot", str(k), seed=s)
            if "sensitivity" in a.preset:                 # Bảng IV, V
                for lam in [0.03, 0.05, 0.07, 0.3, 0.7]:
                    gnn(f"multi_fecograph_lam{lam}", "--task", "multi", "--lam", str(lam), seed=s)
                for tau in [0.1, 0.5, 0.7]:
                    gnn(f"multi_fecograph_tau{tau}", "--task", "multi", "--tau", str(tau), seed=s)

    if a.commands_file:                                    # lệnh tự viết, mỗi dòng một lệnh
        with open(a.commands_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    cmd = shlex.split(line)
                    if cmd[0] in ("python", "python3"):
                        cmd[0] = PY
                    name = "custom__" + hashlib.md5(line.encode()).hexdigest()[:10]
                    jobs.append((name, cmd))
    # bỏ trùng tên (vd. label ratio 0.3 đã có trong paper)
    seen, out = set(), []
    for n, c in jobs:
        if n not in seen:
            seen.add(n)
            out.append((n, c))
    return out


# ----------------------------------------------------------------------------
# Cache dữ liệu: dựng 1 lần cho mỗi cấu hình dữ liệu khác nhau
# ----------------------------------------------------------------------------
CACHE_ARGS = ["--data", "--sample-frac", "--host-key", "--max-clique", "--k-sample", "--data-seed", "--cache-dir"]


def cache_signature(cmd: list[str]) -> tuple:
    """Trích các tham số ảnh hưởng tới cache (flows + line graph) từ một lệnh."""
    sig, i = [], 0
    while i < len(cmd):
        if cmd[i] in CACHE_ARGS:
            j = i + 1
            vals = []
            while j < len(cmd) and not cmd[j].startswith("--"):
                vals.append(cmd[j])
                j += 1
            sig.append((cmd[i], *vals))
            i = j
        else:
            i += 1
    return tuple(sorted(sig))


def warm_caches(jobs, a, log_dir):
    sigs = []
    for _, cmd in jobs:
        s = cache_signature(cmd)
        if s and s not in sigs:
            sigs.append(s)
    for k, s in enumerate(sigs):
        args = [x for item in s for x in item]
        cmd = [PY, "-m", "src.prepare", *args]
        print(f"[cache {k + 1}/{len(sigs)}] {' '.join(args)}")
        if a.dry_run:
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        with open(os.path.join(log_dir, f"_prepare_{k}.log"), "w") as lf:
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        if rc != 0:
            sys.exit(f"Dựng cache thất bại, xem {log_dir}/_prepare_{k}.log")


# ----------------------------------------------------------------------------
# Lập lịch
# ----------------------------------------------------------------------------
def detect_gpus() -> list[str]:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is not None and vis.strip():
        return [g.strip() for g in vis.split(",") if g.strip()]
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        return [str(i) for i, line in enumerate(out.strip().splitlines()) if line.startswith("GPU")]
    except (OSError, subprocess.CalledProcessError):
        return []


def run(jobs, a):
    os.makedirs(a.state_dir, exist_ok=True)
    os.makedirs(a.log_dir, exist_ok=True)
    gpus = detect_gpus() if a.gpus == "auto" else [g for g in a.gpus.split(",") if g != ""]
    if not gpus:
        print("Không thấy GPU -> chạy trên CPU với 1 slot")
        gpus = [""]
    slots = [(g, k) for g in gpus for k in range(a.jobs_per_gpu)]
    threads = max(1, a.cpus_per_gpu // a.jobs_per_gpu)
    done = lambda n: os.path.exists(os.path.join(a.state_dir, n + ".done"))  # noqa: E731
    pending = [(n, c) for n, c in jobs if a.rerun or not done(n)]
    print(f"[sweep] {len(jobs)} job ({len(jobs) - len(pending)} đã xong, bỏ qua) | GPU {gpus} | "
          f"{a.jobs_per_gpu} job/GPU | {threads} luồng CPU/job")
    if a.dry_run:
        for n, c in pending:
            print(f"  {n}: {' '.join(c)}")
        return 0

    running: dict[tuple, tuple] = {}            # slot -> (name, proc, t0, logfile, attempt)
    attempts: dict[str, int] = {}
    failed: list[str] = []
    queue = list(pending)
    t_start = time.time()

    def stop_all(*_):
        for _, (n, p, *_rest) in running.items():
            p.terminate()
        sys.exit(130)
    signal.signal(signal.SIGINT, stop_all)
    signal.signal(signal.SIGTERM, stop_all)

    while queue or running:
        for slot in slots:
            if slot in running or not queue:
                continue
            name, cmd = queue.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=slot[0], OMP_NUM_THREADS=str(threads),
                       MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
                       PYTHONUNBUFFERED="1")
            lf = open(os.path.join(a.log_dir, name + ".log"), "w")
            lf.write(f"# GPU={slot[0]} threads={threads}\n# {' '.join(cmd)}\n")
            lf.flush()
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
            attempts[name] = attempts.get(name, 0) + 1
            running[slot] = (name, p, time.time(), lf, cmd)
            print(f"[start] GPU{slot[0]}#{slot[1]}  {name}")
        time.sleep(a.poll)
        for slot in list(running):
            name, p, t0, lf, cmd = running[slot]
            rc = p.poll()
            if rc is None:
                continue
            lf.close()
            del running[slot]
            dt = time.time() - t0
            if rc == 0:
                open(os.path.join(a.state_dir, name + ".done"), "w").write(f"{dt:.0f}s\n")
                n_done = sum(done(n) for n, _ in jobs)
                print(f"[done ] {name} ({dt:.0f}s) | {n_done}/{len(jobs)} | đã chạy {(time.time() - t_start) / 60:.1f} phút")
            elif attempts[name] <= a.retries:
                print(f"[retry] {name} (mã lỗi {rc}), thử lại lần {attempts[name]}")
                queue.append((name, cmd))
            else:
                failed.append(name)
                print(f"[FAIL ] {name} (mã lỗi {rc}) -> xem {a.log_dir}/{name}.log")

    print(f"\n[sweep] xong trong {(time.time() - t_start) / 60:.1f} phút, {len(failed)} job lỗi")
    for n in failed:
        print(f"  - {n}")
    return 1 if failed else 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", nargs="*", default=[],
                   choices=["paper", "ablation", "label", "kshot", "sensitivity"])
    p.add_argument("--data", nargs="*", default=[], help="một hay nhiều CSV; mỗi file là một bộ dữ liệu riêng")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--extra", default="", help='tham số thêm cho MỌI job, vd. "--sample-frac 0.05 --cache-dir /pvc/cache"')
    p.add_argument("--extra-gnn", default="", help='tham số thêm chỉ cho train.py, vd. "--epochs 1000"')
    p.add_argument("--xgb-gpu", action="store_true")
    p.add_argument("--commands-file", default=None, help="file lệnh tự viết, mỗi dòng một lệnh")
    p.add_argument("--gpus", default="auto", help='"auto" hoặc danh sách, vd. "0,1,2,3"')
    p.add_argument("--jobs-per-gpu", type=int, default=2)
    p.add_argument("--cpus-per-gpu", type=int, default=11)
    p.add_argument("--shard", type=int, default=int(os.environ.get("JOB_COMPLETION_INDEX", 0)))
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--poll", type=float, default=2.0)
    p.add_argument("--state-dir", default="sweep_state")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--rerun", action="store_true", help="chạy lại cả job đã xong")
    p.add_argument("--dry-run", action="store_true", help="chỉ in danh sách job")
    a = p.parse_args()

    jobs = build_jobs(a)
    jobs = jobs[a.shard::a.num_shards]
    if not jobs:
        sys.exit("Không có job nào: cần --preset (kèm --data) hoặc --commands-file")
    os.makedirs(a.log_dir, exist_ok=True)
    warm_caches(jobs, a, a.log_dir)
    sys.exit(run(jobs, a))


if __name__ == "__main__":
    main()
