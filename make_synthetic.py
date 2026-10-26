"""
Sinh một CSV giả lập có ĐÚNG schema NetFlow v2 (43 đặc trưng + Label + Attack)
để chạy thử pipeline trước khi dùng dữ liệu thật.

Dữ liệu giả có cấu trúc đồ thị đơn giản: mỗi loại tấn công dùng một nhóm
attacker IP và nhắm vào một nhóm victim, nên line graph có tín hiệu topo.

    python make_synthetic.py --rows 20000 --out data/synthetic_nf_v2.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

COLUMNS = [
    "IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT", "PROTOCOL", "L7_PROTO",
    "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS", "TCP_FLAGS", "CLIENT_TCP_FLAGS",
    "SERVER_TCP_FLAGS", "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
    "MIN_TTL", "MAX_TTL", "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN",
    "MAX_IP_PKT_LEN", "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS", "RETRANSMITTED_OUT_BYTES",
    "RETRANSMITTED_OUT_PKTS", "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES", "TCP_WIN_MAX_IN",
    "TCP_WIN_MAX_OUT", "ICMP_TYPE", "ICMP_IPV4_TYPE", "DNS_QUERY_ID", "DNS_QUERY_TYPE",
    "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE", "Label", "Attack",
]

# (tên lớp, tỉ lệ, độ lệch đặc trưng, port đích ưa dùng, protocol)
CLASSES = [
    ("Benign", 0.60, 0.0, [80, 443, 53], 6),
    ("DDoS", 0.15, 1.5, [80], 6),
    ("DoS", 0.10, 1.2, [80], 6),
    ("scanning", 0.08, 0.8, list(range(20, 1024)), 6),
    ("injection", 0.04, 0.6, [80, 8080], 6),
    ("xss", 0.025, 0.5, [80, 8080], 6),
    ("mitm", 0.005, 0.3, [53], 17),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--out", default="data/synthetic_nf_v2.csv")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    rows = []
    for ci, (name, frac, shift, ports, proto) in enumerate(CLASSES):
        n = max(int(a.rows * frac), 20)
        if name == "Benign":
            src = [f"192.168.1.{rng.integers(1, 200)}" for _ in range(n)]
            dst = [f"10.0.0.{rng.integers(1, 30)}" for _ in range(n)]
        else:
            attackers = [f"172.16.{ci}.{i}" for i in range(1, 6 if name != "DDoS" else 40)]
            victims = [f"10.0.0.{i}" for i in range(1, 5)]
            src = list(rng.choice(attackers, n))
            dst = list(rng.choice(victims, n))
        base = rng.lognormal(mean=5 + shift, sigma=1.0, size=(n, 39))
        df = pd.DataFrame(base, columns=COLUMNS[4:43])
        df.insert(0, "IPV4_SRC_ADDR", src)
        df.insert(1, "L4_SRC_PORT", rng.integers(1024, 65535, n))
        df.insert(2, "IPV4_DST_ADDR", dst)
        df.insert(3, "L4_DST_PORT", rng.choice(ports, n))
        df["PROTOCOL"] = proto
        df["L7_PROTO"] = rng.choice([7.0, 91.0, 5.0], n)
        df["ICMP_TYPE"] = rng.choice([0, 771, 2048], n)
        df["ICMP_IPV4_TYPE"] = rng.choice([0, 3, 8], n)
        df["DNS_QUERY_TYPE"] = rng.choice([0, 1, 28], n)
        df["FTP_COMMAND_RET_CODE"] = rng.choice([0, 200, 230, 530], n)
        for c in ["TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS"]:
            df[c] = rng.choice([2, 18, 24, 27] if name == "Benign" else [2, 4, 20], n)
        df["Label"] = int(name != "Benign")
        df["Attack"] = name
        rows.append(df)

    out = pd.concat(rows, ignore_index=True).sample(frac=1.0, random_state=a.seed)
    # chèn vài giá trị lỗi để thử bước làm sạch
    out.iloc[:5, out.columns.get_loc("SRC_TO_DST_SECOND_BYTES")] = np.inf
    out.iloc[5:10, out.columns.get_loc("DNS_TTL_ANSWER")] = np.nan
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out[COLUMNS].to_csv(a.out, index=False)
    print(f"Đã ghi {len(out):,} dòng -> {a.out}")
    print(out["Attack"].value_counts().to_string())


if __name__ == "__main__":
    main()
