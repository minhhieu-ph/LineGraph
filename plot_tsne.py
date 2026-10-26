"""
Tái hiện Fig. 6: t-SNE của đặc trưng thô vs. embedding của encoder.
Cần chạy train.py với --save-emb trước.

  python plot_tsne.py results/emb_run0.npz --max-points 5000
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--max-points", type=int, default=5000)
    ap.add_argument("--out", default="tsne.png")
    a = ap.parse_args()
    z = np.load(a.npz, allow_pickle=True)
    y, names = z["y"], list(z["class_names"])
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y), min(a.max_points, len(y)), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, key, title in [(axes[0], "raw", "(a) Đặc trưng flow thô"), (axes[1], "emb", "(b) Embedding encoder")]:
        p = TSNE(n_components=2, init="pca", random_state=0, perplexity=30).fit_transform(z[key][idx])
        for c in np.unique(y[idx]):
            m = y[idx] == c
            ax.scatter(p[m, 0], p[m, 1], s=3, label=names[c])
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
    axes[1].legend(markerscale=4, fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(a.out, dpi=150)
    print(f"Đã lưu {a.out}")


if __name__ == "__main__":
    main()
