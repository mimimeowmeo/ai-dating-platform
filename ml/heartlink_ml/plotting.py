"""畫圖工具（無視窗後端，直接存 PNG）。自動挑一個能顯示中文的字型。"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402


def _setup_cjk_font() -> None:
    candidates = ["PingFang TC", "Heiti TC", "Arial Unicode MS", "Noto Sans CJK TC", "Microsoft JhengHei"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name] + list(plt.rcParams["font.sans-serif"])
            break
    plt.rcParams["axes.unicode_minus"] = False


_setup_cjk_font()


def scatter_2d(emb, labels, title, path, noise_label=-1) -> None:
    labels = np.asarray(labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    for u in np.unique(labels):
        m = labels == u
        is_noise = u == noise_label
        ax.scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.5,
                   c="lightgray" if is_noise else None,
                   label="雜訊" if is_noise else f"群 {u}（{m.sum()}）")
    ax.set_title(title)
    ax.legend(markerscale=3, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def line_plot(x, series: dict, title, xlabel, ylabel, path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, ys in series.items():
        ax.plot(x, ys, marker="o", label=name)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def dendrogram_plot(Z, title, path, p=40) -> None:
    from scipy.cluster.hierarchy import dendrogram

    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z, truncate_mode="lastp", p=p, ax=ax, no_labels=True)
    ax.set_title(title)
    ax.set_ylabel("Jaccard 距離")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def bar_plot(names, values, title, ylabel, path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(names, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
