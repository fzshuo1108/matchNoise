import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    csv_path = Path("outputs/mnist.csv")
    save_dir = Path("outputs")
    save_dir.mkdir(exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # 全局字体大小可按需要调整
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 20,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
    })

    # =========================
    # Figure 1: Epsilon vs Epochs
    # =========================
    fig, ax = plt.subplots(figsize=(4.2, 4.2))

    ax.set_facecolor("#f2f2f2")
    fig.patch.set_facecolor("#f2f2f2")
    ax.grid(True, color="#d0d0d0", linewidth=0.8, alpha=0.8)

    ax.plot(
        df["epoch"],
        df["product_epsilon"],
        color="red",
        linewidth=2.0,
        linestyle="-",
        label="product noise via distribution-independent composition",
    )

    ax.plot(
        df["epoch"],
        df["gaussian_epsilon_ma"],
        color="blue",
        linewidth=1.8,
        linestyle="--",
        label="classical Gaussian noise via MA",
    )

    ax.plot(
        df["epoch"],
        df["gaussian_epsilon_clt"],
        color="black",
        linewidth=1.8,
        linestyle="-.",
        label="classical Gaussian noise via CLT",
    )

    ax.plot(
        df["epoch"],
        df["gaussian_epsilon_prv"],
        color="green",
        linewidth=1.8,
        linestyle="--",
        label="classical Gaussian noise via PRV",
    )

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Epsilon")

    ax.legend(loc="upper left", frameon=False)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")

    plt.subplots_adjust(bottom=0.24)

    fig.text(
        0.5,
        0.04,
        "(c) Epsilon vs. epochs",
        ha="center",
        va="center",
        fontsize=20,
    )

    out_path_eps = save_dir / "epsilon_vs_epoch_styled.png"
    plt.savefig(out_path_eps, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure to: {out_path_eps}")

    # =========================
    # Figure 2: Accuracy vs Epochs
    # =========================
    fig, ax = plt.subplots(figsize=(4.2, 4.2))

    ax.set_facecolor("#f2f2f2")
    fig.patch.set_facecolor("#f2f2f2")
    ax.grid(True, color="#d0d0d0", linewidth=0.8, alpha=0.8)

    ax.plot(
        df["epoch"],
        df["product_accuracy"],
        color="red",
        linewidth=2.0,
        linestyle="-",
        label="product noise",
    )

    ax.plot(
        df["epoch"],
        df["gaussian_accuracy"],
        color="blue",
        linewidth=1.8,
        linestyle="--",
        label="classical Gaussian noise",
    )

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")

    ax.legend(loc="lower right", frameon=False)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")

    plt.subplots_adjust(bottom=0.24)

    fig.text(
        0.5,
        0.04,
        "(d) Accuracy vs. epochs",
        ha="center",
        va="center",
        fontsize=20,
    )

    out_path_acc = save_dir / "accuracy_vs_epoch_styled.png"
    plt.savefig(out_path_acc, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure to: {out_path_acc}")


if __name__ == "__main__":
    main()