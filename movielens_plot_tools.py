import os
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 基本配置
# =========================
CSV_FILE = "outputs/movielens.csv"
SAVE_DIR = os.path.dirname(CSV_FILE)

# 读取数据
df = pd.read_csv(CSV_FILE)

# 字体设置
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 26,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

# Epsilon 图线型
DASH_1 = (0, (3, 1, 1, 2))
DASH_2 = (0, (6, 1, 1, 3))
DASH_3 = (0, (9, 1, 1, 4))


# =========================
# Epsilon 图
# =========================
plt.figure(figsize=(8, 6))

plt.plot(
    df["epoch"],
    df["product_epsilon"],
    color="red",
    linewidth=2,
    linestyle="-",
    label="PLRV-relaxed composition"
)

plt.plot(
    df["epoch"],
    df["gaussian_epsilon_ma"],
    color="blue",
    linewidth=2,
    linestyle=DASH_1,
    label="classical Gaussian noise via MA"
)

plt.plot(
    df["epoch"],
    df["gaussian_epsilon_clt"],
    color="black",
    linewidth=2,
    linestyle=DASH_2,
    label="classical Gaussian noise via CLT"
)

plt.plot(
    df["epoch"],
    df["gaussian_epsilon_prv"],
    color="green",
    linewidth=2,
    linestyle=DASH_3,
    label="classical Gaussian noise via PRV"
)

plt.xlabel("Epochs")
plt.ylabel("Epsilon")
# plt.title("MovieLens Privacy Budget")

plt.grid(True, linestyle="--", alpha=0.3)

plt.legend(
    loc="lower right",
    bbox_to_anchor=(1, 0.1),
    frameon=True,
    framealpha=0.9
)

plt.tight_layout()
plt.savefig(
    os.path.join(SAVE_DIR, "movielens_epsilon_plot.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# =========================
# RMSE 图
# =========================
plt.figure(figsize=(8, 6))

plt.plot(
    df["epoch"],
    df["product_test_rmse"],
    color="red",
    linewidth=2,
    linestyle="-",
    label="proposed product noise"
)

plt.plot(
    df["epoch"],
    df["gaussian_test_rmse"],
    color="blue",
    linewidth=2,
    linestyle="--",
    label="classical Gaussian noise"
)

plt.plot(
    df["epoch"],
    df["baseline_test_rmse"],
    color="black",
    linewidth=2,
    linestyle="-.",
    label="non-private baseline"
)

plt.xlabel("Epochs")
plt.ylabel("Test RMSE")
# plt.title("MovieLens Test RMSE")

plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(loc="upper right", frameon=True, framealpha=0.9)

plt.tight_layout()
plt.savefig(
    os.path.join(SAVE_DIR, "movielens_rmse_plot.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()