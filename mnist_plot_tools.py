# import os
# import pandas as pd
# import matplotlib.pyplot as plt
#
#
# # =========================
# # 基本配置
# # =========================
# CSV_FILE = "outputs/mnist.csv"   # ← 改这里
# SAVE_DIR = os.path.dirname(CSV_FILE)
#
# # 读取数据
# df = pd.read_csv(CSV_FILE)
#
# # 字体设置
# plt.rcParams.update({
#     "font.size": 14,
#     "axes.labelsize": 26,
#     "xtick.labelsize": 16,
#     "ytick.labelsize": 16,
#     "legend.fontsize": 14,
# })
#
# # Epsilon 图线型
# DASH_1 = (0, (3, 1, 1, 2))
# DASH_2 = (0, (6, 1, 1, 3))
# DASH_3 = (0, (9, 1, 1, 4))
#
#
# # =========================
# # Epsilon 图
# # =========================
# plt.figure(figsize=(8, 6))
#
# plt.plot(
#     df["epoch"],
#     df["product_epsilon"],
#     color="red",
#     linewidth=2,
#     linestyle="-",
#     label="PLRV-relaxed composition"
# )
#
# plt.plot(
#     df["epoch"],
#     df["gaussian_epsilon_ma"],
#     color="blue",
#     linewidth=2,
#     linestyle=DASH_1,
#     label="classical Gaussian noise via MA"
# )
#
# plt.plot(
#     df["epoch"],
#     df["gaussian_epsilon_clt"],
#     color="black",
#     linewidth=2,
#     linestyle=DASH_2,
#     label="classical Gaussian noise via CLT"
# )
#
# plt.plot(
#     df["epoch"],
#     df["gaussian_epsilon_prv"],
#     color="green",
#     linewidth=2,
#     linestyle=DASH_3,
#     label="classical Gaussian noise via PRV"
# )
#
# plt.xlabel("Epochs")
# plt.ylabel("Epsilon")
# # plt.title("MNIST Privacy Budget")   # ← 改标题
#
# plt.grid(True, linestyle="--", alpha=0.3)
#
# plt.legend(
#     loc="lower right",
#     bbox_to_anchor=(1, 0.15),
#     frameon=True,
#     framealpha=0.9
# )
#
# plt.tight_layout()
# plt.savefig(
#     os.path.join(SAVE_DIR, "mnist_epsilon_plot.png"),  # ← 文件名
#     dpi=300,
#     bbox_inches="tight"
# )
# plt.show()
#
#
# # =========================
# # Accuracy 图
# # =========================
# plt.figure(figsize=(8, 6))
#
# plt.plot(
#     df["epoch"],
#     df["product_accuracy"] * 100,
#     color="red",
#     linewidth=2,
#     linestyle="-",
#     label="proposed product noise"
# )
#
# plt.plot(
#     df["epoch"],
#     df["gaussian_accuracy"] * 100,
#     color="blue",
#     linewidth=2,
#     linestyle="--",
#     label="classical Gaussian noise"
# )
#
# plt.plot(
#     df["epoch"],
#     df["baseline_accuracy"] * 100,
#     color="black",
#     linewidth=2,
#     linestyle="-.",
#     label="non-private baseline"
# )
#
# plt.xlabel("Epochs")
# plt.ylabel("Test Accuracy (%)")
# # plt.title("MNIST Test Accuracy")   # ← 改标题
#
# plt.grid(True, linestyle="--", alpha=0.3)
# plt.legend(loc="lower right", frameon=True, framealpha=0.9)
#
# plt.tight_layout()
# plt.savefig(
#     os.path.join(SAVE_DIR, "mnist_accuracy_plot.png"),  # ← 文件名
#     dpi=300,
#     bbox_inches="tight"
# )
# plt.show()

import os
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 基本配置
# =========================
CSV_FILE = "outputs/mnist_accuracy_only_2026-04-15_13-58-35.csv"
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


# =========================
# Test Accuracy 图
# =========================
plt.figure(figsize=(8, 6))

plt.plot(
    df["epoch"],
    df["product_accuracy"] * 100,
    color="red",
    linewidth=2,
    linestyle="-",
    label="proposed product noise"
)

plt.plot(
    df["epoch"],
    df["gaussian_accuracy"] * 100,
    color="blue",
    linewidth=2,
    linestyle="--",
    label="classical Gaussian noise"
)

plt.plot(
    df["epoch"],
    df["baseline_accuracy"] * 100,
    color="black",
    linewidth=2,
    linestyle="-.",
    label="non-private baseline"
)

plt.xlabel("Epochs")
plt.ylabel("Test Accuracy (%)")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(loc="lower right", frameon=True, framealpha=0.9)

plt.tight_layout()
plt.savefig(
    os.path.join(SAVE_DIR, "mnist_test_accuracy_plot.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# =========================
# Training Accuracy 图
# =========================
plt.figure(figsize=(8, 6))

plt.plot(
    df["epoch"],
    df["product_train_accuracy"] * 100,
    color="red",
    linewidth=2,
    linestyle="-",
    label="proposed product noise"
)

plt.plot(
    df["epoch"],
    df["gaussian_train_accuracy"] * 100,
    color="blue",
    linewidth=2,
    linestyle="--",
    label="classical Gaussian noise"
)

plt.plot(
    df["epoch"],
    df["baseline_train_accuracy"] * 100,
    color="black",
    linewidth=2,
    linestyle="-.",
    label="non-private baseline"
)

plt.xlabel("Epochs")
plt.ylabel("Training Accuracy (%)")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(loc="lower right", frameon=True, framealpha=0.9)

plt.tight_layout()
plt.savefig(
    os.path.join(SAVE_DIR, "mnist_training_accuracy_plot.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()