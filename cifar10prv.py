import argparse
import copy
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from opacus import PrivacyEngine
from opacus.accountants import PRVAccountant
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def convnet(num_classes: int):
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),

        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),

        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),

        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),

        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(128, num_classes, bias=True),
    )


def build_train_loader(batch_size: int, data_root: str):
    augmentations = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]

    train_transform = transforms.Compose(augmentations + normalize)

    train_dataset = CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader


def main():
    parser = argparse.ArgumentParser(description="Compute PRV epsilon per epoch for CIFAR10 without training")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)

    parser.add_argument("--lr", type=float, default=0.25)
    parser.add_argument("--momentum", type=float, default=0.3)

    parser.add_argument("--gaussian-noise-multiplier", type=float, default=0.50)
    parser.add_argument("--max-grad-norm", type=float, default=1.5)

    parser.add_argument("--target-delta", type=float, default=1e-5)

    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)

    # 为了减少 PRV 计算时的内存压力，开放这两个参数
    parser.add_argument("--eps-error", type=float, default=0.1)
    parser.add_argument("--delta-error", type=float, default=1e-7)

    args = parser.parse_args()

    set_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 只需要 train_loader，用来获得和原实验一致的 sample_rate / steps_per_epoch
    train_loader = build_train_loader(
        batch_size=args.batch_size,
        data_root=args.data_root,
    )

    # 这里复刻你原来的 make_private 过程，
    # 目的不是训练，而是拿到和原代码一致的 private train_loader 长度
    base_model = convnet(num_classes=10)
    gaussian_model = copy.deepcopy(base_model)

    gaussian_optimizer = optim.SGD(
        gaussian_model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=0.0,
    )

    gaussian_engine = PrivacyEngine(accountant="rdp")

    gaussian_model, gaussian_optimizer, train_loader_g = gaussian_engine.make_private(
        module=gaussian_model,
        optimizer=gaussian_optimizer,
        data_loader=train_loader,
        noise_multiplier=args.gaussian_noise_multiplier,
        max_grad_norm=args.max_grad_norm,
    )

    # 和你原代码一致：sample_rate = 1.0 / len(train_loader_g)
    steps_per_epoch = len(train_loader_g)
    sample_rate = 1.0 / steps_per_epoch

    print("\n===== PRV Composition Only =====")
    print("dataset=CIFAR10")
    print(f"epochs={args.epochs}")
    print(f"batch_size={args.batch_size}")
    print(f"gaussian_noise_multiplier={args.gaussian_noise_multiplier}")
    print(f"max_grad_norm={args.max_grad_norm}")
    print(f"target_delta={args.target_delta}")
    print(f"steps_per_epoch={steps_per_epoch}")
    print(f"sample_rate={sample_rate}")
    print(f"eps_error={args.eps_error}")
    print(f"delta_error={args.delta_error}")
    print("================================\n")

    prv_accountant = PRVAccountant()

    csv_path = save_dir / "cifar10_prv_epsilon_per_epoch.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "epoch",
            "gaussian_epsilon_prv",
        ])

        total_steps = 0
        for epoch in range(1, args.epochs + 1):
            for _ in range(steps_per_epoch):
                prv_accountant.step(
                    noise_multiplier=args.gaussian_noise_multiplier,
                    sample_rate=sample_rate,
                )
                total_steps += 1

            eps_prv = prv_accountant.get_epsilon(
                delta=args.target_delta,
                eps_error=args.eps_error,
                delta_error=args.delta_error,
            )

            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"total_steps={total_steps} | "
                f"PRV epsilon={eps_prv:.6f}"
            )

            writer.writerow([epoch, eps_prv])
            csv_file.flush()

    print(f"\nPRV epsilon CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()