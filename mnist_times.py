#!/usr/bin/env python3

import argparse
import copy
import csv
import math
import random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from opacus import PrivacyEngine
from product_noise import ProductPrivacyEngine


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)

        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 1)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 1)

        x = x.view(-1, 32 * 4 * 4)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def build_dataloaders(batch_size, data_root):
    transform = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )

    test_set = datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    losses = []
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)
        pred = logits.argmax(dim=1)

        losses.append(loss.item())
        correct += (pred == y).sum().item()
        total += y.size(0)

    avg_loss = float(np.mean(losses)) if losses else 0.0
    avg_acc = correct / total if total > 0 else 0.0
    return avg_loss, avg_acc


def train_one_epoch(model, loader, optimizer, device, epoch=None, tag="model"):
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)
        pred = logits.argmax(dim=1)

        losses.append(loss.item())
        correct += (pred == y).sum().item()
        total += y.size(0)

        loss.backward()
        optimizer.step()

    avg_loss = float(np.mean(losses)) if losses else 0.0
    avg_acc = correct / total if total > 0 else 0.0

    print(f"[{tag}] epoch={epoch} train_loss={avg_loss:.6f} train_acc={avg_acc:.6f}")
    return avg_loss, avg_acc


def get_model_dimension(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def matched_product_sigma_M(model, gaussian_sigma, max_grad_norm):
    M = get_model_dimension(model)
    sigma_M = math.sqrt(M) * gaussian_sigma * max_grad_norm
    return sigma_M, M


def run_one_experiment(args, run_seed):
    set_seed(run_seed)
    device = torch.device(args.device)

    train_loader, test_loader = build_dataloaders(args.batch_size, args.data_root)

    base_model = SampleConvNet()

    gaussian_model = copy.deepcopy(base_model).to(device)
    product_model = copy.deepcopy(base_model).to(device)
    baseline_model = copy.deepcopy(base_model).to(device)

    gaussian_optimizer = optim.SGD(
        gaussian_model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
    )

    product_optimizer = optim.SGD(
        product_model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
    )

    baseline_optimizer = optim.SGD(
        baseline_model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
    )

    product_sigma_M, M = matched_product_sigma_M(
        product_model,
        args.gaussian_noise_multiplier,
        args.max_grad_norm,
    )

    # Gaussian side: keep DP training, but no epsilon collection
    gaussian_engine = PrivacyEngine(accountant="rdp")
    gaussian_model, gaussian_optimizer, train_loader_g = gaussian_engine.make_private(
        module=gaussian_model,
        optimizer=gaussian_optimizer,
        data_loader=train_loader,
        noise_multiplier=args.gaussian_noise_multiplier,
        max_grad_norm=args.max_grad_norm,
    )

    # Product side: keep DP training, but no epsilon collection
    product_engine = ProductPrivacyEngine(
        clipping_norm=args.max_grad_norm,
        delta0=args.delta0,
        M=M,
        k=args.k,
    )

    product_model, product_optimizer, train_loader_p = product_engine.make_private(
        module=product_model,
        optimizer=product_optimizer,
        data_loader=train_loader,
        noise_multiplier=product_sigma_M,
        max_grad_norm=args.max_grad_norm,
    )

    results = []

    print(f"\n========== Run seed = {run_seed} ==========")
    print(f"dataset=MNIST")
    print(f"model={base_model.__class__.__name__}")
    print(f"device={device}")
    print(f"model_dimension_M={M}")
    print(f"product_sigma_M={product_sigma_M:.6f}")

    for epoch in range(1, args.epochs + 1):
        print(f"\nRun {run_seed} | Epoch {epoch}/{args.epochs}")

        g_train_loss, g_train_acc = train_one_epoch(
            model=gaussian_model,
            loader=train_loader_g,
            optimizer=gaussian_optimizer,
            device=device,
            epoch=epoch,
            tag="gaussian",
        )

        p_train_loss, p_train_acc = train_one_epoch(
            model=product_model,
            loader=train_loader_p,
            optimizer=product_optimizer,
            device=device,
            epoch=epoch,
            tag="product",
        )

        b_train_loss, b_train_acc = train_one_epoch(
            model=baseline_model,
            loader=train_loader,
            optimizer=baseline_optimizer,
            device=device,
            epoch=epoch,
            tag="baseline",
        )

        g_test_loss, g_acc = evaluate(gaussian_model, test_loader, device)
        p_test_loss, p_acc = evaluate(product_model, test_loader, device)
        b_test_loss, b_acc = evaluate(baseline_model, test_loader, device)

        print(
            f"[Gaussian] train_acc={g_train_acc:.6f} test_acc={g_acc:.6f}"
        )
        print(
            f"[Product] train_acc={p_train_acc:.6f} test_acc={p_acc:.6f}"
        )
        print(
            f"[Baseline] train_acc={b_train_acc:.6f} test_acc={b_acc:.6f}"
        )

        results.append({
            "epoch": epoch,
            "gaussian_train_accuracy": g_train_acc,
            "gaussian_accuracy": g_acc,
            "product_train_accuracy": p_train_acc,
            "product_accuracy": p_acc,
            "baseline_train_accuracy": b_train_acc,
            "baseline_accuracy": b_acc,
        })

    return results


def average_results(all_runs_results, epochs):
    averaged = []

    for epoch_idx in range(epochs):
        rows = [run[epoch_idx] for run in all_runs_results]

        averaged.append({
            "epoch": rows[0]["epoch"],
            "gaussian_train_accuracy": float(np.mean([r["gaussian_train_accuracy"] for r in rows])),
            "gaussian_accuracy": float(np.mean([r["gaussian_accuracy"] for r in rows])),
            "product_train_accuracy": float(np.mean([r["product_train_accuracy"] for r in rows])),
            "product_accuracy": float(np.mean([r["product_accuracy"] for r in rows])),
            "baseline_train_accuracy": float(np.mean([r["baseline_train_accuracy"] for r in rows])),
            "baseline_accuracy": float(np.mean([r["baseline_accuracy"] for r in rows])),
        })

    return averaged


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)

    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--momentum", type=float, default=0.3)

    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gaussian-noise-multiplier", type=float, default=1.3)

    parser.add_argument("--delta0", type=float, default=1e-12)
    parser.add_argument("--k", type=int, default=40000)

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-runs", type=int, default=5)

    args = parser.parse_args()

    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    temp_model = SampleConvNet()
    _, M = matched_product_sigma_M(
        temp_model,
        args.gaussian_noise_multiplier,
        args.max_grad_norm,
    )
    product_sigma_M = math.sqrt(M) * args.gaussian_noise_multiplier * args.max_grad_norm

    print("\n===== Experiment Configuration =====")
    print("dataset=MNIST")
    print(f"model={temp_model.__class__.__name__}")
    print(f"device={device}")
    print(f"epochs={args.epochs}")
    print(f"batch_size={args.batch_size}")
    print(f"lr={args.lr}")
    print(f"momentum={args.momentum}")
    print(f"max_grad_norm={args.max_grad_norm}")
    print(f"gaussian_sigma={args.gaussian_noise_multiplier}")
    print(f"product_sigma_M={product_sigma_M}")
    print(f"delta0={args.delta0}")
    print(f"k={args.k}")
    print(f"model_dimension_M={M}")
    print(f"num_runs={args.num_runs}")
    print("===================================\n")

    all_runs_results = []
    for run_idx in range(args.num_runs):
        run_seed = args.seed + run_idx
        run_results = run_one_experiment(args, run_seed)
        all_runs_results.append(run_results)

    averaged_results = average_results(all_runs_results, args.epochs)

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = save_dir / f"mnist_accuracy_only_{now}.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "epoch",
            "gaussian_train_accuracy",
            "gaussian_accuracy",
            "product_train_accuracy",
            "product_accuracy",
            "baseline_train_accuracy",
            "baseline_accuracy",
        ])

        for row in averaged_results:
            writer.writerow([
                row["epoch"],
                row["gaussian_train_accuracy"],
                row["gaussian_accuracy"],
                row["product_train_accuracy"],
                row["product_accuracy"],
                row["baseline_train_accuracy"],
                row["baseline_accuracy"],
            ])

    print(f"\nAveraged CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()