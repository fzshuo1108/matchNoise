import argparse
import copy
import csv
import math
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from opacus import PrivacyEngine
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from product_noise import ProductPrivacyEngine


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


def accuracy(preds, labels):
    return (preds == labels).mean()


def build_dataloaders(batch_size: int, batch_size_test: int, data_root: str):
    augmentations = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]

    train_transform = transforms.Compose(augmentations + normalize)
    test_transform = transforms.Compose(normalize)

    train_dataset = CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    losses = []
    top1_acc = []

    for images, target in tqdm(loader, leave=False):
        images = images.to(device)
        target = target.to(device)

        output = model(images)
        loss = criterion(output, target)

        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = target.detach().cpu().numpy()
        acc1 = accuracy(preds, labels)

        losses.append(loss.item())
        top1_acc.append(acc1)

    avg_loss = float(np.mean(losses)) if losses else 0.0
    avg_acc = float(np.mean(top1_acc)) if top1_acc else 0.0
    return avg_loss, avg_acc


def train_epoch_gaussian(
    model,
    train_loader,
    optimizer,
    device,
):
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    top1_acc = []

    for images, target in tqdm(train_loader, leave=False):
        images = images.to(device)
        target = target.to(device)

        output = model(images)
        loss = criterion(output, target)

        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = target.detach().cpu().numpy()
        acc1 = accuracy(preds, labels)

        top1_acc.append(acc1)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return float(np.mean(losses)), float(np.mean(top1_acc))


def train_epoch_product(
    model,
    train_loader,
    optimizer,
    device,
):
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    top1_acc = []

    for images, target in tqdm(train_loader, leave=False):
        images = images.to(device)
        target = target.to(device)

        output = model(images)
        loss = criterion(output, target)

        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = target.detach().cpu().numpy()
        acc1 = accuracy(preds, labels)

        top1_acc.append(acc1)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return float(np.mean(losses)), float(np.mean(top1_acc))


def train_epoch_baseline(
    model,
    train_loader,
    optimizer,
    device,
):
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    top1_acc = []

    for images, target in tqdm(train_loader, leave=False):
        images = images.to(device)
        target = target.to(device)

        output = model(images)
        loss = criterion(output, target)

        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = target.detach().cpu().numpy()
        acc1 = accuracy(preds, labels)

        top1_acc.append(acc1)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return float(np.mean(losses)), float(np.mean(top1_acc))


def get_model_dimension(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def matched_product_sigma_M(model, gaussian_sigma, max_grad_norm):
    M = get_model_dimension(model)
    sigma_M = math.sqrt(M) * gaussian_sigma * max_grad_norm
    return sigma_M, M


def run_one_experiment(args, run_seed):
    set_seed(run_seed)
    device = torch.device(args.device)

    train_loader, test_loader = build_dataloaders(
        batch_size=args.batch_size,
        batch_size_test=args.batch_size_test,
        data_root=args.data_root,
    )

    base_model = convnet(num_classes=10)

    gaussian_model = copy.deepcopy(base_model).to(device)
    product_model = copy.deepcopy(base_model).to(device)
    baseline_model = copy.deepcopy(base_model).to(device)

    gaussian_optimizer = optim.SGD(
        gaussian_model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=0.0,
    )

    product_optimizer = optim.SGD(
        product_model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=0.0,
    )

    baseline_optimizer = optim.SGD(
        baseline_model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=0.0,
    )

    product_sigma_M, M = matched_product_sigma_M(
        product_model,
        args.gaussian_noise_multiplier,
        args.max_grad_norm,
    )

    gaussian_engine = PrivacyEngine(accountant="rdp")
    gaussian_model, gaussian_optimizer, train_loader_g = gaussian_engine.make_private(
        module=gaussian_model,
        optimizer=gaussian_optimizer,
        data_loader=train_loader,
        noise_multiplier=args.gaussian_noise_multiplier,
        max_grad_norm=args.max_grad_norm,
    )

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
    for epoch in range(1, args.epochs + 1):
        print(f"\nRun {run_seed} | Epoch {epoch}/{args.epochs}")

        g_train_loss, g_train_acc = train_epoch_gaussian(
            model=gaussian_model,
            train_loader=train_loader_g,
            optimizer=gaussian_optimizer,
            device=device,
        )

        p_train_loss, p_train_acc = train_epoch_product(
            model=product_model,
            train_loader=train_loader_p,
            optimizer=product_optimizer,
            device=device,
        )

        b_train_loss, b_train_acc = train_epoch_baseline(
            model=baseline_model,
            train_loader=train_loader,
            optimizer=baseline_optimizer,
            device=device,
        )

        g_test_loss, g_acc = evaluate(gaussian_model, test_loader, device)
        p_test_loss, p_acc = evaluate(product_model, test_loader, device)
        b_test_loss, b_acc = evaluate(baseline_model, test_loader, device)

        print(
            f"[Gaussian] train_loss={g_train_loss:.6f} "
            f"train_acc={g_train_acc:.6f} "
            f"test_loss={g_test_loss:.6f} "
            f"test_acc={g_acc:.6f}"
        )
        print(
            f"[Product ] train_loss={p_train_loss:.6f} "
            f"train_acc={p_train_acc:.6f} "
            f"test_loss={p_test_loss:.6f} "
            f"test_acc={p_acc:.6f}"
        )
        print(
            f"[Baseline] train_loss={b_train_loss:.6f} "
            f"train_acc={b_train_acc:.6f} "
            f"test_loss={b_test_loss:.6f} "
            f"test_acc={b_acc:.6f}"
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
    parser = argparse.ArgumentParser(
        description="CIFAR10 Gaussian vs Product vs Baseline (accuracy only, 5-run average)"
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--batch-size-test", type=int, default=512)

    parser.add_argument("--lr", type=float, default=0.25)
    parser.add_argument("--momentum", type=float, default=0.9)

    parser.add_argument("--gaussian-noise-multiplier", type=float, default=0.50)
    parser.add_argument("--max-grad-norm", type=float, default=1.5)

    parser.add_argument("--delta0", type=float, default=1e-12)
    parser.add_argument("--k", type=int, default=300000)

    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-runs", type=int, default=5)

    args = parser.parse_args()

    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    temp_model = convnet(num_classes=10)
    _, M = matched_product_sigma_M(
        temp_model,
        args.gaussian_noise_multiplier,
        args.max_grad_norm,
    )
    product_sigma_M = math.sqrt(M) * args.gaussian_noise_multiplier * args.max_grad_norm

    print("\n===== CIFAR10 Experiment Configuration =====")
    print("dataset=CIFAR10")
    print("model=convnet(num_classes=10)")
    print(f"device={device}")
    print("train_size=50000")
    print(f"epochs={args.epochs}")
    print(f"batch_size={args.batch_size}")
    print(f"batch_size_test={args.batch_size_test}")
    print(f"lr={args.lr}")
    print(f"momentum={args.momentum}")
    print(f"gaussian_sigma={args.gaussian_noise_multiplier}")
    print(f"max_grad_norm={args.max_grad_norm}")
    print(f"delta0={args.delta0}")
    print(f"k={args.k}")
    print(f"model_dimension_M={M}")
    print(f"matched_product_sigma_M={product_sigma_M}")
    print(f"num_runs={args.num_runs}")
    print("===========================================\n")

    all_runs_results = []
    for run_idx in range(args.num_runs):
        run_seed = args.seed + run_idx
        run_results = run_one_experiment(args, run_seed)
        all_runs_results.append(run_results)

    averaged_results = average_results(all_runs_results, args.epochs)

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = save_dir / f"cifar10_epoch_results_multi_accountants_{now}.csv"

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