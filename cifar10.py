import argparse
import copy
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from opacus import PrivacyEngine
from opacus.accountants import GaussianAccountant, PRVAccountant
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
def test(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    losses = []
    top1_acc = []

    for images, target in tqdm(test_loader, leave=False):
        images = images.to(device)
        target = target.to(device)

        output = model(images)
        loss = criterion(output, target)

        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = target.detach().cpu().numpy()
        acc1 = accuracy(preds, labels)

        losses.append(loss.item())
        top1_acc.append(acc1)

    avg_loss = float(np.mean(losses))
    avg_acc = float(np.mean(top1_acc))
    return avg_loss, avg_acc


def train_epoch_gaussian(
    model,
    train_loader,
    optimizer,
    device,
    extra_accountants,
    noise_multiplier,
    sample_rate,
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

        for acc in extra_accountants:
            acc.step(
                noise_multiplier=noise_multiplier,
                sample_rate=sample_rate,
            )

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


def get_model_dimension(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def matched_product_sigma_M(model, gaussian_sigma, max_grad_norm):
    M = get_model_dimension(model)
    sigma_M = math.sqrt(M) * gaussian_sigma * max_grad_norm
    return sigma_M, M


def main():
    parser = argparse.ArgumentParser(description="CIFAR10 Gaussian vs Product Noise")

    # 你截图对应的默认参数
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--batch-size-test", type=int, default=512)

    parser.add_argument("--lr", type=float, default=0.25)
    parser.add_argument("--momentum", type=float, default=0.9)

    parser.add_argument("--gaussian-noise-multiplier", type=float, default=0.50)
    parser.add_argument("--max-grad-norm", type=float, default=1.5)

    parser.add_argument("--target-delta", type=float, default=1e-5)
    parser.add_argument("--delta0", type=float, default=1e-12)

    # 这里就是你要的外部传入 k
    parser.add_argument("--k", type=int, default=300000)

    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    train_loader, test_loader = build_dataloaders(
        batch_size=args.batch_size,
        batch_size_test=args.batch_size_test,
        data_root=args.data_root,
    )

    base_model = convnet(num_classes=10)

    gaussian_model = copy.deepcopy(base_model).to(device)
    product_model = copy.deepcopy(base_model).to(device)

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

    product_sigma_M, M = matched_product_sigma_M(
        product_model,
        args.gaussian_noise_multiplier,
        args.max_grad_norm,
    )

    # Gaussian 主训练：RDP / MA
    gaussian_engine = PrivacyEngine(accountant="rdp")
    gaussian_model, gaussian_optimizer, train_loader_g = gaussian_engine.make_private(
        module=gaussian_model,
        optimizer=gaussian_optimizer,
        data_loader=train_loader,
        noise_multiplier=args.gaussian_noise_multiplier,
        max_grad_norm=args.max_grad_norm,
    )

    gaussian_ma_accountant = gaussian_engine.accountant
    gaussian_clt_accountant = GaussianAccountant()
    # gaussian_prv_accountant = PRVAccountant()

    gaussian_sample_rate = 1.0 / len(train_loader_g)

    # Product engine：这里明确传入 k
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

    csv_path = save_dir / "cifar10_epoch_results_multi_accountants.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)

    writer.writerow([
        "epoch",
        "gaussian_epsilon_ma",
        "gaussian_epsilon_clt",
        # "gaussian_epsilon_prv",
        "gaussian_accuracy",
        "product_epsilon",
        "product_accuracy",
    ])

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
    print(f"target_delta={args.target_delta}")
    print(f"delta0={args.delta0}")
    print(f"k={args.k}")
    print(f"model_dimension_M={M}")
    print(f"matched_product_sigma_M={product_sigma_M}")
    print("===========================================\n")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        g_train_loss, g_train_acc = train_epoch_gaussian(
            model=gaussian_model,
            train_loader=train_loader_g,
            optimizer=gaussian_optimizer,
            device=device,
            # extra_accountants=[gaussian_clt_accountant, gaussian_prv_accountant],
            extra_accountants=[gaussian_clt_accountant],
            noise_multiplier=args.gaussian_noise_multiplier,
            sample_rate=gaussian_sample_rate,
        )

        p_train_loss, p_train_acc = train_epoch_product(
            model=product_model,
            train_loader=train_loader_p,
            optimizer=product_optimizer,
            device=device,
        )

        g_test_loss, g_acc = test(gaussian_model, test_loader, device)
        p_test_loss, p_acc = test(product_model, test_loader, device)

        g_eps_ma = gaussian_ma_accountant.get_epsilon(args.target_delta)
        g_eps_clt = gaussian_clt_accountant.get_epsilon(args.target_delta)
        # g_eps_prv = gaussian_prv_accountant.get_epsilon(args.target_delta)

        p_eps = product_engine.accountant.get_epsilon(args.target_delta)

        print(
            f"[Gaussian] train_loss={g_train_loss:.6f} "
            f"train_acc={g_train_acc:.6f} "
            f"test_loss={g_test_loss:.6f} "
            f"test_acc={g_acc:.6f}"
        )
        print(
            f"[Gaussian eps] MA/RDP={g_eps_ma:.6f}, "
            f"CLT/GDP={g_eps_clt:.6f}, "
            # f"PRV={g_eps_prv:.6f}"
        )
        print(
            f"[Product ] train_loss={p_train_loss:.6f} "
            f"train_acc={p_train_acc:.6f} "
            f"test_loss={p_test_loss:.6f} "
            f"test_acc={p_acc:.6f} "
            f"eps={p_eps:.6f}"
        )

        writer.writerow([
            epoch,
            g_eps_ma,
            g_eps_clt,
            # g_eps_prv,
            g_acc,
            p_eps,
            p_acc,
        ])

    csv_file.close()
    print(f"\nCSV results saved to: {csv_path}")


if __name__ == "__main__":
    main()