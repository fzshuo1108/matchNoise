import argparse
import copy
import math
import random
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from opacus import PrivacyEngine
from opacus.accountants import GaussianAccountant, PRVAccountant

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

    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.size(0)

    return correct / total


def train_one_epoch_with_extra_accountants(
    model,
    loader,
    optimizer,
    device,
    extra_accountants=None,
    noise_multiplier=None,
    sample_rate=None,
):
    """
    For Gaussian model:
    - optimizer.step() already updates the accountant inside gaussian_engine
    - we additionally step the standalone CLT/GDP and PRV accountants manually
    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if extra_accountants is not None:
            for acc in extra_accountants:
                acc.step(
                    noise_multiplier=noise_multiplier,
                    sample_rate=sample_rate,
                )


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()


def get_model_dimension(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def matched_product_sigma_M(model, gaussian_sigma, max_grad_norm):
    M = get_model_dimension(model)
    sigma_M = math.sqrt(M) * gaussian_sigma * max_grad_norm
    return sigma_M, M


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)

    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--momentum", type=float, default=0.3)

    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gaussian-noise-multiplier", type=float, default=1.3)

    parser.add_argument("--target-delta", type=float, default=1e-5)
    parser.add_argument("--delta0", type=float, default=1e-12)

    parser.add_argument("--k", type=int, default=40000)

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", default="./outputs")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

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

    # -----------------------------
    # Gaussian side
    # Main engine uses RDP ("MA" here)
    # -----------------------------
    gaussian_engine = PrivacyEngine(accountant="rdp")

    gaussian_model, gaussian_optimizer, train_loader_g = gaussian_engine.make_private(
        module=gaussian_model,
        optimizer=gaussian_optimizer,
        data_loader=train_loader,
        noise_multiplier=args.gaussian_noise_multiplier,
        max_grad_norm=args.max_grad_norm,
    )

    # Standalone Gaussian accountants for CLT/GDP and PRV
    gaussian_ma_accountant = gaussian_engine.accountant
    gaussian_clt_accountant = GaussianAccountant()
    gaussian_prv_accountant = PRVAccountant()

    # For manual stepping of CLT/GDP and PRV
    gaussian_sample_rate = 1.0 / len(train_loader_g)

    # -----------------------------
    # Product side
    # -----------------------------
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

    # -----------------------------
    # Save CSV
    # -----------------------------
    csv_path = save_dir / "mnist.csv"

    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)

    writer.writerow([
        "epoch",
        "gaussian_epsilon_ma",
        "gaussian_epsilon_clt",
        "gaussian_epsilon_prv",
        "gaussian_accuracy",
        "product_epsilon",
        "product_accuracy",
        "baseline_accuracy",
    ])

    print("\n===== Experiment Configuration =====")
    print("dataset=MNIST")
    print(f"model={base_model.__class__.__name__}")
    print(f"device={device}")
    print(f"epochs={args.epochs}")
    print(f"batch_size={args.batch_size}")
    print(f"lr={args.lr}")
    print(f"momentum={args.momentum}")
    print(f"max_grad_norm={args.max_grad_norm}")
    print(f"gaussian_sigma={args.gaussian_noise_multiplier}")
    print(f"product_sigma_M={product_sigma_M}")
    print(f"target_delta={args.target_delta}")
    print(f"delta0={args.delta0}")
    print(f"k={args.k}")
    print(f"model_dimension_M={M}")
    print("===================================\n")

    # -----------------------------
    # Train
    # -----------------------------
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}")

        # Gaussian model: train once, but step extra accountants manually
        train_one_epoch_with_extra_accountants(
            model=gaussian_model,
            loader=train_loader_g,
            optimizer=gaussian_optimizer,
            device=device,
            extra_accountants=[gaussian_clt_accountant, gaussian_prv_accountant],
            noise_multiplier=args.gaussian_noise_multiplier,
            sample_rate=gaussian_sample_rate,
        )

        # Product model
        train_one_epoch(
            model=product_model,
            loader=train_loader_p,
            optimizer=product_optimizer,
            device=device,
        )

        # Baseline model (non-private)
        train_one_epoch(
            model=baseline_model,
            loader=train_loader,
            optimizer=baseline_optimizer,
            device=device,
        )

        # Accuracy
        g_acc = evaluate(gaussian_model, test_loader, device)
        p_acc = evaluate(product_model, test_loader, device)
        b_acc = evaluate(baseline_model, test_loader, device)

        # Epsilons
        g_eps_ma = gaussian_ma_accountant.get_epsilon(args.target_delta)
        g_eps_clt = gaussian_clt_accountant.get_epsilon(args.target_delta)
        g_eps_prv = gaussian_prv_accountant.get_epsilon(args.target_delta)

        p_eps = product_engine.accountant.get_epsilon(args.target_delta)

        print(
            f"Gaussian MA/RDP eps: {g_eps_ma:.6f}, "
            f"CLT/GDP eps: {g_eps_clt:.6f}, "
            f"PRV eps: {g_eps_prv:.6f}, "
            f"acc: {g_acc:.6f}"
        )
        print(
            f"Product eps: {p_eps:.6f}, "
            f"acc: {p_acc:.6f}"
        )
        print(
            f"Baseline acc: {b_acc:.6f}"
        )

        writer.writerow([
            epoch,
            g_eps_ma,
            g_eps_clt,
            g_eps_prv,
            g_acc,
            p_eps,
            p_acc,
            b_acc,
        ])

    csv_file.close()

    print("CSV results saved to:", csv_path)


if __name__ == "__main__":
    main()