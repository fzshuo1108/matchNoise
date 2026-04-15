#!/usr/bin/env python3

import argparse
import copy
import csv
import math
import os
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from opacus import PrivacyEngine
from product_noise import ProductPrivacyEngine


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TwoLayerNN(nn.Module):
    def __init__(self, input_dim):
        super(TwoLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def download_adult_data(data_path):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    if not os.path.exists(data_path):
        df = pd.read_csv(
            url,
            header=None,
            names=[
                "age", "workclass", "fnlwgt", "education", "education-num",
                "marital-status", "occupation", "relationship", "race", "sex",
                "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
            ]
        )
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)
    return df


def load_adult_data(data_path):
    data = download_adult_data(data_path)
    data.dropna(inplace=True)

    label_encoder = LabelEncoder()
    data["income"] = label_encoder.fit_transform(data["income"])

    categorical_columns = data.select_dtypes(include=["object"]).columns
    data = pd.get_dummies(data, columns=categorical_columns)

    X = data.drop("income", axis=1).values
    y = data["income"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


@torch.no_grad()
def evaluate(model, device, loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    losses = []
    correct = 0
    total = 0

    for data, target in tqdm(loader, leave=False):
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        loss = criterion(output, target)

        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        losses.append(loss.item())

    avg_loss = float(np.mean(losses)) if losses else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


def train_epoch_gaussian(
    model,
    device,
    train_loader,
    optimizer,
):
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    correct = 0
    total = 0

    for data, target in tqdm(train_loader, leave=False):
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        loss = criterion(output, target)

        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    train_acc = correct / total if total > 0 else 0.0
    train_loss = float(np.mean(losses)) if losses else 0.0
    return train_loss, train_acc


def train_epoch_product(
    model,
    device,
    train_loader,
    optimizer,
):
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    correct = 0
    total = 0

    for data, target in tqdm(train_loader, leave=False):
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        loss = criterion(output, target)

        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    train_acc = correct / total if total > 0 else 0.0
    train_loss = float(np.mean(losses)) if losses else 0.0
    return train_loss, train_acc


def train_epoch_baseline(
    model,
    device,
    train_loader,
    optimizer,
):
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    correct = 0
    total = 0

    for data, target in tqdm(train_loader, leave=False):
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        loss = criterion(output, target)

        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    train_acc = correct / total if total > 0 else 0.0
    train_loss = float(np.mean(losses)) if losses else 0.0
    return train_loss, train_acc


def get_model_dimension(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def matched_product_sigma_M(model, gaussian_sigma, max_grad_norm):
    M = get_model_dimension(model)
    sigma_M = math.sqrt(M) * gaussian_sigma * max_grad_norm
    return sigma_M, M


def run_one_experiment(args, run_seed, X_train, X_test, y_train, y_test):
    set_seed(run_seed)
    device = torch.device(args.device)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
    )

    input_dim = X_train.shape[1]
    base_model = TwoLayerNN(input_dim)

    gaussian_model = copy.deepcopy(base_model).to(device)
    product_model = copy.deepcopy(base_model).to(device)
    baseline_model = copy.deepcopy(base_model).to(device)

    gaussian_optimizer = optim.SGD(
        gaussian_model.parameters(),
        lr=args.lr,
    )
    product_optimizer = optim.SGD(
        product_model.parameters(),
        lr=args.lr,
    )
    baseline_optimizer = optim.SGD(
        baseline_model.parameters(),
        lr=args.lr,
    )

    product_sigma_M, M = matched_product_sigma_M(
        product_model,
        args.gaussian_noise_multiplier,
        args.max_grad_norm,
    )

    # Gaussian side
    gaussian_engine = PrivacyEngine(accountant="rdp")
    gaussian_model, gaussian_optimizer, train_loader_g = gaussian_engine.make_private(
        module=gaussian_model,
        optimizer=gaussian_optimizer,
        data_loader=train_loader,
        noise_multiplier=args.gaussian_noise_multiplier,
        max_grad_norm=args.max_grad_norm,
    )

    # Product side
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
            device=device,
            train_loader=train_loader_g,
            optimizer=gaussian_optimizer,
        )

        p_train_loss, p_train_acc = train_epoch_product(
            model=product_model,
            device=device,
            train_loader=train_loader_p,
            optimizer=product_optimizer,
        )

        b_train_loss, b_train_acc = train_epoch_baseline(
            model=baseline_model,
            device=device,
            train_loader=train_loader,
            optimizer=baseline_optimizer,
        )

        g_test_loss, g_acc = evaluate(gaussian_model, device, test_loader)
        p_test_loss, p_acc = evaluate(product_model, device, test_loader)
        b_test_loss, b_acc = evaluate(baseline_model, device, test_loader)

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
    parser = argparse.ArgumentParser(description="Adult Gaussian vs Product Noise vs Baseline (5-run average)")

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--test-batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=40)

    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--gaussian-noise-multiplier", type=float, default=0.55)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--delta0", type=float, default=1e-12)
    parser.add_argument("--k", type=int, default=150000)

    # parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-path", type=str, default="../adult.data")
    parser.add_argument("--save-dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-runs", type=int, default=5)

    args = parser.parse_args()

    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    X_train, X_test, y_train, y_test = load_adult_data(args.data_path)
    input_dim = X_train.shape[1]

    print("\n===== Adult Experiment Configuration =====")
    print("dataset=Adult")
    print(f"input_dim={input_dim}")
    print(f"device={device}")
    print(f"epochs={args.epochs}")
    print(f"batch_size={args.batch_size}")
    print(f"test_batch_size={args.test_batch_size}")
    print(f"lr={args.lr}")
    print(f"gaussian_sigma={args.gaussian_noise_multiplier}")
    print(f"max_grad_norm={args.max_grad_norm}")
    print(f"delta0={args.delta0}")
    print(f"k={args.k}")
    print(f"num_runs={args.num_runs}")
    print("=========================================\n")

    all_runs_results = []
    for run_idx in range(args.num_runs):
        run_seed = args.seed + run_idx
        run_results = run_one_experiment(
            args=args,
            run_seed=run_seed,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
        all_runs_results.append(run_results)

    averaged_results = average_results(all_runs_results, args.epochs)

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = save_dir / f"adult_compare_product_noise_{now}.csv"

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