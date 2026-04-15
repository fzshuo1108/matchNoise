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
from opacus.accountants import GaussianAccountant, PRVAccountant

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
def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    losses = []
    correct = 0
    total = 0

    for data, target in tqdm(test_loader, leave=False):
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        loss = criterion(output, target)

        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        losses.append(loss.item())

    avg_loss = float(np.mean(losses))
    acc = correct / total
    return avg_loss, acc


def train_epoch_gaussian(
    model,
    device,
    train_loader,
    optimizer,
    extra_accountants,
    noise_multiplier,
    sample_rate,
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

        for acc in extra_accountants:
            acc.step(
                noise_multiplier=noise_multiplier,
                sample_rate=sample_rate,
            )

    train_acc = correct / total
    train_loss = float(np.mean(losses))
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

    train_acc = correct / total
    train_loss = float(np.mean(losses))
    return train_loss, train_acc


def get_model_dimension(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def matched_product_sigma_M(model, gaussian_sigma, max_grad_norm):
    M = get_model_dimension(model)
    sigma_M = math.sqrt(M) * gaussian_sigma * max_grad_norm
    return sigma_M, M


def main():
    parser = argparse.ArgumentParser(description="Adult Gaussian vs Product Noise")

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--test-batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=40)

    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--gaussian-noise-multiplier", type=float, default=0.55)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--target-delta", type=float, default=1e-5)
    parser.add_argument("--delta0", type=float, default=1e-12)

    parser.add_argument("--k", type=int, default=150000)

    # parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-path", type=str, default="../adult.data")
    parser.add_argument("--save-dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    X_train, X_test, y_train, y_test = load_adult_data(args.data_path)

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

    gaussian_optimizer = optim.SGD(
        gaussian_model.parameters(),
        lr=args.lr,
    )
    product_optimizer = optim.SGD(
        product_model.parameters(),
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

    gaussian_ma_accountant = gaussian_engine.accountant
    gaussian_clt_accountant = GaussianAccountant()
    gaussian_prv_accountant = PRVAccountant()

    gaussian_sample_rate = 1.0 / len(train_loader_g)

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

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = save_dir / f"adult_compare_product_noise_{now}.csv"

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
    ])

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
    print(f"target_delta={args.target_delta}")
    print(f"delta0={args.delta0}")
    print(f"k={args.k}")
    print(f"model_dimension_M={M}")
    print(f"matched_product_sigma_M={product_sigma_M}")
    print("=========================================\n")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        g_train_loss, g_train_acc = train_epoch_gaussian(
            model=gaussian_model,
            device=device,
            train_loader=train_loader_g,
            optimizer=gaussian_optimizer,
            extra_accountants=[gaussian_clt_accountant, gaussian_prv_accountant],
            noise_multiplier=args.gaussian_noise_multiplier,
            sample_rate=gaussian_sample_rate,
        )

        p_train_loss, p_train_acc = train_epoch_product(
            model=product_model,
            device=device,
            train_loader=train_loader_p,
            optimizer=product_optimizer,
        )

        g_test_loss, g_acc = test(gaussian_model, device, test_loader)
        p_test_loss, p_acc = test(product_model, device, test_loader)

        g_eps_ma = gaussian_ma_accountant.get_epsilon(args.target_delta)
        g_eps_clt = gaussian_clt_accountant.get_epsilon(args.target_delta)
        g_eps_prv = gaussian_prv_accountant.get_epsilon(args.target_delta)

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
            f"PRV={g_eps_prv:.6f}"
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
            g_eps_prv,
            g_acc,
            p_eps,
            p_acc,
        ])

    csv_file.close()
    print(f"\nCSV saved to: {csv_path}")


if __name__ == "__main__":
    main()