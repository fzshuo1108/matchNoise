#!/usr/bin/env python3
# This file is prepared for anonymous submission or public release.
# Original license, authorship, and institutional references have been removed for anonymity.
# Annotated for clarity and educational understanding.

import argparse
import copy
import math
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from datasets import load_dataset
from opacus import PrivacyEngine
from opacus.accountants import GaussianAccountant, PRVAccountant
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizerFast

# 这里假设你本地已经有这个文件
from product_noise import ProductPrivacyEngine


class SampleNet(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 16)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.emb(x)               # [B, L, 16]
        x = x.transpose(1, 2)         # [B, 16, L]
        x = self.pool(x).squeeze(-1)  # [B, 16]
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def name(self):
        return "SampleNet"


def binary_accuracy(preds, y):
    correct = (y.long() == torch.argmax(preds, dim=1)).float()
    acc = correct.sum() / len(correct)
    return acc


def padded_collate(batch, padding_idx=0):
    x = pad_sequence(
        [elem["input_ids"] for elem in batch],
        batch_first=True,
        padding_value=padding_idx,
    )
    y = torch.stack([elem["label"] for elem in batch]).long()
    return x, y


@torch.no_grad()
def evaluate(model, loader, device, desc="Eval"):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    losses = []
    accuracies = []

    for data, label in tqdm(loader, desc=desc, leave=False):
        data = data.to(device)
        label = label.to(device)

        predictions = model(data)
        loss = criterion(predictions, label)
        acc = binary_accuracy(predictions, label)

        losses.append(loss.item())
        accuracies.append(acc.item())

    mean_loss = float(np.mean(losses)) if losses else 0.0
    mean_acc = float(np.mean(accuracies)) if accuracies else 0.0

    print(f"{desc} | loss={mean_loss:.6f} | acc={mean_acc:.6f}")
    return mean_acc


def train_one_epoch_with_extra_accountants(
    model,
    loader,
    optimizer,
    device,
    extra_accountants=None,
    noise_multiplier=None,
    sample_rate=None,
    epoch=None,
    tag="gaussian",
):
    """
    Gaussian:
    - optimizer.step() 已经会更新主 privacy engine 内部的 RDP accountant
    - 额外手动 step GaussianAccountant / PRVAccountant
    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    accuracies = []

    for data, label in tqdm(loader, desc=f"Train-{tag}-Epoch{epoch}", leave=False):
        data = data.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        predictions = model(data)
        loss = criterion(predictions, label)
        acc = binary_accuracy(predictions, label)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        accuracies.append(acc.item())

        if extra_accountants is not None:
            for acc_obj in extra_accountants:
                acc_obj.step(
                    noise_multiplier=noise_multiplier,
                    sample_rate=sample_rate,
                )

    mean_loss = float(np.mean(losses)) if losses else 0.0
    mean_acc = float(np.mean(accuracies)) if accuracies else 0.0

    print(f"[{tag}] epoch={epoch} train_loss={mean_loss:.6f} train_acc={mean_acc:.6f}")
    return mean_acc


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch=None,
    tag="product",
):
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    accuracies = []

    for data, label in tqdm(loader, desc=f"Train-{tag}-Epoch{epoch}", leave=False):
        data = data.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        predictions = model(data)
        loss = criterion(predictions, label)
        acc = binary_accuracy(predictions, label)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        accuracies.append(acc.item())

    mean_loss = float(np.mean(losses)) if losses else 0.0
    mean_acc = float(np.mean(accuracies)) if accuracies else 0.0

    print(f"[{tag}] epoch={epoch} train_loss={mean_loss:.6f} train_acc={mean_acc:.6f}")
    return mean_acc


def get_model_dimension(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def matched_product_sigma_M(model, gaussian_sigma, max_grad_norm):
    M = get_model_dimension(model)
    sigma_M = math.sqrt(M) * gaussian_sigma * max_grad_norm
    return sigma_M, M


def main():
    parser = argparse.ArgumentParser(
        description="IMDB Gaussian vs Product Noise",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-b", "--batch-size", type=int, default=512, metavar="B")
    parser.add_argument("-n", "--epochs", type=int, default=60, metavar="N")
    parser.add_argument("--lr", type=float, default=0.02, metavar="LR")
    parser.add_argument("--sigma", type=float, default=0.56, metavar="S")
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=1.0,
        metavar="C",
    )
    parser.add_argument("-k", default=40000, type=int, metavar="N")
    parser.add_argument("--epsilon", default=1.0, type=float, metavar="N")
    parser.add_argument("--delta", type=float, default=1e-5, metavar="D")
    parser.add_argument("--delta0", type=float, default=1e-12)
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=256,
        metavar="SL",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--secure-rng", action="store_true", default=False)
    parser.add_argument("--data-root", type=str, default="../imdb")
    parser.add_argument("-j", "--workers", default=0, type=int, metavar="N")
    parser.add_argument("--save-dir", type=str, default="./outputs")

    args = parser.parse_args()
    device = torch.device(args.device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Load IMDB
    # -----------------------------
    raw_dataset = load_dataset("imdb", cache_dir=args.data_root)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

    dataset = raw_dataset.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            max_length=args.max_sequence_length,
        ),
        batched=True,
    )
    dataset.set_format(type="torch", columns=["input_ids", "label"])

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    train_loader = DataLoader(
        train_dataset,
        num_workers=args.workers,
        batch_size=args.batch_size,
        collate_fn=padded_collate,
        pin_memory=True,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=padded_collate,
        pin_memory=True,
    )

    # -----------------------------
    # Build two models
    # -----------------------------
    base_model = SampleNet(vocab_size=len(tokenizer))
    gaussian_model = copy.deepcopy(base_model).to(device)
    product_model = copy.deepcopy(base_model).to(device)

    gaussian_optimizer = optim.Adam(gaussian_model.parameters(), lr=args.lr)
    product_optimizer = optim.Adam(product_model.parameters(), lr=args.lr)

    # -----------------------------
    # Gaussian side
    # 主 privacy engine: RDP/MA
    # 额外 standalone: CLT/GDP, PRV
    # -----------------------------
    gaussian_engine = PrivacyEngine(
        secure_mode=args.secure_rng,
        accountant="rdp",
    )

    gaussian_model, gaussian_optimizer, train_loader_g = gaussian_engine.make_private(
        module=gaussian_model,
        optimizer=gaussian_optimizer,
        data_loader=train_loader,
        noise_multiplier=args.sigma,
        max_grad_norm=args.max_per_sample_grad_norm,
        poisson_sampling=False,
    )

    gaussian_ma_accountant = gaussian_engine.accountant
    gaussian_clt_accountant = GaussianAccountant()
    gaussian_prv_accountant = PRVAccountant()

    gaussian_sample_rate = 1.0 / len(train_loader_g)

    # -----------------------------
    # Product side
    # -----------------------------
    product_sigma_M, M = matched_product_sigma_M(
        product_model,
        args.sigma,
        args.max_per_sample_grad_norm,
    )

    product_engine = ProductPrivacyEngine(
        clipping_norm=args.max_per_sample_grad_norm,
        delta0=args.delta0,
        M=M,
        k=args.k,
    )

    product_model, product_optimizer, train_loader_p = product_engine.make_private(
        module=product_model,
        optimizer=product_optimizer,
        data_loader=train_loader,
        noise_multiplier=product_sigma_M,
        max_grad_norm=args.max_per_sample_grad_norm,
    )

    # -----------------------------
    # CSV
    # 你要的是：2个acc + 4个epsilon
    # -----------------------------
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = save_dir / f"imdb_epoch_results_2acc_4eps_{now}.csv"

    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)

    writer.writerow([
        "epoch",
        "gaussian_epsilon_ma",
        "gaussian_epsilon_clt",
        "gaussian_epsilon_prv",
        "product_epsilon",
        "gaussian_accuracy",
        "product_accuracy",
    ])

    print("\n===== Experiment Configuration =====")
    print("dataset=IMDB")
    print(f"model={base_model.__class__.__name__}")
    print(f"device={device}")
    print(f"epochs={args.epochs}")
    print(f"batch_size={args.batch_size}")
    print(f"lr={args.lr}")
    print(f"max_grad_norm={args.max_per_sample_grad_norm}")
    print(f"gaussian_sigma={args.sigma}")
    print(f"product_sigma_M={product_sigma_M}")
    print(f"delta={args.delta}")
    print(f"delta0={args.delta0}")
    print(f"k={args.k}")
    print(f"model_dimension_M={M}")
    print("===================================\n")

    # -----------------------------
    # Train by epoch
    # -----------------------------
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}")

        # Gaussian
        train_one_epoch_with_extra_accountants(
            model=gaussian_model,
            loader=train_loader_g,
            optimizer=gaussian_optimizer,
            device=device,
            extra_accountants=[gaussian_clt_accountant, gaussian_prv_accountant],
            noise_multiplier=args.sigma,
            sample_rate=gaussian_sample_rate,
            epoch=epoch,
            tag="gaussian",
        )

        # Product
        train_one_epoch(
            model=product_model,
            loader=train_loader_p,
            optimizer=product_optimizer,
            device=device,
            epoch=epoch,
            tag="product",
        )

        # Accuracy
        g_acc = evaluate(gaussian_model, test_loader, device, desc=f"Eval-Gaussian-Epoch{epoch}")
        p_acc = evaluate(product_model, test_loader, device, desc=f"Eval-Product-Epoch{epoch}")

        # Four epsilons
        g_eps_ma = gaussian_ma_accountant.get_epsilon(args.delta)
        g_eps_clt = gaussian_clt_accountant.get_epsilon(args.delta)
        g_eps_prv = gaussian_prv_accountant.get_epsilon(args.delta)
        p_eps = product_engine.accountant.get_epsilon(args.delta)

        print(
            f"Gaussian | eps_ma={g_eps_ma:.6f}, "
            f"eps_clt={g_eps_clt:.6f}, "
            f"eps_prv={g_eps_prv:.6f}, "
            f"acc={g_acc:.6f}"
        )
        print(
            f"Product  | eps={p_eps:.6f}, "
            f"acc={p_acc:.6f}"
        )

        # Save one row per epoch
        writer.writerow([
            epoch,
            g_eps_ma,
            g_eps_clt,
            g_eps_prv,
            p_eps,
            g_acc,
            p_acc,
        ])

        csv_file.flush()

    csv_file.close()
    print("\nCSV results saved to:", csv_path)


if __name__ == "__main__":
    main()