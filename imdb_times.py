#!/usr/bin/env python3
# This file is prepared for anonymous submission or public release.
# Original license, authorship, and institutional references have been removed for anonymity.
# Annotated for clarity and educational understanding.

import argparse
import copy
import math
import csv
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from datasets import load_dataset
from opacus import PrivacyEngine
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizerFast

from product_noise import ProductPrivacyEngine


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    return mean_loss, mean_acc


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch=None,
    tag="model",
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

    print(f"[{tag}] epoch={epoch} train_acc={mean_acc:.6f}")
    return mean_loss, mean_acc


def get_model_dimension(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def matched_product_sigma_M(model, gaussian_sigma, max_grad_norm):
    M = get_model_dimension(model)
    sigma_M = math.sqrt(M) * gaussian_sigma * max_grad_norm
    return sigma_M, M


def build_dataloaders(args):
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

    return train_loader, test_loader, tokenizer


def run_one_experiment(args, run_seed):
    set_seed(run_seed)
    device = torch.device(args.device)

    train_loader, test_loader, tokenizer = build_dataloaders(args)

    base_model = SampleNet(vocab_size=len(tokenizer))
    gaussian_model = copy.deepcopy(base_model).to(device)
    product_model = copy.deepcopy(base_model).to(device)
    baseline_model = copy.deepcopy(base_model).to(device)

    gaussian_optimizer = optim.Adam(gaussian_model.parameters(), lr=args.lr)
    product_optimizer = optim.Adam(product_model.parameters(), lr=args.lr)
    baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=args.lr)

    # Gaussian side: keep DP training, but no epsilon collection
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

    # Product side: keep DP training, but no epsilon collection
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

    results = []

    print(f"\n========== Run seed = {run_seed} ==========")
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

        g_test_loss, g_acc = evaluate(
            gaussian_model,
            test_loader,
            device,
            desc=f"Eval-Gaussian-Epoch{epoch}",
        )
        p_test_loss, p_acc = evaluate(
            product_model,
            test_loader,
            device,
            desc=f"Eval-Product-Epoch{epoch}",
        )
        b_test_loss, b_acc = evaluate(
            baseline_model,
            test_loader,
            device,
            desc=f"Eval-Baseline-Epoch{epoch}",
        )

        print(
            f"[Gaussian] train_acc={g_train_acc:.6f} test_acc={g_acc:.6f}"
        )
        print(
            f"[Product ] train_acc={p_train_acc:.6f} test_acc={p_acc:.6f}"
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
    parser = argparse.ArgumentParser(
        description="IMDB Gaussian vs Product vs Baseline (accuracy only, 5-run average)",
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
    parser.add_argument("--delta0", type=float, default=1e-12)
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=256,
        metavar="SL",
    )
    # parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--secure-rng", action="store_true", default=False)
    parser.add_argument("--data-root", type=str, default="../imdb")
    parser.add_argument("-j", "--workers", default=0, type=int, metavar="N")
    parser.add_argument("--save-dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-runs", type=int, default=5)

    args = parser.parse_args()
    device = torch.device(args.device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # only for configuration display
    temp_train_loader, _, tokenizer = build_dataloaders(args)
    temp_model = SampleNet(vocab_size=len(tokenizer))
    _, M = matched_product_sigma_M(
        temp_model,
        args.sigma,
        args.max_per_sample_grad_norm,
    )
    product_sigma_M = math.sqrt(M) * args.sigma * args.max_per_sample_grad_norm

    print("\n===== Experiment Configuration =====")
    print("dataset=IMDB")
    print(f"model={temp_model.__class__.__name__}")
    print(f"device={device}")
    print(f"epochs={args.epochs}")
    print(f"batch_size={args.batch_size}")
    print(f"lr={args.lr}")
    print(f"max_grad_norm={args.max_per_sample_grad_norm}")
    print(f"gaussian_sigma={args.sigma}")
    print(f"product_sigma_M={product_sigma_M}")
    print(f"delta0={args.delta0}")
    print(f"k={args.k}")
    print(f"model_dimension_M={M}")
    print(f"num_runs={args.num_runs}")
    print("===================================\n")

    del temp_train_loader

    all_runs_results = []
    for run_idx in range(args.num_runs):
        run_seed = args.seed + run_idx
        run_results = run_one_experiment(args, run_seed)
        all_runs_results.append(run_results)

    averaged_results = average_results(all_runs_results, args.epochs)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = save_dir / f"imdb_epoch_results_accuracy_only_{now}.csv"

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

    print("\nAveraged CSV results saved to:", csv_path)


if __name__ == "__main__":
    main()