#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import csv
import math
import os
import random
import urllib.request
import zipfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from opacus import PrivacyEngine
from product_noise import ProductPrivacyEngine


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Constants for MovieLens 1M
# -----------------------------
NUM_USERS = 6040
NUM_MOVIES = 3706
NUM_CLASSES = 5  # ratings: 1~5 -> mapped to 0~4


# -----------------------------
# Dataset
# -----------------------------
class MovieLensDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user = self.data[idx, 0]
        movie = self.data[idx, 1]
        rating = self.data[idx, 2] - 1  # map [1,5] -> [0,4]
        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(movie, dtype=torch.long),
            torch.tensor(rating, dtype=torch.long),
        )


# -----------------------------
# Model
# -----------------------------
class MovieLensModel(nn.Module):
    def __init__(
        self,
        num_users,
        num_movies,
        latent_dim_user=10,
        latent_dim_movie=10,
        latent_dim_mf=5,
    ):
        super().__init__()

        self.user_embedding_mf = nn.Embedding(num_users, latent_dim_mf)
        self.movie_embedding_mf = nn.Embedding(num_movies, latent_dim_mf)

        self.user_embedding = nn.Embedding(num_users, latent_dim_user)
        self.movie_embedding = nn.Embedding(num_movies, latent_dim_movie)

        self.fc = nn.Linear(
            latent_dim_user + latent_dim_movie + latent_dim_mf,
            NUM_CLASSES,
        )

    def forward(self, user, movie):
        user_embedding_mf = self.user_embedding_mf(user)
        movie_embedding_mf = self.movie_embedding_mf(movie)
        mf_vector = user_embedding_mf * movie_embedding_mf

        user_embedding = self.user_embedding(user)
        movie_embedding = self.movie_embedding(movie)
        mlp_vector = torch.cat([user_embedding, movie_embedding], dim=-1)

        predict_vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        output = self.fc(predict_vector)
        return output


# -----------------------------
# Data utils
# -----------------------------
def Download_Extract_MovieLens(data_root):
    url = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
    zip_path = os.path.join(data_root, "ml-1m.zip")

    if not os.path.exists(data_root):
        os.makedirs(data_root)

    if not os.path.exists(zip_path):
        print("Downloading MovieLens 1M dataset...")
        urllib.request.urlretrieve(url, zip_path)

    extract_dir = os.path.join(data_root, "ml-1m")
    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_root)


def load_MovieLens(data_root, training_data=800000):
    Download_Extract_MovieLens(data_root)

    data_path = os.path.join(data_root, "ml-1m", "ratings.dat")
    ratings = pd.read_csv(
        data_path,
        sep="::",
        engine="python",
        names=["UserID", "MovieID", "Rating", "Timestamp"],
    )

    ratings = ratings[["UserID", "MovieID", "Rating"]].values
    ratings[:, 0] -= 1
    ratings[:, 1] -= 1

    ratings = ratings[
        (ratings[:, 0] < NUM_USERS) &
        (ratings[:, 1] < NUM_MOVIES)
    ]

    np.random.shuffle(ratings)

    train_data = ratings[:training_data]
    test_data = ratings[training_data:]

    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    return train_data, test_data


# -----------------------------
# Evaluate
# -----------------------------
@torch.no_grad()
def evaluate_rmse(model, device, loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    losses = []
    all_preds = []
    all_targets = []

    for users, movies, ratings in tqdm(loader, leave=False):
        users = users.to(device)
        movies = movies.to(device)
        ratings = ratings.to(device)

        output = model(users, movies)
        loss = criterion(output, ratings)

        preds = torch.argmax(output, dim=1)

        losses.append(loss.item())
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(ratings.cpu().numpy())

    avg_loss = float(np.mean(losses)) if losses else 0.0
    rmse = float(np.sqrt(mean_squared_error(all_targets, all_preds)))
    return avg_loss, rmse


# -----------------------------
# Training
# -----------------------------
def train_one_epoch_rmse(
    model,
    device,
    train_loader,
    optimizer,
):
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    all_preds = []
    all_targets = []

    for users, movies, ratings in tqdm(train_loader, leave=False):
        users = users.to(device)
        movies = movies.to(device)
        ratings = ratings.to(device)

        optimizer.zero_grad()
        output = model(users, movies)
        loss = criterion(output, ratings)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(output, dim=1)

        losses.append(loss.item())
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(ratings.detach().cpu().numpy())

    train_loss = float(np.mean(losses)) if losses else 0.0
    train_rmse = float(np.sqrt(mean_squared_error(all_targets, all_preds)))
    return train_loss, train_rmse


# -----------------------------
# Product sigma matching
# -----------------------------
def get_model_dimension(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def matched_product_sigma_M(model, gaussian_sigma, max_grad_norm):
    M = get_model_dimension(model)
    sigma_M = math.sqrt(M) * gaussian_sigma * max_grad_norm
    return sigma_M, M


def run_one_experiment(args, run_seed):
    set_seed(run_seed)
    device = torch.device(args.device)

    train_data, test_data = load_MovieLens(
        data_root=args.data_root,
        training_data=args.training_data,
    )

    train_loader = DataLoader(
        MovieLensDataset(train_data),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        MovieLensDataset(test_data),
        batch_size=args.test_batch_size,
        shuffle=False,
    )

    base_model = MovieLensModel(NUM_USERS, NUM_MOVIES)

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
    print("dataset=MovieLens-1M")
    print(f"device={device}")
    print(f"training_data={args.training_data}")
    print(f"model_dimension_M={M}")
    print(f"product_sigma_M={product_sigma_M:.6f}")

    for epoch in range(1, args.epochs + 1):
        print(f"\nRun {run_seed} | Epoch {epoch}/{args.epochs}")

        g_train_loss, g_train_rmse = train_one_epoch_rmse(
            model=gaussian_model,
            device=device,
            train_loader=train_loader_g,
            optimizer=gaussian_optimizer,
        )

        p_train_loss, p_train_rmse = train_one_epoch_rmse(
            model=product_model,
            device=device,
            train_loader=train_loader_p,
            optimizer=product_optimizer,
        )

        b_train_loss, b_train_rmse = train_one_epoch_rmse(
            model=baseline_model,
            device=device,
            train_loader=train_loader,
            optimizer=baseline_optimizer,
        )

        g_test_loss, g_test_rmse = evaluate_rmse(
            gaussian_model,
            device,
            test_loader,
        )
        p_test_loss, p_test_rmse = evaluate_rmse(
            product_model,
            device,
            test_loader,
        )
        b_test_loss, b_test_rmse = evaluate_rmse(
            baseline_model,
            device,
            test_loader,
        )

        print(
            f"[Gaussian] train_rmse={g_train_rmse:.6f} test_rmse={g_test_rmse:.6f}"
        )
        print(
            f"[Product ] train_rmse={p_train_rmse:.6f} test_rmse={p_test_rmse:.6f}"
        )
        print(
            f"[Baseline] train_rmse={b_train_rmse:.6f} test_rmse={b_test_rmse:.6f}"
        )

        results.append({
            "epoch": epoch,
            "gaussian_train_rmse": g_train_rmse,
            "gaussian_test_rmse": g_test_rmse,
            "product_train_rmse": p_train_rmse,
            "product_test_rmse": p_test_rmse,
            "baseline_train_rmse": b_train_rmse,
            "baseline_test_rmse": b_test_rmse,
        })

    return results


def average_results(all_runs_results, epochs):
    averaged = []

    for epoch_idx in range(epochs):
        rows = [run[epoch_idx] for run in all_runs_results]

        averaged.append({
            "epoch": rows[0]["epoch"],
            "gaussian_train_rmse": float(np.mean([r["gaussian_train_rmse"] for r in rows])),
            "gaussian_test_rmse": float(np.mean([r["gaussian_test_rmse"] for r in rows])),
            "product_train_rmse": float(np.mean([r["product_train_rmse"] for r in rows])),
            "product_test_rmse": float(np.mean([r["product_test_rmse"] for r in rows])),
            "baseline_train_rmse": float(np.mean([r["baseline_train_rmse"] for r in rows])),
            "baseline_test_rmse": float(np.mean([r["baseline_test_rmse"] for r in rows])),
        })

    return averaged


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="MovieLens Gaussian vs Product vs Baseline (RMSE only, 5-run average)"
    )

    parser.add_argument("-N", "--training-data", type=int, default=800167)
    parser.add_argument("-b", "--batch-size", type=int, default=10000)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("-n", "--epochs", type=int, default=20)

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--sigma", type=float, default=0.6)
    parser.add_argument("-c", "--max-per-sample-grad_norm", type=float, default=5.0)

    parser.add_argument("--delta0", type=float, default=1e-12)
    parser.add_argument("-k", type=int, default=20000)

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--secure-rng", action="store_true", default=False)
    parser.add_argument("--data-root", type=str, default="../data")
    parser.add_argument("--save-dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-runs", type=int, default=5)

    args = parser.parse_args()
    device = torch.device(args.device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    temp_model = MovieLensModel(NUM_USERS, NUM_MOVIES)
    _, M = matched_product_sigma_M(
        temp_model,
        args.sigma,
        args.max_per_sample_grad_norm,
    )
    product_sigma_M = math.sqrt(M) * args.sigma * args.max_per_sample_grad_norm

    print("\n===== MovieLens Experiment Configuration =====")
    print("dataset=MovieLens-1M")
    print(f"device={device}")
    print(f"training_data={args.training_data}")
    print(f"epochs={args.epochs}")
    print(f"batch_size={args.batch_size}")
    print(f"test_batch_size={args.test_batch_size}")
    print(f"lr={args.lr}")
    print(f"gaussian_sigma={args.sigma}")
    print(f"max_grad_norm={args.max_per_sample_grad_norm}")
    print(f"delta0={args.delta0}")
    print(f"k={args.k}")
    print(f"model_dimension_M={M}")
    print(f"matched_product_sigma_M={product_sigma_M}")
    print(f"num_runs={args.num_runs}")
    print("=============================================\n")

    all_runs_results = []
    for run_idx in range(args.num_runs):
        run_seed = args.seed + run_idx
        run_results = run_one_experiment(args, run_seed)
        all_runs_results.append(run_results)

    averaged_results = average_results(all_runs_results, args.epochs)

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = save_dir / f"movielens_rmse_only_{now}.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "epoch",
            "gaussian_train_rmse",
            "gaussian_test_rmse",
            "product_train_rmse",
            "product_test_rmse",
            "baseline_train_rmse",
            "baseline_test_rmse",
        ])

        for row in averaged_results:
            writer.writerow([
                row["epoch"],
                row["gaussian_train_rmse"],
                row["gaussian_test_rmse"],
                row["product_train_rmse"],
                row["product_test_rmse"],
                row["baseline_train_rmse"],
                row["baseline_test_rmse"],
            ])

    print(f"\nAveraged CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()