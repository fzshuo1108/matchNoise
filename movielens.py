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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from opacus import PrivacyEngine
from opacus.accountants import GaussianAccountant, PRVAccountant

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
def evaluate_rmse(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    losses = []
    all_preds = []
    all_targets = []

    for users, movies, ratings in tqdm(test_loader, leave=False):
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
def train_one_epoch_with_extra_accountants(
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

        for acc_obj in extra_accountants:
            acc_obj.step(
                noise_multiplier=noise_multiplier,
                sample_rate=sample_rate,
            )

    train_loss = float(np.mean(losses)) if losses else 0.0
    train_rmse = float(np.sqrt(mean_squared_error(all_targets, all_preds)))
    return train_loss, train_rmse


def train_one_epoch_product(
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


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="MovieLens Gaussian vs Product Noise")

    parser.add_argument("-N", "--training-data", type=int, default=800167)
    parser.add_argument("-b", "--batch-size", type=int, default=10000)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("-n", "--epochs", type=int, default=20)

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--sigma", type=float, default=0.6)
    parser.add_argument("-c", "--max-per-sample-grad_norm", type=float, default=5.0)

    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--delta0", type=float, default=1e-12)
    parser.add_argument("-k", type=int, default=20000)

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--secure-rng", action="store_true", default=False)
    parser.add_argument("--data-root", type=str, default="../data")
    parser.add_argument("--save-dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Data
    # -----------------------------
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

    # -----------------------------
    # Two models
    # -----------------------------
    base_model = MovieLensModel(NUM_USERS, NUM_MOVIES)

    gaussian_model = copy.deepcopy(base_model).to(device)
    product_model = copy.deepcopy(base_model).to(device)

    gaussian_optimizer = optim.Adam(gaussian_model.parameters(), lr=args.lr)
    product_optimizer = optim.Adam(product_model.parameters(), lr=args.lr)

    # -----------------------------
    # Gaussian side
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
    # -----------------------------
    csv_path = save_dir / "movielens_epoch_results_rmse_multi_accountants.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)

    writer.writerow([
        "epoch",
        "gaussian_epsilon_ma",
        "gaussian_epsilon_clt",
        "gaussian_epsilon_prv",
        "product_epsilon",
        "gaussian_test_rmse",
        "product_test_rmse",
    ])

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
    print(f"delta={args.delta}")
    print(f"delta0={args.delta0}")
    print(f"k={args.k}")
    print(f"model_dimension_M={M}")
    print(f"matched_product_sigma_M={product_sigma_M}")
    print("=============================================\n")

    # -----------------------------
    # Train
    # -----------------------------
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        g_train_loss, g_train_rmse = train_one_epoch_with_extra_accountants(
            model=gaussian_model,
            device=device,
            train_loader=train_loader_g,
            optimizer=gaussian_optimizer,
            extra_accountants=[gaussian_clt_accountant, gaussian_prv_accountant],
            noise_multiplier=args.sigma,
            sample_rate=gaussian_sample_rate,
        )

        p_train_loss, p_train_rmse = train_one_epoch_product(
            model=product_model,
            device=device,
            train_loader=train_loader_p,
            optimizer=product_optimizer,
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

        g_eps_ma = gaussian_ma_accountant.get_epsilon(args.delta)
        g_eps_clt = gaussian_clt_accountant.get_epsilon(args.delta)
        g_eps_prv = gaussian_prv_accountant.get_epsilon(args.delta)

        p_eps = product_engine.accountant.get_epsilon(args.delta)

        print(
            f"[Gaussian] train_loss={g_train_loss:.6f} "
            f"train_rmse={g_train_rmse:.6f} "
            f"test_loss={g_test_loss:.6f} "
            f"test_rmse={g_test_rmse:.6f}"
        )
        print(
            f"[Gaussian eps] MA/RDP={g_eps_ma:.6f}, "
            f"CLT/GDP={g_eps_clt:.6f}, "
            f"PRV={g_eps_prv:.6f}"
        )
        print(
            f"[Product ] train_loss={p_train_loss:.6f} "
            f"train_rmse={p_train_rmse:.6f} "
            f"test_loss={p_test_loss:.6f} "
            f"test_rmse={p_test_rmse:.6f} "
            f"eps={p_eps:.6f}"
        )

        writer.writerow([
            epoch,
            g_eps_ma,
            g_eps_clt,
            g_eps_prv,
            p_eps,
            g_test_rmse,
            p_test_rmse,
        ])
        csv_file.flush()

    csv_file.close()
    print(f"\nCSV results saved to: {csv_path}")


if __name__ == "__main__":
    main()