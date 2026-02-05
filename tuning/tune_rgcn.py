"""
Optuna hyperparameter search for RGCN pretraining.

This script tunes RGCNEncoder on standard datasets by minimizing
validation link-prediction loss. It is designed to run before RL tuning.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from typing import Optional, Tuple

import optuna
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

from rgcn_rl_planner.data_loader import load_standard_dataset
from rgcn_rl_planner.models import RGCNEncoder
from rgcn_rl_planner.utils.data_processing import process_standard_kg
from rgcn_rl_planner.utils.seeding import set_seed


def train_one_epoch(
        encoder: RGCNEncoder,
        data: Data,
        optimizer: torch.optim.Optimizer,
        neg_sample_ratio: float,
        use_amp: bool,
        scaler: Optional[GradScaler],
) -> float:
    encoder.train()
    optimizer.zero_grad()

    with autocast('cuda', enabled=use_amp):
        z = encoder(data.edge_index, data.edge_type)
        pos_edge_index = data.edge_index
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=int(pos_edge_index.size(1) * neg_sample_ratio),
            method='sparse',
        )

        pos_logits = encoder.decode(z, pos_edge_index)
        neg_logits = encoder.decode(z, neg_edge_index)
        pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
        neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
        loss = pos_loss + neg_loss

    if use_amp and scaler:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    return loss.item()


def build_validation_edges(
        data: Data,
        valid_triplets,
        device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pos_edge_index = torch.tensor(
        [[h, t] for h, _, t in valid_triplets], dtype=torch.long
    ).t().contiguous().to(device)
    combined_edge_index = torch.cat([data.edge_index, pos_edge_index], dim=1)
    return pos_edge_index, combined_edge_index


def evaluate_loss(
        encoder: RGCNEncoder,
        data: Data,
        pos_edge_index: torch.Tensor,
        combined_edge_index: torch.Tensor,
        neg_sample_ratio: float,
        use_amp: bool,
        seed: int,
        device: torch.device,
) -> float:
    encoder.eval()
    with torch.no_grad():
        with autocast('cuda', enabled=use_amp):
            z = encoder(data.edge_index, data.edge_type)

            num_neg_samples = int(pos_edge_index.size(1) * neg_sample_ratio)
            with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
                torch.manual_seed(seed)
                neg_edge_index = negative_sampling(
                    edge_index=combined_edge_index,
                    num_nodes=data.num_nodes,
                    num_neg_samples=num_neg_samples,
                    method='sparse',
                )

            pos_logits = encoder.decode(z, pos_edge_index)
            neg_logits = encoder.decode(z, neg_edge_index)
            pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
            neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
            loss = pos_loss + neg_loss

    return loss.item()


def run_trial(trial: optuna.Trial, base_args: argparse.Namespace) -> float:
    args = base_args

    if args.dataset_type != 'standard':
        print("Optuna search for RGCN only supports 'standard' datasets.")
        return 1e9

    set_seed(args.seed, force_deterministic=args.force_deterministic)
    use_cuda = torch.cuda.is_available() and args.use_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    use_amp = use_cuda and args.use_amp
    scaler = GradScaler() if use_amp else None

    # Hyperparameters to tune
    embedding_dim = trial.suggest_categorical('embedding_dim', [64, 128, 256])
    hidden_channels = trial.suggest_categorical('hidden_channels', [16, 32, 64])
    out_channels = trial.suggest_categorical('out_channels', [16, 32, 64])
    num_bases = trial.suggest_categorical('num_bases', [None, 4, 8, 16])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 5e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    neg_sample_ratio = trial.suggest_float('neg_sample_ratio', 0.5, 5.0)

    data_root = os.path.join(args.data_dir, args.dataset_name)
    try:
        train_raw, valid_raw, test_raw = load_standard_dataset(data_root)
        data, _, relation_map, _, valid_triplets, _ = process_standard_kg(
            train_raw, valid_raw, test_raw
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: data load failed. {e}")
        return 1e9

    data = data.to(device)
    pos_edge_index, combined_edge_index = build_validation_edges(data, valid_triplets, device)

    encoder = RGCNEncoder(
        num_nodes=data.num_nodes,
        embedding_dim=embedding_dim,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_relations=len(relation_map),
        num_bases=num_bases,
    ).to(device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_loss = None
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(
            encoder, data, optimizer, neg_sample_ratio, use_amp, scaler
        )

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_loss = evaluate_loss(
                encoder,
                data,
                pos_edge_index,
                combined_edge_index,
                neg_sample_ratio,
                use_amp,
                seed=args.seed,
                device=device,
            )
            trial.report(val_loss, step=epoch)
            best_val_loss = val_loss if best_val_loss is None else min(best_val_loss, val_loss)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return best_val_loss if best_val_loss is not None else 1e9


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for RGCN pretraining")
    parser.add_argument('--dataset_type', type=str, default='standard', choices=['standard'], help='Dataset type')
    parser.add_argument('--dataset_name', type=str, default='NELL-995', help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='./data', help='Dataset root directory')

    parser.add_argument('--epochs', type=int, default=200, help='Training epochs per trial')
    parser.add_argument('--eval_every', type=int, default=50, help='Evaluate every N epochs')
    parser.add_argument('--num_trials', type=int, default=50, help='Number of Optuna trials')

    parser.add_argument('--seed', type=int, default=45, help='Random seed')
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--use_amp', action='store_true', help='Enable AMP for training')
    parser.add_argument('--force_deterministic', action='store_true', help='Force deterministic algorithms')
    parser.add_argument('--study_dir', type=str, default='./optuna', help='Directory to store Optuna DB files')

    args = parser.parse_args()

    os.makedirs(args.study_dir, exist_ok=True)
    storage_path = os.path.join(args.study_dir, f'optuna_studies_rgcn_{args.dataset_name}.db')

    study = optuna.create_study(
        direction='minimize',
        study_name=f'rgcn-hyperparam-search-{args.dataset_name}',
        storage=f'sqlite:///{storage_path}',
        load_if_exists=True,
    )

    objective = lambda trial: run_trial(trial, args)
    study.optimize(objective, n_trials=args.num_trials)

    print("\n--- Optuna RGCN search completed ---")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_value:.6f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
