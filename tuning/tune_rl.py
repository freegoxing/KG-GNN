"""
Optuna 超参数优化脚本 for RL 路径规划

核心功能:
- 使用 Optuna 自动化地为 RL 训练过程寻找最佳超参数组合。
- 定义一个 `objective` 函数，该该封装了一次完整的“训练+评估”流程。
- 在每次试验 (trial) 中，从预设的超参数空间中采样一组值。
- 以 MRR (Mean Reciprocal Rank) 作为优化目标，最大化该指标。
- 记录和展示最佳超参数和对应的 MRR 值。
"""
import sys
import os

# -- V3 修复: 解决 ModuleNotFoundError --
# 将项目根目录（'KG-GNN/'）添加到 Python 路径中
# 这样，无论从哪里运行脚本，都可以正确地找到 'rgcn_rl_planner' 和 'tests' 模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import random
from collections import defaultdict, deque
from typing import Dict, List, Any, Tuple

import optuna
import torch
from torch_geometric.data import Data

# 导入项目模块
from rgcn_rl_planner.data_loader import load_standard_dataset, load_custom_kg_from_json
from rgcn_rl_planner.models import RLPolicyNet
from rgcn_rl_planner.trainer import RLEnvironment, RLTrainer
from rgcn_rl_planner.utils.data_processing import process_standard_kg, calculate_pagerank, process_custom_kg
from rgcn_rl_planner.utils.seeding import set_seed


def run_training_trial(trial: optuna.Trial, default_args: argparse.Namespace) -> float:
    """
    Optuna 的目标函数，执行一次完整的训练和评估流程。

    Args:
        trial: Optuna 的 Trial 对象，用于建议超参数。
        default_args: 包含所有默认参数的命名空间。

    Returns:
        float: 本次试验的评估结果 (MRR)，Optuna 将据此进行优化。
    """
    args = default_args

    # --- 1. 定义超参数搜索空间 ---
    # 通过 trial 对象建议新的超参数值，覆盖默认值
    args.learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    args.discount_factor = trial.suggest_float('discount_factor', 0.9, 0.999)
    args.entropy_coeff = trial.suggest_float('entropy_coeff', 1e-3, 1e-1, log=True)
    args.gru_hidden_dim = trial.suggest_categorical('gru_hidden_dim', [16, 32, 64])
    args.reward_alpha = trial.suggest_float('reward_alpha', 0.05, 0.5)
    args.reward_eta = trial.suggest_float('reward_eta', 0.5, 2.0)
    args.action_pruning_k = trial.suggest_int('action_pruning_k', 5, 20)

    # --- 复用 train_rl.py 中的大部分逻辑 ---
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    set_seed(args.seed)

    data_root = os.path.join(args.data_dir, args.dataset_name)
    model_dir = os.path.join(args.model_root_dir, args.dataset_name, f"trial_{trial.number}")
    os.makedirs(model_dir, exist_ok=True)

    embedding_path = os.path.join(data_root, args.embedding_filename)
    node_map_path = os.path.join(data_root, args.node_map_filename)
    relation_map_path = os.path.join(data_root, args.relation_map_filename)

    try:
        # 仅使用 standard 类型的数据集进行超参数搜索，因为它有独立的验证集
        if args.dataset_type != 'standard':
            print("Optuna 搜索目前仅支持 'standard' 数据集类型。")
            return 0.0

        train_raw, valid_raw, test_raw = load_standard_dataset(data_root)
        data, entity_map, relation_map, train_triplets, valid_triplets, _ = process_standard_kg(
            train_raw, valid_raw, test_raw
        )

        all_triplets = train_triplets + valid_triplets
        edge_index_list = [[h, t] for h, r, t in all_triplets]
        edge_type_list = [r for h, r, t in all_triplets]
        node_embeddings = torch.load(embedding_path, map_location=torch.device('cpu'))

        data_for_rl_env = Data(
            x=node_embeddings,
            edge_index=torch.tensor(edge_index_list, dtype=torch.long).t().contiguous(),
            edge_type=torch.tensor(edge_type_list, dtype=torch.long),
            num_nodes=data.num_nodes
        )
        pagerank_values = calculate_pagerank(data_for_rl_env)

        node_embeddings = node_embeddings.to(device)
        data.x = node_embeddings
        data_for_rl_env = data_for_rl_env.to(device)
        embedding_dim = data.num_features

    except (FileNotFoundError, ValueError) as e:
        print(f"错误: 数据加载失败。 {e}")
        trial.report(0, 0)  # 报告失败
        return 0.0  # 返回一个差的分数

    # --- 2. 初始化模型、环境和训练器 ---
    model = RLPolicyNet(embedding_dim, args.gru_hidden_dim).to(device)
    env = RLEnvironment(
        data=data_for_rl_env,
        node_map=entity_map,
        relation_map=relation_map,
        node_embeddings=node_embeddings,
        max_path_length=args.max_path_length,
        pagerank_values=pagerank_values,
        optimal_path_length=args.optimal_path_length,
        reward_alpha=args.reward_alpha,
        reward_eta=args.reward_eta,
        length_reward_n=args.length_reward_n,
        length_reward_sigma=args.length_reward_sigma,
        action_pruning_k=args.action_pruning_k,
        low_freq_relations=set(),  # 在超参搜索中暂时不考虑低频关系
        low_freq_penalty=0.0,
        reward_clipping_value=args.reward_clipping_value,
        reward_ema_alpha=args.reward_ema_alpha,
        pagerank_exploration_steps=args.pagerank_exploration_steps,
    )
    trainer = RLTrainer(
        env, model, node_embeddings, device, args.learning_rate,
        args.discount_factor, args.entropy_coeff,
        args.use_scheduler, args.scheduler_step_size, args.scheduler_gamma,
        args.use_advantage_moving_average, args.advantage_ema_alpha
    )

    # --- 3. 创建训练和验证对 ---
    training_pairs = list(set([(h, t) for h, r, t in train_triplets if h != t]))
    if len(training_pairs) > args.num_training_pairs:
        training_pairs = random.sample(training_pairs, args.num_training_pairs)

    validation_pairs = list(set([(h, t) for h, r, t in valid_triplets if h != t]))
    # 同样可以对验证集进行采样，以加快评估速度
    if len(validation_pairs) > 500:  # 取 500 对进行验证
        validation_pairs = random.sample(validation_pairs, 500)

    if not training_pairs or not validation_pairs:
        print("错误：未能创建训练对或验证对。")
        trial.report(0, 0)
        return 0.0

    # --- 4. 开始训练 ---
    trainer.train(
        training_pairs, args.num_episodes, args.gradient_accumulation_steps,
        args.print_every, args.save_every, model_dir
    )

    # --- 5. 执行评估 ---
    # 这是关键步骤：在训练后，使用验证集评估模型性能
    mrr = trainer.run_evaluation(validation_pairs)
    print(f"Trial {trial.number} 完成 | MRR: {mrr:.4f}")

    # --- 6. 向 Optuna 报告结果 ---
    trial.report(mrr, step=args.num_episodes)

    # 检查是否应该被剪枝
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return mrr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna 超参数优化脚本 for RL")

    # 复用 train_rl.py 的所有参数，以便在需要时可以从命令行覆盖默认值
    # --- 数据集和路径 ---
    parser.add_argument('--dataset_type', type=str, default='standard', choices=['standard'], help='数据集类型')
    parser.add_argument('--dataset_name', type=str, default='NELL-995', help='数据集名称')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据根目录')
    parser.add_argument('--model_root_dir', type=str, default='./checkpoints/optuna_trials', help='Optuna试验根目录')
    parser.add_argument('--embedding_filename', type=str, default='node_embeddings.pt', help='节点嵌入文件名')
    parser.add_argument('--node_map_filename', type=str, default='node_map.json', help='节点映射文件名')
    parser.add_argument('--relation_map_filename', type=str, default='relation_map.json', help='关系映射文件名')

    # --- RL 模型和训练 ---
    parser.add_argument('--gru_hidden_dim', type=int, default=16, help='路径记忆 GRU 的隐藏层维度')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='优化器学习率')
    parser.add_argument('--discount_factor', type=float, default=0.99, help='奖励折扣因子')
    parser.add_argument('--entropy_coeff', type=float, default=0.05, help='熵损失系数')
    parser.add_argument('--max_path_length', type=int, default=10, help='最大路径长度')
    parser.add_argument('--action_pruning_k', type=int, default=None, help='Top-K 动作剪枝')
    parser.add_argument('--num_episodes', type=int, default=40000, help='训练 episodes 数')
    parser.add_argument('--num_training_pairs', type=int, default=1000, help='训练节点对数量')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=32, help='梯度累积步数')

    # --- 学习率调度器 ---
    parser.add_argument('--use_scheduler', action='store_true', help='启用学习率调度器')
    parser.add_argument('--scheduler_step_size', type=int, default=1000, help='调度器步长')
    parser.add_argument('--scheduler_gamma', type=float, default=0.95, help='调度器衰减因子')

    # --- 奖励函数 ---
    parser.add_argument('--optimal_path_length', type=int, default=None, help='钟形长度奖励中心')
    parser.add_argument('--reward_alpha', type=float, default=0.1, help='PageRank 奖励权重')
    parser.add_argument('--reward_eta', type=float, default=1.0, help='势能整形奖励权重')
    parser.add_argument('--length_reward_n', type=float, default=2.0, help='钟形长度奖励峰值')
    parser.add_argument('--length_reward_sigma', type=float, default=3.0, help='钟形长度奖励宽度')
    parser.add_argument('--reward_clipping_value', type=float, default=0.3, help='势能奖励裁剪值')
    parser.add_argument('--reward_ema_alpha', type=float, default=0.1, help='势能奖励EMA平滑系数')
    parser.add_argument('--pagerank_exploration_steps', type=int, default=3, help='PageRank奖励生效步数')

    # --- 优势方差缩减 ---
    parser.add_argument('--use_advantage_moving_average', action='store_true', help='启用优势移动平均标准化')
    parser.add_argument('--advantage_ema_alpha', type=float, default=0.01, help='优势移动平均的平滑系数')

    # --- 其他 ---
    parser.add_argument('--seed', type=int, default=45, help='随机种子')
    parser.add_argument('--use_cuda', action='store_true', help='强制使用 CUDA')
    parser.add_argument('--print_every', type=int, default=500, help='日志打印频率')
    parser.add_argument('--save_every', type=int, default=500, help='模型保存频率')
    parser.add_argument('--num_trials', type=int, default=50, help='Optuna 试验次数')
    parser.add_argument('--study_dir', type=str, default='./optuna_runs', help='Optuna DB 存储目录')

    default_args, _ = parser.parse_known_args()

    os.makedirs(default_args.study_dir, exist_ok=True)
    storage_path = os.path.join(
        default_args.study_dir, f'optuna_studies_rl_{default_args.dataset_name}.db'
    )

    # --- Optuna Study 设置 ---
    study = optuna.create_study(
        direction='maximize',
        study_name=f'rl-hyperparam-search-{default_args.dataset_name}',
        storage=f'sqlite:///{storage_path}',  # 将结果保存到数据库
        load_if_exists=True  # 如果数据库文件存在，则加载它
    )

    # 使用 lambda 将 default_args 传递给目标函数
    objective_func = lambda trial: run_training_trial(trial, default_args)

    # 开始优化
    study.optimize(objective_func, n_trials=default_args.num_trials)

    # --- 输出结果 ---
    print("\n--- Optuna 超参数搜索完成 ---")
    print(f"最佳试验 Trial: {study.best_trial.number}")
    print(f"最佳 MRR: {study.best_value:.4f}")
    print("最佳超参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
