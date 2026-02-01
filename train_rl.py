"""
RL 策略网络训练脚本 (V2, 解耦数据加载)

职责：
- 支持 "custom" 和 "standard" 数据集类型。
- 加载预训练的节点嵌入和知识图谱结构。
- 根据数据集类型，智能地创建训练对 (h, t)。
- 初始化 RL 策略网络 (RLPolicyNet) 和强化学习环境。
- 启动 Actor-Critic 训练过程并保存模型检查点。
"""
import argparse
import json
import os
import random
from collections import defaultdict, deque
from typing import Dict, List

import torch
from torch_geometric.data import Data

from rgcn_rl_planner.data_loader import load_custom_kg_from_json, load_standard_dataset
from rgcn_rl_planner.data_utils import process_custom_kg, process_standard_kg, calculate_pagerank
# 本项目模块导入
from rgcn_rl_planner.models import RLPolicyNet
from rgcn_rl_planner.trainer import RLEnvironment, RLTrainer


def has_path(start_node: int, end_node: int, adj: Dict[int, List[int]]) -> bool:
    """使用广度优先搜索 (BFS) 检查两个节点之间是否存在路径。"""
    if start_node == end_node: return True
    q = deque([start_node])
    visited = {start_node}
    while q:
        curr = q.popleft()
        if curr not in adj: continue
        for neighbor in adj[curr]:
            if neighbor == end_node: return True
            if neighbor not in visited:
                visited.add(neighbor)
                q.append(neighbor)
    return False


def main(args):
    """主训练函数 (V2)"""
    # --- 1. 环境和路径设置 ---
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print(f"--- 使用设备: {device} ---")
    torch.manual_seed(args.seed)
    if device.type == 'cuda': torch.cuda.manual_seed(args.seed)

    # 根据数据集名称动态设置路径
    data_root = os.path.join(args.data_dir, args.dataset_name)
    model_dir = os.path.join(args.model_root_dir, args.dataset_name)
    os.makedirs(model_dir, exist_ok=True)

    embedding_path = os.path.join(data_root, args.embedding_filename)
    node_map_path = os.path.join(data_root, args.node_map_filename)
    relation_map_path = os.path.join(data_root, args.relation_map_filename)

    # --- 2. 加载和处理数据 ---
    print(f"--- 正在加载和处理数据集: {args.dataset_name} ({args.dataset_type}) ---")
    try:
        if args.dataset_type == 'custom':
            raw_kg = load_custom_kg_from_json(os.path.join(data_root, "kg_data.json"))
            with open(node_map_path, 'r', encoding='utf-8') as f:
                entity_map = json.load(f)
            with open(relation_map_path, 'r', encoding='utf-8') as f:
                relation_map = json.load(f)
            data, entity_map, relation_map, pagerank_values = process_custom_kg(raw_kg, entity_map, relation_map)
            # 对于 custom 类型，训练环境和图是一致的
            data_for_rl_env = data

        elif args.dataset_type == 'standard':
            train_raw, valid_raw, test_raw = load_standard_dataset(data_root)
            # **重要改动**: 现在接收所有三元组
            data, entity_map, relation_map, train_triplets, valid_triplets, test_triplets = process_standard_kg(
                train_raw, valid_raw, test_raw
            )

            # **V3 修复**: 为 RL 环境构建一个包含所有知识的完整图
            print("--- 正在为 RL 训练环境构建完整的知识图 ---")
            all_triplets = train_triplets + valid_triplets + test_triplets
            edge_index_list = [[h, t] for h, r, t in all_triplets]
            edge_type_list = [r for h, r, t in all_triplets]

            node_embeddings_for_data = torch.load(embedding_path, map_location=torch.device('cpu'))

            data_for_rl_env = Data(
                x=node_embeddings_for_data,  # 从 data 对象中获取 x
                edge_index=torch.tensor(edge_index_list, dtype=torch.long).t().contiguous(),
                edge_type=torch.tensor(edge_type_list, dtype=torch.long),
                num_nodes=data.num_nodes
            )
            print(f"RL 环境图构建完成: {data_for_rl_env.num_edges} 条边。")

            # 在完整的图上计算 PageRank
            pagerank_values = calculate_pagerank(data_for_rl_env)
        else:
            raise ValueError("`dataset_type` 必须是 'custom' 或 'standard'")

        # --- 2.1. [FB15k-237 specific] 识别低频关系 ---
        low_freq_relations = set()
        if args.dataset_type == 'standard' and args.low_freq_relation_threshold > 0:
            print(f"--- 正在识别频率低于 {args.low_freq_relation_threshold} 的低频关系 ---")
            relation_counts = defaultdict(int)
            for _, r, _ in train_triplets:
                relation_counts[r] += 1

            total_relations = len(relation_map)
            # 阈值可以是绝对数量或百分比，这里使用绝对数量
            for r_id, count in relation_counts.items():
                if count < args.low_freq_relation_threshold:
                    low_freq_relations.add(r_id)
            print(f"--- 已识别 {len(low_freq_relations)} 个低频关系 ---")


        node_embeddings = torch.load(embedding_path, map_location=device)
        # 确保两个 data 对象都有嵌入
        data.x = node_embeddings
        data_for_rl_env.x = node_embeddings

    except (FileNotFoundError, ValueError) as e:
        print(f"错误: 数据加载失败。 {e}")
        print("请确保您已为该数据集运行了 `train_rgcn.py` 来生成嵌入和映射文件。")
        return

    data = data.to(device)
    data_for_rl_env = data_for_rl_env.to(device)  # 将新图也移动到设备
    embedding_dim = data.num_features
    print(f"数据加载完成: {data.num_nodes} 个节点, 嵌入维度: {embedding_dim}。")

    # --- 3. 初始化模型和环境 ---
    print("--- 正在初始化 RLPolicyNet、环境和训练器 ---")
    model = RLPolicyNet(embedding_dim, args.gru_hidden_dim).to(device)

    # **V3 修复**: 使用为环境专门构建的 `data_for_rl_env`
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
        action_pruning_k=args.action_pruning_k,  # 新增: 传递剪枝参数
        low_freq_relations=low_freq_relations,  # 新增: 传递低频关系ID
        low_freq_penalty=args.low_freq_penalty,  # 新增: 传递低频关系惩罚值
        # --- 传递稳定性改进超参数 ---
        reward_clipping_value=args.reward_clipping_value,
        reward_ema_alpha=args.reward_ema_alpha,
        pagerank_exploration_steps=args.pagerank_exploration_steps,
    )
    trainer = RLTrainer(env, model, node_embeddings, device, args.learning_rate, args.discount_factor,
                        args.entropy_coeff, args.use_scheduler, args.scheduler_step_size, args.scheduler_gamma,
                        args.use_advantage_moving_average, args.advantage_ema_alpha)

    # --- 4. 创建训练对 ---
    print("--- 正在创建训练对 ---")
    training_pairs = []
    validation_pairs = []  # 新增: 初始化验证集

    if args.dataset_type == 'custom':
        adj = defaultdict(list)
        for i in range(data.edge_index.size(1)):
            adj[data.edge_index[0, i].item()].append(data.edge_index[1, i].item())
        inv_node_map = {v: k for k, v in entity_map.items()}
        basic_nodes = [int(inv_node_map[n['name']]) for n in raw_kg['nodes'] if n.get('type') == 'Basic_Knowledge']
        task_nodes = [int(inv_node_map[n['name']]) for n in raw_kg['nodes'] if n.get('type') == 'Task']

        if not basic_nodes or not task_nodes:
            print("错误：在自定义数据中未找到 'Basic_Knowledge' 或 'Task' 类型的节点。")
            return

        # 采样有真实路径的节点对进行训练
        while len(training_pairs) < args.num_training_pairs:
            start, end = random.choice(basic_nodes), random.choice(task_nodes)
            if start != end and has_path(start, end, adj):
                training_pairs.append((start, end))
    else:  # standard
        # 使用训练集中的 (头, 尾) 作为训练对
        training_pairs = list(set([(h, t) for h, r, t in train_triplets if h != t]))
        if len(training_pairs) > args.num_training_pairs:
            training_pairs = random.sample(training_pairs, args.num_training_pairs)

        # 新增: 使用验证集中的 (头, 尾) 作为验证对
        if args.use_early_stopping:
            validation_pairs = list(set([(h, t) for h, r, t in valid_triplets if h != t]))
            if len(validation_pairs) > args.num_validation_pairs:
                validation_pairs = random.sample(validation_pairs, args.num_validation_pairs)

    if not training_pairs:
        print("错误：未能创建任何训练对。")
        return
    print(f"已创建 {len(training_pairs)} 个训练对。")
    if validation_pairs:
        print(f"已创建 {len(validation_pairs)} 个验证对。")

    # --- 5. 开始训练 ---
    trainer.train(training_pairs, args.num_episodes, args.gradient_accumulation_steps,
                  args.print_every, args.save_every, model_dir,
                  validation_pairs, args.use_early_stopping, args.early_stopping_patience)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL 路径规划模型训练脚本 (V2)")

    # 数据集和路径参数
    parser.add_argument('--dataset_type', type=str, default='custom', choices=['custom', 'standard'], help='数据集类型')
    parser.add_argument('--dataset_name', type=str, default='my_custom_kg', help='数据集名称 (将作为子目录名)')
    parser.add_argument('--data_dir', type=str, default='./data', help='存放所有数据集的根目录')
    parser.add_argument('--model_root_dir', type=str, default='./checkpoints', help='存放所有模型检查点的根目录')

    # 文件名参数
    parser.add_argument('--embedding_filename', type=str, default='node_embeddings.pt', help='节点嵌入文件名')
    parser.add_argument('--node_map_filename', type=str, default='node_map.json', help='节点/实体映射文件名')
    parser.add_argument('--relation_map_filename', type=str, default='relation_map.json', help='关系映射文件名')

    # RL 模型和训练参数
    parser.add_argument('--gru_hidden_dim', type=int, default=16, help='路径记忆 GRU 的隐藏层维度')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='优化器学习率')
    parser.add_argument('--discount_factor', type=float, default=0.99, help='奖励折扣因子 (gamma)')
    parser.add_argument('--entropy_coeff', type=float, default=0.05, help='熵损失系数，鼓励探索')
    parser.add_argument('--max_path_length', type=int, default=10, help='智能体探索的最大路径长度')
    parser.add_argument('--action_pruning_k', type=int, default=None,
                        help='Top-K 动作剪枝, 限制每个步骤的动作空间大小 (默认: None, 不启用)')
    parser.add_argument('--num_episodes', type=int, default=40000, help='训练的总 episodes 数量')
    parser.add_argument('--num_training_pairs', type=int, default=1000,
                        help='用于训练的节点对数量 (对于custom是目标数，对于standard是最大采样数)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=32, help='梯度累积的步数')

    # 学习率调度器参数
    parser.add_argument('--use_scheduler', action='store_true', help='启用学习率调度器')
    parser.add_argument('--scheduler_step_size', type=int, default=1000, help='学习率调度器的步长 (episodes)')
    parser.add_argument('--scheduler_gamma', type=float, default=0.95, help='学习率调度器的衰减因子')

    # 早停参数 (NELL-995 specific)
    parser.add_argument('--use_early_stopping', action='store_true',
                        help='[NELL-995] 启用基于验证集 MRR 的早停机制')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                        help='[NELL-995] 早停的 patience (连续多少次验证无提升后停止)')
    parser.add_argument('--num_validation_pairs', type=int, default=500,
                        help='[NELL-995] 用于验证的节点对数量')

    # 奖励函数参数
    parser.add_argument('--optimal_path_length', type=int, default=None, help='钟形长度奖励的中心 (最佳路径长度)')
    parser.add_argument('--reward_alpha', type=float, default=0.1, help='PageRank 知识增益奖励权重')
    parser.add_argument('--reward_eta', type=float, default=1.0, help='势能整形奖励的权重')
    parser.add_argument('--length_reward_n', type=float, default=2.0, help='钟形长度奖励的峰值大小')
    parser.add_argument('--length_reward_sigma', type=float, default=3.0, help='钟形长度奖励的宽度 (标准差)')
    # --- 建议1&2 新增参数 ---
    parser.add_argument('--reward_clipping_value', type=float, default=0.3,
                        help='[建议1] 势能奖励单步变化的裁剪值 (设为0则不裁剪)')
    parser.add_argument('--reward_ema_alpha', type=float, default=0.1,
                        help='[建议1] 势能奖励EMA平滑系数')
    parser.add_argument('--pagerank_exploration_steps', type=int, default=3,
                        help='[建议2] PageRank奖励仅在每个episode的前N步生效')

    # 数据集特定改进参数
    parser.add_argument('--low_freq_relation_threshold', type=int, default=0,
                        help='[FB15k-237] 低频关系的绝对数量阈值 (默认: 0, 不启用)')
    parser.add_argument('--low_freq_penalty', type=float, default=0.0,
                        help='[FB15k-237] 应用于低频关系的奖励惩罚 (默认: 0.0)')

    # 其他
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--use_cuda', action='store_true', help='强制使用 CUDA (如果可用)')
    parser.add_argument('--print_every', type=int, default=500, help='每隔多少个 episode 打印一次日志')
    parser.add_argument('--save_every', type=int, default=500, help='每隔多少个 episode 保存一次模型检查点')

    # --- 新增: 优势方差缩减参数 ---
    parser.add_argument('--use_advantage_moving_average', action='store_true',
                        help='启用优势的移动平均标准化，以稳定训练。')
    parser.add_argument('--advantage_ema_alpha', type=float, default=0.01,
                        help='优势移动平均的平滑系数 (EMA alpha)。')

    args = parser.parse_args()
    main(args)
