"""
模型批量评估脚本 (V2, 解耦数据加载)

职责:
- 支持 "custom" (本项目) 和 "standard" (如 FB15k-237) 两种数据集类型。
- 根据数据集类型，使用 data_loader 和 data_utils 加载和处理数据。
- 加载预训练的节点嵌入和训练好的 RL 策略网络。
- 对 standard 类型数据集，使用测试集作为正样本，并生成负样本。
- 对 custom 类型数据集，沿用旧的采样逻辑。
- 收集并可视化关键性能指标。
"""
import json
import argparse
import os
import torch
from torch_geometric.data import Data
import numpy as np
import random
import glob
import re
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict, deque

# --- 动态导入 ---
# 根据环境和用户选择，决定使用 cuML 还是 scikit-learn
USE_CUML = False
try:
    from cuml.model_selection import KFold
    import cupy
    USE_CUML = True
    print("--- 已成功导入 cuML，将使用 GPU 进行交叉验证 ---")
except ImportError:
    from sklearn.model_selection import KFold
    print("--- 未找到 cuML 或 cupy，将使用 scikit-learn (CPU) 进行交叉验证 ---")

# --- 本项目模块导入 ---
from rgcn_rl_planner.models import RLPolicyNet
from rgcn_rl_planner.trainer import RLEnvironment
from tests.visualization import plot_metric_over_time
# 解耦后的数据加载和处理模块
from rgcn_rl_planner.data_loader import load_custom_kg_from_json, load_standard_dataset
from rgcn_rl_planner.data_utils import process_custom_kg, process_standard_kg, calculate_pagerank


# --- 核心函数 (大部分保持不变) ---

def has_path(start_node: int, end_node: int, adj: Dict[int, List[int]]) -> bool:
    """使用广度优先搜索 (BFS) 检查两个节点之间是否存在路径。"""
    if start_node == end_node:
        return True
    q = deque([start_node])
    visited = {start_node}
    while q:
        curr = q.popleft()
        if curr not in adj:
            continue
        for neighbor in adj[curr]:
            if neighbor == end_node:
                return True
            if neighbor not in visited:
                visited.add(neighbor)
                q.append(neighbor)
    return False


def _calculate_mrr_hits(ranks: List[int], hits_n: List[int] = [1, 3, 10]) -> Dict[str, float]:
    """
    计算 MRR 和 Hits@N 指标。
    Args:
        ranks: 真实目标在所有候选中的排名列表。
        hits_n: 列表，包含要计算的 Hits@N 值（例如 [1, 3, 10]）。
    Returns:
        包含 MRR 和 Hits@N 值的字典。
    """
    if not ranks:
        return {f"hits@{n}": 0.0 for n in hits_n}

    # 计算 MRR
    mrr = np.mean([1.0 / rank for rank in ranks])

    # 计算 Hits@N
    metrics = {f"hits@{n}": np.mean([1 if rank <= n else 0 for rank in ranks]) for n in hits_n}
    metrics["mrr"] = mrr
    return metrics


def get_path_details(start_node_id: int,
                     end_node_id: int,
                     model: RLPolicyNet,
                     env: RLEnvironment,
                     node_embeddings: torch.Tensor,
                     device: torch.device) -> Tuple[List[int], float, bool]:
    """
    使用训练好的策略网络在两个节点之间寻找路径，并返回路径详情。
    V2: 增加已访问节点跟踪，防止循环。
    """
    state = env.reset(start_node_id, end_node_id)
    path = [start_node_id]
    visited = {start_node_id}  # 跟踪已访问的节点
    total_reward = 0
    done = False
    with torch.no_grad():
        path_memory = torch.zeros(1, model.gru_hidden_dim, device=device)
        for _ in range(env.max_path_length):
            all_valid_actions = env.get_valid_actions()
            # 从有效动作中排除已访问的节点
            valid_actions = [action for action in all_valid_actions if action not in visited]

            if not valid_actions:
                break

            # 使用过滤后的动作列表
            action_dist, _, path_memory = model(
                node_embeddings[state].unsqueeze(0),
                node_embeddings[end_node_id],
                node_embeddings[valid_actions], # 只考虑未访问节点的嵌入
                path_memory
            )

            if action_dist is None:
                break
            
            # 选择概率最高的动作（贪心策略），注意索引要映射回原始 valid_actions
            action_index = action_dist.probs.argmax().item()
            action = valid_actions[action_index]

            next_state, reward, done, _ = env.step(action)

            # 更新路径和已访问集合
            path.append(next_state)
            visited.add(next_state)
            total_reward += reward
            state = next_state
            
            if done:
                break
    
    success = path[-1] == end_node_id
    return path, total_reward, success


def get_path_score(start_node_id: int,
                   target_node_id: int,
                   model: RLPolicyNet,
                   env: RLEnvironment,
                   node_embeddings: torch.Tensor,
                   device: torch.device) -> Tuple[float, bool]:
    """
    获取从起始节点到目标节点的路径分数（总奖励）和是否成功。
    """
    _, total_reward, success = get_path_details(start_node_id, target_node_id, model, env, node_embeddings, device)
    return total_reward, success


def evaluate_ranking_metrics(
        model: RLPolicyNet,
        env: RLEnvironment,
        eval_pairs_positive: List[Tuple[int, int]],
        all_known_triplets: Optional[Set[Tuple[int, int, int]]], # For filtered ranking in standard datasets
        adj: Dict[int, List[int]], # For has_path filtering
        num_entities: int,
        node_embeddings: torch.Tensor,
        device: torch.device,
        num_candidate_neg_samples: int = 99 # 1 true + num_candidate_neg_samples false
) -> Dict[str, float]:
    """
    对模型进行排名指标评估 (MRR, Hits@N)。
    Args:
        model: RL策略网络。
        env: RL环境。
        eval_pairs_positive: 正样本 (start_id, true_target_id) 对。
        all_known_triplets: 所有已知的三元组，用于Filtered Ranking。
        adj: 评估图的邻接列表，用于判断路径连通性。
        num_entities: 实体总数。
        node_embeddings: 节点嵌入。
        device: 设备。
        num_candidate_neg_samples: 每个正样本对应的负样本数量。
    Returns:
        包含 MRR 和 Hits@N 值的字典。
    """
    model.eval()
    ranks = []
    
    # 获取所有实体ID列表，用于负采样
    all_entity_ids = list(range(num_entities))

    for start_id, true_target_id in eval_pairs_positive:
        candidate_targets = []
        candidate_targets.append(true_target_id) # 添加真实目标

        # 生成负样本
        neg_samples_generated = 0
        attempts = 0
        max_attempts = num_candidate_neg_samples * 10 # 避免无限循环
        
        while neg_samples_generated < num_candidate_neg_samples and attempts < max_attempts:
            # 随机选择一个负样本
            false_target_id = random.choice(all_entity_ids)
            
            # 确保负样本不是真实目标
            if false_target_id == true_target_id:
                attempts += 1
                continue
            
            # 过滤：如果 (start_id, false_target_id) 存在真实路径，则不作为负样本
            # 注意: 这里的has_path是基于adj的，adj是评估图的边，包含了训练/验证/测试的边
            if has_path(start_id, false_target_id, adj):
                 attempts += 1
                 continue

            # 针对标准数据集，如果 false_target_id 曾经在训练/验证/测试集中与 start_id 形成过任何关系 (h,r,t)，则过滤
            # 这里的all_known_triplets是(h,r,t)形式，has_path是(h,t)形式
            # 这是一个更严格的过滤，确保 false_target_id 确实是“不应该被连接”的
            if all_known_triplets:
                found_known_relation = False
                for r_idx in range(len(env.relation_map)): # 使用 env.relation_map 的长度获取关系总数
                    if (start_id, r_idx, false_target_id) in all_known_triplets:
                        found_known_relation = True
                        break
                if found_known_relation:
                    attempts += 1
                    continue
            
            candidate_targets.append(false_target_id)
            neg_samples_generated += 1
            attempts += 1
        
        if neg_samples_generated < num_candidate_neg_samples:
            # 如果未能生成足够多的负样本，发出警告或调整策略
            print(f"警告: 为 ({start_id}, {true_target_id}) 生成的负样本不足 ({neg_samples_generated}/{num_candidate_neg_samples})")

        # 对所有候选目标进行评分
        target_scores = [] # 存储 (target_id, score, path_found)
        for cand_target_id in candidate_targets:
            score, path_found = get_path_score(start_id, cand_target_id, model, env, node_embeddings, device)
            target_scores.append((cand_target_id, score, path_found))
        
        # 排序: 优先成功路径，其次是高分，最后是节点ID (保持确定性)
        # 注意: 如果path_found为False，score可能为0或负数。为了排名准确，成功路径应该总是排在失败路径之前。
        target_scores.sort(key=lambda x: (x[2], x[1], -x[0]), reverse=True) # x[2] (path_found) True>False, x[1] (score) High->Low, -x[0] (target_id) Low->High for deterministic tie-break

        # 找到真实目标的排名
        rank = -1
        for i, (cand_id, _, _) in enumerate(target_scores):
            if cand_id == true_target_id:
                rank = i + 1
                break
        
        if rank != -1:
            ranks.append(rank)
        else:
            # 如果真实目标未被列入排名 (理论上不应发生，因为已明确添加)
            # 或者未成功找到路径导致分数过低排名非常靠后
            # 可以考虑将其排名设置为 num_candidate_neg_samples + 1 或 num_entities
            # 为了MRR/Hits@N的计算，假设它排在所有负样本之后
            ranks.append(num_candidate_neg_samples + 1)
            
    return _calculate_mrr_hits(ranks)


def evaluate_model_checkpoint(
        model: RLPolicyNet,
        env: RLEnvironment,
        eval_pairs_positive: List[Tuple[int, int]],
        adj: Dict[int, List[int]], # 新增: 传入邻接表
        num_entities: int, # 新增: 传入实体总数
        node_embeddings: torch.Tensor,
        device: torch.device,
        all_known_triplets: Optional[Set[Tuple[int, int, int]]] = None, # 新增: 传入所有已知三元组，用于Filtered Ranking
        num_candidate_neg_samples: int = 99 # 新增: 传入排名评估的负样本数量
) -> Dict[str, float]:
    """对单个模型检查点进行评估 (计算 MRR, Hits@N)。"""
    model.eval()

    # 直接调用 evaluate_ranking_metrics
    ranking_metrics = evaluate_ranking_metrics(
        model=model,
        env=env,
        eval_pairs_positive=eval_pairs_positive,
        all_known_triplets=all_known_triplets,
        adj=adj,
        num_entities=num_entities,
        node_embeddings=node_embeddings,
        device=device,
        num_candidate_neg_samples=num_candidate_neg_samples
    )
    return ranking_metrics

def main(args):
    """主评估函数 (V2)"""
    # --- 0. 环境和路径设置 ---
    global USE_CUML
    if not args.use_cuda:
        USE_CUML = False
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print(f"--- 使用设备: {device} ---")

    all_known_triplets_main: Optional[Set[Tuple[int, int, int]]] = None # 初始化以避免UnboundLocalError

    # 根据数据集名称动态设置路径
    data_root = os.path.join(args.data_dir, args.dataset_name)
    model_dir = os.path.join(args.model_dir, args.dataset_name)
    report_dir = os.path.join(args.report_dir, args.dataset_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    embedding_path = os.path.join(data_root, args.embedding_filename)
    node_map_path = os.path.join(data_root, args.node_map_filename)
    relation_map_path = os.path.join(data_root, args.relation_map_filename)
    save_plot_path = os.path.join(report_dir, args.plot_filename_base)

    # --- 1. 数据加载和处理 ---
    print(f"--- 正在加载和处理数据集: {args.dataset_name} ({args.dataset_type}) ---")
    try:
        if args.dataset_type == 'custom':
            raw_kg = load_custom_kg_from_json(os.path.join(data_root, "kg_data.json"))
            with open(node_map_path, 'r', encoding='utf-8') as f:
                entity_map = json.load(f)
            with open(relation_map_path, 'r', encoding='utf-8') as f:
                relation_map = json.load(f)
                data, entity_map, relation_map, pagerank_values = process_custom_kg(raw_kg, entity_map, relation_map)
                all_known_triplets_main = None # 自定义数据集没有all_known_triplets的概念
        elif args.dataset_type == 'standard':
            train_raw, valid_raw, test_raw = load_standard_dataset(data_root)
            data, entity_map, relation_map, train_triplets, valid_triplets, test_triplets = process_standard_kg(
                train_raw, valid_raw, test_raw
            )
            pagerank_values = calculate_pagerank(data)
            # 用于负采样的全量知识
            all_known_triplets = set(train_triplets + valid_triplets + test_triplets)
            all_known_triplets_main = all_known_triplets
        else:
            raise ValueError("`dataset_type` 必须是 'custom' 或 'standard'")

        node_embeddings = torch.load(embedding_path, map_location=device)
        data.x = node_embeddings
    except (FileNotFoundError, ValueError) as e:
        print(f"错误: 数据加载失败。 {e}")
        return

    data = data.to(device)
    embedding_dim = data.num_features
    print(f"数据加载完成: {data.num_nodes} 个节点, 嵌入维度: {embedding_dim}。")

    # --- 2. 准备邻接表 ---
    adj = defaultdict(list)
    for i in range(data.edge_index.size(1)):
        adj[data.edge_index[0, i].item()].append(data.edge_index[1, i].item())

    # --- 3. 准备评估节点对 ---
    print("--- 正在准备评估节点对 ---")
    if args.dataset_type == 'custom':
        inv_node_map = {v: k for k, v in entity_map.items()}
        basic_nodes = [int(inv_node_map[n['name']]) for n in raw_kg['nodes'] if n.get('type') == 'Basic_Knowledge']
        task_nodes = [int(inv_node_map[n['name']]) for n in raw_kg['nodes'] if n.get('type') == 'Task']

        if not basic_nodes or not task_nodes:
            print("错误：在自定义数据中未找到 'Basic_Knowledge' 或 'Task' 类型的节点。")
            return

        eval_pairs_positive = []
        while len(eval_pairs_positive) < args.num_eval_pairs:
            start, end = random.choice(basic_nodes), random.choice(task_nodes)
            if start != end and has_path(start, end, adj):
                eval_pairs_positive.append((start, end))

        eval_pairs_negative = []
        node_ids = list(range(data.num_nodes))
        while len(eval_pairs_negative) < args.num_eval_pairs:
            start, end = random.choice(node_ids), random.choice(node_ids)
            if start != end and not has_path(start, end, adj):
                eval_pairs_negative.append((start, end))

    else: # standard
        eval_pairs_positive = [(h, t) for h, r, t in test_triplets]
        # 如果数量太多，可以进行采样
        if len(eval_pairs_positive) > args.num_eval_pairs:
            eval_pairs_positive = random.sample(eval_pairs_positive, args.num_eval_pairs)

    print(f"已创建 {len(eval_pairs_positive)} 个正样本对。")

    # --- 4. 初始化环境和 K-Fold ---
    # 为了评估，需要在一个更丰富的图上进行，这个图包含训练集和验证集的边。
    # 这为测试集中的节点对提供了寻找路径的可能性。
    print("\n--- 正在为评估环境构建图 ---")
    
    if args.dataset_type == 'standard':
        # 在评估时，代理应该能够访问完整的图（训练+验证+测试），以确保所有测试对的路径都是理论上可达的。
        # 这是评估路径发现算法的标准做法。
        print(f"将使用 {len(train_triplets)} 个训练、{len(valid_triplets)} 个验证和 {len(test_triplets)} 个测试三元组构建评估图。")
        eval_graph_triplets = train_triplets + valid_triplets + test_triplets

        edge_index_list = [[h, t] for h, r, t in eval_graph_triplets]
        edge_type_list = [r for h, r, t in eval_graph_triplets]

        data_for_eval_env = Data(
            x=data.x,
            edge_index=torch.tensor(edge_index_list, dtype=torch.long).t().contiguous(),
            edge_type=torch.tensor(edge_type_list, dtype=torch.long),
            num_nodes=data.num_nodes
        ).to(device)
        
        print(f"评估图构建完成: {data_for_eval_env.num_edges} 条边。")
        
        print("--- 正在为评估图重新计算 PageRank ---")
        pagerank_for_eval = calculate_pagerank(data_for_eval_env)

    else: # For 'custom' type
        print("--- 正在为 'custom' 数据集评估使用原始训练图 ---")
        data_for_eval_env = data
        pagerank_for_eval = pagerank_values

    env = RLEnvironment(
        data=data_for_eval_env,
        node_map=entity_map,
        relation_map=relation_map,
        node_embeddings=node_embeddings,
        max_path_length=args.max_path_length,
        pagerank_values=pagerank_for_eval,
        optimal_path_length=args.optimal_path_length,
        reward_alpha=args.reward_alpha,
        reward_eta=args.reward_eta
    )
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)

    # --- 5. 查找并排序模型检查点 ---
    model_files = glob.glob(os.path.join(model_dir, args.model_name_pattern))
    if not model_files:
        print(f"错误: 在 '{model_dir}' 中未找到匹配 '{args.model_name_pattern}' 的模型文件。")
        return
    checkpoints = sorted([(int(re.search(r'_(\d+)\.pt$', f).group(1)), f)
                           for f in model_files if re.search(r'_(\d+)\.pt$', f)])
    print(f"找到 {len(checkpoints)} 个模型检查点进行评估。")

    # --- 6. 循环评估 ---
    history = defaultdict(list)
    model = RLPolicyNet(embedding_dim, args.gru_hidden_dim).to(device)

    for i, (episode, model_path) in enumerate(checkpoints):
        print(f"--- 评估模型 [{i+1}/{len(checkpoints)}]: {os.path.basename(model_path)} ---")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(f"警告: 无法加载模型 {model_path}，跳过。错误: {e}")
            continue

        # K-Fold 交叉验证
        fold_metrics = defaultdict(list)
        pos_splits = kf.split(np.array(eval_pairs_positive))

        for fold, (train_idx, test_idx) in enumerate(pos_splits): # Changed to iterate only pos_splits
            if USE_CUML:
                test_idx = test_idx.get()

            test_pos = np.array(eval_pairs_positive)[test_idx].tolist()

            metrics = evaluate_model_checkpoint(
                model, env, test_pos, adj, data.num_nodes, node_embeddings, device,
                all_known_triplets=all_known_triplets_main, # 使用main函数中的变量
                num_candidate_neg_samples=args.num_candidate_neg_samples
            )
            for key, value in metrics.items():
                fold_metrics[key].append(value)

        avg_metrics = {k: np.mean(v) for k, v in fold_metrics.items()}
        history["episodes"].append(episode)
        for key, value in avg_metrics.items(): history[key].append(value)
        # 更新打印输出，显示MRR和Hits@N
        print(f"  - MRR: {avg_metrics['mrr']:.4f}, Hits@1: {avg_metrics['hits@1']:.4f}, Hits@3: {avg_metrics['hits@3']:.4f}, Hits@10: {avg_metrics['hits@10']:.4f}")

    # --- 7. 绘制并保存结果 ---
    if args.save_plot and history["episodes"]:
        print(f"\n--- 正在生成并保存评估图表至 {report_dir} ---")
        plot_map = {
            "mrr": "MRR", "hits@1": "Hits@1", "hits@3": "Hits@3", "hits@10": "Hits@10"
        }
        for key, title_key in plot_map.items():
            if key in history:
                try:
                    path = f"{save_plot_path}_{title_key}.png"
                    plot_metric_over_time(history[key], f"{args.dataset_name} - {key}", history["episodes"], save_path=path)
                    print(f"图表已保存: {path}")
                except Exception as e:
                    print(f"错误: 无法保存图表 {path}。{e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL 模型批量评估脚本 (V2, 支持多数据集)")

    # 数据集和路径参数 (重要改动)
    parser.add_argument('--dataset_type', type=str, default='custom', choices=['custom', 'standard'], help='数据集类型')
    parser.add_argument('--dataset_name', type=str, default='default_kg', help='数据集名称 (将作为子目录名)')
    parser.add_argument('--data_dir', type=str, default='./data', help='存放所有数据集的根目录')
    parser.add_argument('--model_dir', type=str, default='./checkpoints', help='存放所有模型检查点的根目录')
    parser.add_argument('--report_dir', type=str, default='./reports', help='存放所有报告和图表的根目录')

    # 文件名参数 (在各自的数据集子目录中)
    parser.add_argument('--embedding_filename', type=str, default='node_embeddings.pt', help='节点嵌入文件名')
    parser.add_argument('--node_map_filename', type=str, default='node_map.json', help='节点/实体映射文件名')
    parser.add_argument('--relation_map_filename', type=str, default='relation_map.json', help='关系映射文件名')
    parser.add_argument('--model_name_pattern', type=str, default='rl_policy_net_episode_*.pt', help='模型文件的 glob 匹配模式')

    # 模型和评估参数
    parser.add_argument('--gru_hidden_dim', type=int, default=16, help='路径记忆 GRU 的隐藏层维度')
    parser.add_argument('--k_folds', type=int, default=5, help='K-fold 交叉验证的折数')
    parser.add_argument('--max_path_length', type=int, default=10, help='智能体探索的最大路径长度')
    parser.add_argument('--num_eval_pairs', type=int, default=100, help='用于评估的正样本对数量 (对于standard, 这是最大采样数)')
    parser.add_argument('--num_candidate_neg_samples', type=int, default=19, help='排名评估中每个正样本对应的负样本数量 (1 true + N false)')

    parser.add_argument('--use_cuda', action='store_true', help='启用 CUDA 和 cuML')

    # 奖励函数参数
    parser.add_argument('--optimal_path_length', type=int, default=None, help='鼓励探索的最佳路径长度')
    parser.add_argument('--reward_alpha', type=float, default=0.1, help='PageRank 奖励权重')
    parser.add_argument('--reward_eta', type=float, default=1.0, help='势能整形奖励的权重')

    # 可视化参数
    parser.add_argument('--save_plot', action='store_true', help='是否保存评估指标图表')
    parser.add_argument('--plot_filename_base', type=str, default='evaluation_summary_v1', help='保存图表的基础文件名')

    args = parser.parse_args()
    main(args)

