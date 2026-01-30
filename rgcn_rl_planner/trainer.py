"""
强化学习环境与训练器 (重构版)

本模块包含 RLEnvironment 和 RLTrainer 两个核心类。
- RLEnvironment: 定义了智能体在知识图谱中导航的规则、状态、动作和奖励 (已移除手动权重)。
- RLTrainer: 封装了基于预训练嵌入的 Actor-Critic 算法的完整训练和更新逻辑。
"""

import logging
import os
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR  # 导入学习率调度器
from torch_geometric.data import Data

# 导入解耦后的 RLPolicyNet 模型
from .models import RLPolicyNet

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 类型注释 ---
NodeMap = Dict[int, str]
RelationMap = Dict[str, int]


class RLEnvironment:
    """
    强化学习环境 (重构版)。

    引入了基于嵌入空间的势能奖励整形 (Potential-Based Reward Shaping)，
    以提供更密集的引导信号，解决学习停滞问题。
    """

    def __init__(self,
                 data: Data,
                 node_map: NodeMap,
                 relation_map: RelationMap,
                 node_embeddings: torch.Tensor,  # 传入节点嵌入
                 max_path_length: int,
                 pagerank_values: Dict[int, float],
                 optimal_path_length: Optional[int] = None,  # 鼓励探索的最佳路径长度
                 reward_alpha: float = 0.1,  # PageRank 奖励权重
                 reward_eta: float = 1.0,  # 势能整形奖励的权重
                 length_reward_n: float = 2.0,  # 钟形长度奖励的峰值
                 length_reward_sigma: float = 3.0,  # 钟形长度奖励的宽度 (标准差)
                 action_pruning_k: Optional[int] = None,  # 新增: Top-K 动作剪枝的 K值
                 low_freq_relations: Optional[set] = None,  # 新增: 低频关系的 ID 集合
                 low_freq_penalty: float = 0.0,  # 新增: 对低频关系的惩罚值
                 ):
        self.data = data
        self.node_map = node_map
        self.relation_map = relation_map
        self.node_embeddings = node_embeddings
        self.num_nodes = data.num_nodes
        self.max_path_length = max_path_length
        self.pagerank_values = pagerank_values
        # 如果未提供最佳路径长度，则默认设置为最大长度的80%
        self.optimal_path_length = optimal_path_length if optimal_path_length is not None else int(
            max_path_length * 0.8)

        # 奖励函数超参数
        self.REWARD_ALPHA = reward_alpha
        self.REWARD_ETA = reward_eta  # 势能奖励权重
        self.LENGTH_REWARD_N = length_reward_n
        self.LENGTH_REWARD_SIGMA = length_reward_sigma
        self.action_pruning_k = action_pruning_k
        self.low_freq_relations = low_freq_relations if low_freq_relations is not None else set()
        self.low_freq_penalty = low_freq_penalty

        self.adjacency_list = self._build_adjacency_list()

        # Episode 状态
        self.current_node: int = -1
        self.target_node: int = -1
        self.path: List[int] = []
        self.visited: set = set()
        self.step_count: int = 0
        self.previous_potential: float = 0.0  # 用于计算势能奖励

    def _build_adjacency_list(self) -> Dict[int, List[Tuple[int, int]]]:
        """构建邻接表"""
        adj = defaultdict(list)
        edge_index = self.data.edge_index
        edge_type = self.data.edge_type
        for i in range(edge_index.shape[1]):
            src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
            rel = edge_type[i].item()
            adj[src].append((tgt, rel))
        return adj

    def _calculate_potential(self, node_id: int) -> float:
        """计算给定节点与目标节点之间的势能（余弦相似度）"""
        node_emb = self.node_embeddings[node_id]
        target_emb = self.node_embeddings[self.target_node]
        return F.cosine_similarity(node_emb.unsqueeze(0), target_emb.unsqueeze(0)).item()

    def reset(self, start_node: int, target_node: int) -> int:
        """重置环境"""
        self.current_node = start_node
        self.target_node = target_node
        self.path = [start_node]
        self.visited = {start_node}
        self.step_count = 0
        # 在 reset 时计算初始势能
        self.previous_potential = self._calculate_potential(self.current_node)
        return self.current_node

    def get_valid_actions(self) -> List[int]:
        """
        获取合法动作。
        如果配置了 action_pruning_k，则执行 Top-K 剪枝。
        """
        neighbors_with_rels = self.adjacency_list.get(self.current_node, [])
        unvisited_neighbors = [n for n, rel in neighbors_with_rels if n not in self.visited]

        # 如果没有设置K值，或者没有有效邻居，则返回所有未访问过的邻居
        if self.action_pruning_k is None or not unvisited_neighbors:
            return unvisited_neighbors

        # --- Top-K 动作剪枝 ---
        # 1. 计算所有有效邻居与目标节点的余弦相似度
        target_emb = self.node_embeddings[self.target_node].unsqueeze(0)
        neighbor_embs = self.node_embeddings[unvisited_neighbors]
        similarities = F.cosine_similarity(neighbor_embs, target_emb)

        # 2. 选择相似度最高的 Top-K 个邻居
        num_to_keep = min(self.action_pruning_k, len(unvisited_neighbors))
        if num_to_keep > 0:
            # `torch.topk` 返回 (values, indices)
            _, top_k_indices = torch.topk(similarities, k=num_to_keep)
            
            # 从原始列表中根据索引选出节点
            top_k_neighbors = [unvisited_neighbors[i] for i in top_k_indices]
            return top_k_neighbors
        else:
            return []

    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        """执行一步动作"""
        self.step_count += 1
        info = {}

        if action in self.visited:
            reward = -2.0
            done = True
            info["status"] = "revisit_termination"
            return self.current_node, reward, done, info

        # --- 更新状态 ---
        # 记录当前节点，以便查找对应的关系
        prev_node = self.current_node
        self.current_node = action
        self.path.append(action)
        self.visited.add(self.current_node)

        # 查找当前步所使用的关系类型
        relation_id = -1  # 默认值，如果没有找到有效关系
        for neighbor, rel in self.adjacency_list.get(prev_node, []):
            if neighbor == self.current_node:
                relation_id = rel
                break
        info["relation_id"] = relation_id

        # --- 计算复合奖励 ---
        reward = 0.0
        done = False

        # 1. 知识增益奖励 (R_gain) - 基于 PageRank
        r_gain = self.pagerank_values.get(action, 0.0)
        reward += self.REWARD_ALPHA * r_gain

        # 2. 核心改动：势能整形奖励 (R_shaping)
        current_potential = self._calculate_potential(self.current_node)
        r_shaping = current_potential - self.previous_potential
        reward += self.REWARD_ETA * r_shaping
        self.previous_potential = current_potential  # 更新势能

        # 3. 新增: 低频关系惩罚
        if relation_id in self.low_freq_relations:
            reward -= self.low_freq_penalty

        # 5. 终点奖励 (R_terminal)
        if self.current_node == self.target_node:
            # 5.1 基础成功奖励
            reward += 10.0

            # 5.2 新增：钟形路径长度奖励 (R_length)
            # 仅在成功到达终点时，根据路径长度与“最佳长度”的接近程度给予奖励
            path_length_diff = self.step_count - self.optimal_path_length
            r_length = self.LENGTH_REWARD_N * np.exp(-(path_length_diff ** 2) / (2 * self.LENGTH_REWARD_SIGMA ** 2))
            reward += r_length

            done = True
            info["status"] = "success"
        elif self.step_count >= self.max_path_length:
            reward -= 2.0
            done = True
            info["status"] = "max_length_reached"

        return self.current_node, reward, done, info


class RLTrainer:
    """
    强化学习训练器 (重构版)。
    使用预训练的节点嵌入进行训练。
    """
    def __init__(self,
                 environment: RLEnvironment,
                 model: RLPolicyNet,
                 node_embeddings: torch.Tensor,
                 device: torch.device,
                 learning_rate: float = 0.001,
                 discount_factor: float = 0.99,
                 entropy_coeff: float = 0.01,
                 use_scheduler: bool = False,
                 scheduler_step_size: int = 5000,
                 scheduler_gamma: float = 0.95):
        self.env = environment
        self.model = model
        self.node_embeddings = node_embeddings
        self.device = device
        self.discount_factor = discount_factor
        self.entropy_coeff = entropy_coeff
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        self.scheduler = None
        if use_scheduler:
            self.scheduler = StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
            logging.info(f"已启用学习率调度器: StepLR(step_size={scheduler_step_size}, gamma={scheduler_gamma})")

        logging.info(f"RLTrainer initialized with device: {self.device}")

    def train_episode(self, start_node: int, target_node: int) -> Tuple[float, int, bool, Optional[torch.Tensor]]:
        """
        在一个完整的 episode 中运行，计算损失但不更新模型。
        返回: (总奖励, 路径长度, 是否成功, 计算出的损失张量)
        """
        state = self.env.reset(start_node, target_node)
        log_probs, values, rewards, entropies = [], [], [], []

        path_memory = torch.zeros(1, self.model.gru_hidden_dim, device=self.device)

        for _ in range(self.env.max_path_length):
            valid_actions = self.env.get_valid_actions()
            if not valid_actions:
                break

            current_emb = self.node_embeddings[state].unsqueeze(0)
            target_emb = self.node_embeddings[target_node]
            neighbor_embs = self.node_embeddings[valid_actions]

            action_dist, value, path_memory = self.model(
                current_emb, target_emb, neighbor_embs, path_memory
            )

            if action_dist is None:
                break

            action_idx = action_dist.sample()
            action = valid_actions[action_idx.item()]

            log_prob = action_dist.log_prob(action_idx)
            entropy = action_dist.entropy()

            state, reward, done, info = self.env.step(action)

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            entropies.append(entropy)

            if done:
                break

        if not rewards:
            return 0.0, 0, False, None

        # --- 计算损失但不更新 ---
        returns = []
        cumulative_return = 0.0
        for r in reversed(rewards):
            cumulative_return = r + self.discount_factor * cumulative_return
            returns.insert(0, cumulative_return)

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        log_probs_tensor = torch.stack(log_probs)
        values_tensor = torch.stack(values)
        entropies_tensor = torch.stack(entropies)

        advantages = returns - values_tensor.detach()
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = -(log_probs_tensor * advantages).mean()
        value_loss = F.mse_loss(values_tensor, returns)
        entropy_loss = -self.entropy_coeff * entropies_tensor.mean()

        total_loss = policy_loss + value_loss + entropy_loss

        success = self.env.current_node == self.env.target_node
        return sum(rewards), len(self.env.path) - 1, success, total_loss

    def evaluate_episode(self, start_node: int, target_node: int) -> Tuple[bool, int]:
        """
        在评估模式下运行单个 episode (使用贪心策略)。
        返回: (是否成功, 路径长度)
        """
        self.model.eval()  # 设置为评估模式
        with torch.no_grad():
            state = self.env.reset(start_node, target_node)
            path_memory = torch.zeros(1, self.model.gru_hidden_dim, device=self.device)

            for _ in range(self.env.max_path_length):
                valid_actions = self.env.get_valid_actions()
                if not valid_actions:
                    break  # 死胡同

                current_emb = self.node_embeddings[state].unsqueeze(0)
                target_emb = self.node_embeddings[target_node]
                neighbor_embs = self.node_embeddings[valid_actions]

                action_dist, _, path_memory = self.model(
                    current_emb, target_emb, neighbor_embs, path_memory
                )

                if action_dist is None:
                    break

                # 贪心选择：选择概率最高的动作
                action_idx = torch.argmax(action_dist.probs)
                action = valid_actions[action_idx.item()]

                state, _, done, _ = self.env.step(action)
                if done:
                    break
        
        success = self.env.current_node == self.env.target_node
        path_len = len(self.env.path) - 1
        return success, path_len

    def run_evaluation(self, validation_pairs: List[Tuple[int, int]]) -> float:
        """
        在验证集上运行评估并计算 MRR。
        简单定义：成功路径的 MRR = 1 / 路径长度。失败为 0。
        """
        if not validation_pairs:
            return 0.0
        
        total_reciprocal_rank = 0.0
        for start_node, target_node in validation_pairs:
            success, path_len = self.evaluate_episode(start_node, target_node)
            if success and path_len > 0:
                total_reciprocal_rank += 1.0 / path_len
        
        mrr = total_reciprocal_rank / len(validation_pairs)
        return mrr

    def train(self,
              training_pairs: List[Tuple[int, int]],
              num_episodes: int,
              gradient_accumulation_steps: int = 16,
              print_every: int = 100,
              save_every: int = 200,
              model_save_dir: str = './checkpoints/',
              validation_pairs: Optional[List[Tuple[int, int]]] = None,
              use_early_stopping: bool = False,
              early_stopping_patience: int = 3):
        """执行完整的训练循环, 并使用梯度累积、早停和验证。"""
        logging.info(f"--- 开始 RL 策略训练，共 {num_episodes} episodes on {self.device} ---")
        logging.info(f"--- 梯度将每 {gradient_accumulation_steps} episodes 累积更新一次 ---")

        if use_early_stopping:
            if not validation_pairs:
                logging.warning("警告: 开启了早停但未提供验证集, 早停将不会生效。")
                use_early_stopping = False
            else:
                logging.info(f"--- 早停机制已启用, Patience: {early_stopping_patience}, "
                             f"每 {save_every} episodes 验证一次 ---")

        success_count = 0
        total_rewards_list, path_lengths_list = [], []
        best_mrr = -1.0
        patience_counter = 0

        self.optimizer.zero_grad()

        for episode in range(num_episodes):
            self.model.train()  # 确保在训练模式
            start_node, target_node = training_pairs[episode % len(training_pairs)]

            reward, path_len, success, loss = self.train_episode(start_node, target_node)

            if success:
                success_count += 1
            total_rewards_list.append(reward)
            path_lengths_list.append(path_len)

            if loss is not None:
                normalized_loss = loss / gradient_accumulation_steps
                normalized_loss.backward()

            if (episode + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler:
                    self.scheduler.step()

            if (episode + 1) % print_every == 0:
                log_window = min(print_every, len(total_rewards_list))
                avg_reward = np.mean(total_rewards_list[-log_window:])
                avg_length = np.mean(path_lengths_list[-log_window:])
                success_rate = success_count / log_window
                logging.info(f"Episode {episode + 1}/{num_episodes} | "
                             f"Avg Reward (last {log_window}): {avg_reward:.4f} | "
                             f"Avg Path Length: {avg_length:.2f} | "
                             f"Success Rate: {success_rate:.2%}")
                success_count = 0

            # --- 验证、早停与保存逻辑 ---
            if (episode + 1) % save_every == 0:
                # 1. 无条件保存周期性检查点
                os.makedirs(model_save_dir, exist_ok=True)
                checkpoint_filename = f"rl_policy_net_episode_{episode + 1}.pt"
                checkpoint_path = os.path.join(model_save_dir, checkpoint_filename)
                torch.save(self.model.state_dict(), checkpoint_path)
                logging.info(f"--- 模型检查点已保存至 {checkpoint_path} ---")

                # 2. 如果启用早停，则执行验证
                if use_early_stopping and validation_pairs:
                    logging.info(f"--- Episode {episode + 1}: 开始在验证集上进行评估 ---")
                    current_mrr = self.run_evaluation(validation_pairs)
                    logging.info(f"--- 验证 MRR: {current_mrr:.6f} | 历史最佳 MRR: {best_mrr:.6f} ---")

                    # 仅更新计数器和最佳分数，不改变保存行为
                    if current_mrr > best_mrr:
                        best_mrr = current_mrr
                        patience_counter = 0
                        logging.info(f"--- 新的最佳 MRR！Patience 重置为 0。 ---")
                    else:
                        patience_counter += 1
                        logging.info(f"--- MRR 未提升。Patience: {patience_counter}/{early_stopping_patience} ---")

                    if patience_counter >= early_stopping_patience:
                        logging.info(f"--- 早停触发！连续 {early_stopping_patience} 次验证性能未提升。训练终止。 ---")
                        break  # 提前结束训练

        logging.info("\n--- 训练完成 ---")
        if use_early_stopping:
            logging.info(f"--- 最终最佳验证 MRR: {best_mrr:.6f} ---")
