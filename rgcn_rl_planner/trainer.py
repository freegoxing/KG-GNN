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
                 node_embeddings: torch.Tensor,  # 新增：传入节点嵌入
                 max_path_length: int,
                 pagerank_values: Dict[int, float],
                 optimal_path_length: Optional[int] = None,  # 新增: 鼓励探索的最佳路径长度
                 reward_alpha: float = 0.1,  # PageRank 奖励权重
                 reward_eta: float = 1.0,  # 新增：势能整形奖励的权重
                 length_reward_n: float = 2.0,  # 新增: 钟形长度奖励的峰值
                 length_reward_sigma: float = 3.0,  # 新增: 钟形长度奖励的宽度 (标准差)
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
        """获取合法动作"""
        neighbors_with_rels = self.adjacency_list.get(self.current_node, [])
        # 默认行为：返回所有未访问过的邻居
        return [n for n, rel in neighbors_with_rels if n not in self.visited]

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

        # 3. 终点奖励 (R_terminal)
        if self.current_node == self.target_node:
            # 3.1 基础成功奖励
            reward += 10.0

            # 3.2 新增：钟形路径长度奖励 (R_length)
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

    def train(self,
              training_pairs: List[Tuple[int, int]],
              num_episodes: int,
              gradient_accumulation_steps: int = 16,  # 新增：梯度累积步数
              print_every: int = 100,
              save_every: int = 200,
              model_save_dir: str = './checkpoints/'):
        """执行完整的训练循环, 并使用梯度累积。"""
        logging.info(f"--- 开始 RL 策略训练，共 {num_episodes} episodes on {self.device} ---")
        logging.info(f"--- 梯度将每 {gradient_accumulation_steps} episodes 累积更新一次 ---")

        success_count = 0
        total_rewards_list, path_lengths_list = [], []
        self.model.train()
        self.optimizer.zero_grad()  # 在训练开始前清零梯度

        for episode in range(num_episodes):
            start_node, target_node = training_pairs[episode % len(training_pairs)]

            # 运行 episode 并获取损失
            reward, path_len, success, loss = self.train_episode(start_node, target_node)

            if success:
                success_count += 1
            total_rewards_list.append(reward)
            path_lengths_list.append(path_len)

            # 如果 episode 产出了有效的损失，则进行累积
            if loss is not None:
                # 标准化损失
                normalized_loss = loss / gradient_accumulation_steps
                # 反向传播以累积梯度
                normalized_loss.backward()

            # --- 在累积足够多的梯度后，执行模型更新 ---
            if (episode + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()  # 更新后清零梯度，为下一批次做准备
                # 如果调度器存在，则更新学习率
                if self.scheduler:
                    self.scheduler.step()

            if (episode + 1) % print_every == 0:
                # 确保 print_every 是 accumulation steps 的倍数，以获取有意义的日志
                log_window = min(print_every, len(total_rewards_list))
                avg_reward = np.mean(total_rewards_list[-log_window:])
                avg_length = np.mean(path_lengths_list[-log_window:])
                success_rate = success_count / log_window
                logging.info(f"Episode {episode + 1}/{num_episodes} | "
                             f"Avg Reward (last {log_window}): {avg_reward:.4f} | "
                             f"Avg Path Length: {avg_length:.2f} | "
                             f"Success Rate: {success_rate:.2%}")
                success_count = 0

            if (episode + 1) % save_every == 0:
                os.makedirs(model_save_dir, exist_ok=True)
                checkpoint_filename = f"rl_policy_net_episode_{episode + 1}.pt"
                checkpoint_path = os.path.join(model_save_dir, checkpoint_filename)
                torch.save(self.model.state_dict(), checkpoint_path)
                logging.info(f"--- 模型检查点已保存至 {checkpoint_path} ---")

        logging.info("\n--- 训练完成 ---")
