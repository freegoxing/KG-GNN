"""
神经网络模型定义模块 (重构版)

存放解耦后的 RGCN 编码器和 RL 策略网络。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv
from typing import List, Tuple, Optional


class RGCNEncoder(nn.Module):
    """
    R-GCN 编码器模型 (V2)。

    职责: 学习知识图谱中节点和关系的嵌入表示。
    使用一个可学习的嵌入层来替代 one-hot 编码，以节省内存。
    """
    def __init__(self, num_nodes: int, embedding_dim: int, hidden_channels: int, out_channels: int, num_relations: int, num_bases: Optional[int] = None):
        """
        初始化 R-GCN 编码器层。

        Args:
            num_nodes (int): 图中的节点总数。
            embedding_dim (int): 节点嵌入的维度。
            hidden_channels (int): R-GCN 隐藏层维度。
            out_channels (int): 输出嵌入维度。
            num_relations (int): 关系（边类型）的数量。
            num_bases (Optional[int]): R-GCN 的基分解数量。用于减少参数量。
        """
        super().__init__()
        # 可学习的节点嵌入层，替换了庞大的 one-hot 特征矩阵
        self.embedding = torch.nn.Embedding(num_nodes, embedding_dim)

        self.rgcn1 = RGCNConv(embedding_dim, hidden_channels, num_relations, num_bases=num_bases)
        self.rgcn2 = RGCNConv(hidden_channels, out_channels, num_relations, num_bases=num_bases)
        self.dropout = nn.Dropout(0.5)

        # 初始化嵌入权重
        nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """
        前向传播，计算所有节点的嵌入。
        注意：输入不再需要 `x` 特征矩阵。
        """
        # 直接从嵌入层获取所有节点的特征
        x = self.embedding.weight
        
        # 后续传播与之前相同
        x = F.relu(self.rgcn1(x, edge_index, edge_type))
        x = self.dropout(x)
        x = self.rgcn2(x, edge_index, edge_type)
        return x

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        解码器，用于链接预测任务。
        """
        head_nodes = z[edge_index[0]]
        tail_nodes = z[edge_index[1]]
        # 逐元素相乘后求和，得到每条边的分数
        logits = (head_nodes * tail_nodes).sum(dim=1)
        return logits


class RLPolicyNet(nn.Module):
    """
    强化学习策略网络 (Actor-Critic)。

    职责: 基于预训练好的节点嵌入进行路径规划决策。
    这是一个纯粹的 RL 模型，不包含图卷积层。
    """

    def __init__(self, embedding_dim: int, gru_hidden_dim: int):
        """
        初始化 RL 策略网络。

        Args:
            embedding_dim (int): 输入节点嵌入的维度 (应与 RGCNEncoder 的 out_channels 匹配)。
            gru_hidden_dim (int): GRU 隐藏状态的维度，用于编码路径记忆。
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.gru_hidden_dim = gru_hidden_dim

        # GRU 单元，用于编码路径历史
        # GRU 的输入是当前节点的嵌入，隐藏状态代表了整个路径的记忆
        self.gru_cell = nn.GRUCell(embedding_dim, gru_hidden_dim)

        # 策略头 (Actor Head)
        # 输入: path_memory (GRU hidden) + neighbor_emb + target_emb
        self.policy_head = nn.Linear(gru_hidden_dim + embedding_dim * 2, 1)

        # 价值头 (Value Head)
        # 输入: path_memory (GRU hidden) + target_emb
        self.value_head = nn.Linear(gru_hidden_dim + embedding_dim, 1)

    def forward(self,
                current_emb: torch.Tensor,
                target_emb: torch.Tensor,
                neighbor_embs: torch.Tensor,
                path_memory: torch.Tensor
                ) -> Tuple[Optional[torch.distributions.Categorical], torch.Tensor, torch.Tensor]:
        """
        前向传播，计算策略和价值。

        Args:
            current_emb (torch.Tensor): 当前节点的嵌入向量, shape: [1, embedding_dim]。
            target_emb (torch.Tensor): 目标节点的嵌入向量, shape: [embedding_dim]。
            neighbor_embs (torch.Tensor): 所有有效邻居节点的嵌入矩阵, shape: [num_neighbors, embedding_dim]。
            path_memory (torch.Tensor): 上一步的路径记忆 (GRU hidden_state), shape: [1, gru_hidden_dim]。

        Returns:
            action_dist (Optional[torch.distributions.Categorical]): 动作概率分布。
            value (torch.Tensor): 当前状态的价值。
            next_path_memory (torch.Tensor): 更新后的路径记忆。
        """
        # 1. 更新路径记忆 (GRU)
        next_path_memory = self.gru_cell(current_emb, path_memory)  # [1, gru_hidden_dim]

        # 2. 价值头 (Critic)
        # 状态表示 = 路径记忆 + 目标节点嵌入
        value_state_repr = torch.cat([path_memory.squeeze(0), target_emb])
        value = self.value_head(value_state_repr)

        # 3. 策略头 (Actor)
        if neighbor_embs.shape[0] == 0:
            # 如果没有有效动作，返回空的分布
            return None, value.squeeze(-1), next_path_memory

        # 为每个有效动作（邻居）计算一个分数 (logit)
        # 策略网络输入 = 路径记忆 + 邻居嵌入 + 目标嵌入
        num_neighbors = neighbor_embs.shape[0]

        # 将路径记忆和目标嵌入重复，以匹配邻居数量
        path_memory_repeated = next_path_memory.repeat(num_neighbors, 1)        # [num_neighbors, gru_hidden_dim]
        target_emb_repeated = target_emb.unsqueeze(0).repeat(num_neighbors, 1)  # [num_neighbors, embedding_dim]

        # 拼接以形成策略网络的输入
        policy_input = torch.cat([path_memory_repeated, neighbor_embs, target_emb_repeated], dim=1)

        # 计算每个邻居的 logit
        logits = self.policy_head(policy_input).squeeze(-1)  # [num_neighbors]

        # 基于 logits 创建动作概率分布
        action_dist = torch.distributions.Categorical(logits=logits)

        return action_dist, value.squeeze(-1), next_path_memory