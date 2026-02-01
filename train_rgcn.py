"""
RGCN 编码器预训练脚本 (V3, OOM 优化版)

职责：
- 支持 "custom" (本项目) 和 "standard" (如 FB15k-237) 两种数据集类型。
- 通过链接预测 (Link Prediction) 任务对 RGCNEncoder 模型进行预训练。
- 学习并保存反映知识图谱结构和语义的节点嵌入 (Node Embeddings) 及映射文件。
- 集成 OOM 解决方案：分离损失计算、负采样率控制、AMP 混合精度。
"""
import argparse
import os
from typing import Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

# 解耦后的数据加载和处理模块
from rgcn_rl_planner.data_loader import load_custom_kg_from_json, load_standard_dataset
from rgcn_rl_planner.utils.data_processing import process_custom_kg, process_standard_kg, save_mappings
from rgcn_rl_planner.models import RGCNEncoder
from rgcn_rl_planner.utils.seeding import set_seed


def train(encoder: RGCNEncoder,
          data: Data,
          optimizer: torch.optim.Optimizer,
          device: torch.device,
          neg_sample_ratio: float,
          use_amp: bool,
          scaler: Optional[GradScaler] = None) -> float:
    """
    执行一个训练周期的逻辑。(V3: OOM 优化)
    """
    encoder.train()
    optimizer.zero_grad()

    # 根据是否启用 AMP，选择性地使用 autocast
    with autocast('cuda', enabled=use_amp):
        # 直接从模型获取嵌入，不再传入 x
        z = encoder(data.edge_index, data.edge_type)

        pos_edge_index = data.edge_index
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            # 根据比例计算负采样数量
            num_neg_samples=int(pos_edge_index.size(1) * neg_sample_ratio),
            method='sparse'
        )

        # --- OOM 优化 1: 分离正负样本的损失计算 ---
        pos_logits = encoder.decode(z, pos_edge_index)
        neg_logits = encoder.decode(z, neg_edge_index)

        pos_loss = F.binary_cross_entropy_with_logits(
            pos_logits, torch.ones_like(pos_logits)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_logits, torch.zeros_like(neg_logits)
        )
        loss = pos_loss + neg_loss

    # --- OOM 优化 2: 使用 AMP (混合精度) ---
    if use_amp and scaler:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    return loss.item()


def main(args):
    """主训练函数 (V4 - 集成 OOM 优化)"""
    # --- 1. 环境和路径设置 ---
    use_cuda = torch.cuda.is_available() and args.use_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"--- 使用设备: {device} ---")
    set_seed(args.seed, force_deterministic=args.force_deterministic)

    # OOM 优化 2: 仅在 CUDA 环境下启用 AMP
    use_amp = use_cuda and args.use_amp
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("--- 自动混合精度 (AMP) 已启用 ---")

    # 根据数据集名称动态设置路径
    data_root = os.path.join(args.data_dir, args.dataset_name)
    model_dir = os.path.join(args.model_root_dir, args.dataset_name)
    os.makedirs(data_root, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    model_save_path = os.path.join(model_dir, args.model_filename)
    embedding_save_path = os.path.join(data_root, args.embedding_filename)
    entity_map_path = os.path.join(data_root, args.node_map_filename)
    relation_map_path = os.path.join(data_root, args.relation_map_filename)

    # --- 2. 加载和预处理数据 ---
    print(f"--- 正在加载和处理数据集: {args.dataset_name} ({args.dataset_type}) ---")
    try:
        if args.dataset_type == 'custom':
            # 自定义数据加载流程
            raw_kg = load_custom_kg_from_json(os.path.join(data_root, "kg_data.json"))
            data, entity_map, relation_map, _ = process_custom_kg(raw_kg)

        elif args.dataset_type == 'standard':
            # 标准数据集加载流程
            train_raw, valid_raw, test_raw = load_standard_dataset(data_root)
            data, entity_map, relation_map, _, _, _ = process_standard_kg(
                train_raw, valid_raw, test_raw
            )
        else:
            raise ValueError("`dataset_type` 必须是 'custom' 或 'standard'")

    except (FileNotFoundError, ValueError) as e:
        print(f"错误: 数据加载失败。 {e}")
        return

    # --- 3. 将数据移动到设备 ---
    # 移除了创建巨大 one-hot 特征矩阵 data.x 的步骤
    data = data.to(device)

    print(
        f"数据准备完成: {data.num_nodes} 个节点, {data.num_edges} 条边, {len(relation_map)} 种关系。"
    )

    # --- 4. 初始化模型和优化器 ---
    print("--- 正在初始化 RGCNEncoder 模型和优化器 ---")
    # 使用新的模型签名，传入 num_nodes 和 embedding_dim
    encoder = RGCNEncoder(
        num_nodes=data.num_nodes,
        embedding_dim=args.embedding_dim,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        num_relations=len(relation_map),
        num_bases=args.num_bases  # OOM 优化 3: 传递 num_bases
    ).to(device)

    if args.num_bases:
        print(f"--- RGCN 'num_bases' 已启用, 数量: {args.num_bases} ---")

    optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # --- 5. 开始训练循环 ---
    print(f"--- 开始链接预测预训练，共 {args.epochs} 个 Epochs ---")
    for epoch in range(1, args.epochs + 1):
        loss = train(
            encoder, data, optimizer, device,
            args.neg_sample_ratio, use_amp, scaler
        )
        if epoch % args.print_every == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    print("--- 预训练完成 ---")

    # --- 6. 保存映射文件 ---
    save_mappings(entity_map, relation_map, entity_map_path, relation_map_path)

    # --- 7. 保存训练好的编码器模型 ---
    torch.save(encoder.state_dict(), model_save_path)
    print(f"训练好的 RGCNEncoder 模型已保存至: {model_save_path}")

    # --- 8. 生成并保存节点嵌入 ---
    print("--- 正在生成并保存最终的节点嵌入 ---")
    encoder.eval()
    with torch.no_grad():
        # 调用更新后的模型 forward 方法
        final_embeddings = encoder(data.edge_index, data.edge_type).cpu()

    torch.save(final_embeddings, embedding_save_path)
    print(f"节点嵌入已保存至: {embedding_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RGCN Encoder 预训练脚本 (V4 - OOM 优化)")

    # 数据集和路径参数
    parser.add_argument('--dataset_type', type=str, default='custom', choices=['custom', 'standard'], help='数据集类型')
    parser.add_argument('--dataset_name', type=str, default='my_custom_kg', help='数据集名称 (将作为子目录名)')
    parser.add_argument('--data_dir', type=str, default='./data', help='存放所有数据集的根目录')
    parser.add_argument('--model_root_dir', type=str, default='./checkpoints', help='存放所有模型检查点的根目录')

    # 文件名参数
    parser.add_argument('--model_filename', type=str, default='rgcn_encoder_pretrained.pt',
                        help='保存的 R-GCN 模型文件名')
    parser.add_argument('--embedding_filename', type=str, default='node_embeddings.pt', help='生成的节点嵌入文件名')
    parser.add_argument('--node_map_filename', type=str, default='node_map.json', help='节点/实体映射文件名')
    parser.add_argument('--relation_map_filename', type=str, default='relation_map.json', help='关系映射文件名')

    # 模型参数
    parser.add_argument('--embedding_dim', type=int, default=128, help='节点嵌入的维度')
    parser.add_argument('--hidden_channels', type=int, default=32, help='R-GCN 隐藏层维度')
    parser.add_argument('--out_channels', type=int, default=16, help='R-GCN 输出层/嵌入维度')
    # OOM 优化参数
    parser.add_argument('--num_bases', type=int, default=None, help='R-GCN 基分解数量，可大幅减少参数')

    # 训练参数
    parser.add_argument('--learning_rate', type=float, default=0.005, help='优化器学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Adam优化器权重衰减')
    parser.add_argument('--epochs', type=int, default=1000, help='训练的总 Epochs 数量')
    # OOM 优化参数
    parser.add_argument('--neg_sample_ratio', type=float, default=1.0, help='负采样比例 (相对于正样本数量)')
    parser.add_argument('--use_amp', action='store_true', help='启用自动混合精度 (AMP) 训练')

    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--use_cuda', action='store_true', help='强制使用 CUDA (如果可用)')
    parser.add_argument('--print_every', type=int, default=50, help='每隔多少个 epoch 打印一次日志')
    parser.add_argument('--force_deterministic', action='store_true', help='强制使用确定性算法以确保可复现性')

    args = parser.parse_args()
    main(args)
