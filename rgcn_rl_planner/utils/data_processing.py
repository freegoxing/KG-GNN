"""
数据处理与转换模块

本模块提供用于将加载后的原始数据转换为模型所需格式的函数。
数据加载功能请参阅 `data_loader.py`。
"""

import json
import os
from typing import List, Dict, Tuple, Set, Union, Optional

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from rgcn_rl_planner.data_loader import KnowledgeGraph

# 动态导入
try:
    import cudf
    import cugraph

    HAS_CUGRAPH = True
except ImportError:
    cudf = None
    cugraph = None
    HAS_CUGRAPH = False

# --- 类型注释 ---
EntityMap = Dict[str, int]
RelationMap = Dict[str, int]
NodeMap = Dict[int, str]  # 这是为自定义KG保留的旧格式
IntTriplets = List[Tuple[int, int, int]]


# --- 通用处理函数 ---

def pyg_to_cugraph(pyg_data: Data, directed: bool = True) -> Optional[cugraph.Graph]:
    """
    将 PyTorch Geometric (PyG) 的 Data 对象转换为 cugraph.Graph 对象。
    此函数封装了从 PyG Tensor 到 cuDF DataFrame 再到 cugraph.Graph 的手动转换逻辑。

    Args:
        pyg_data (Data): 输入的 PyG Data 对象，应位于 CUDA 设备上。
        directed (bool): 是否将图视为有向图。

    Returns:
        Optional[cugraph.Graph]: 转换后的 cugraph.Graph 对象，如果无法转换则返回 None。
    """
    if not HAS_CUGRAPH:
        print("--- cuDF/cuGraph 不可用，无法进行 GPU 图转换。---")
        return None

    if pyg_data.edge_index.device.type != 'cuda':
        print(f"--- 错误: pyg_to_cugraph 需要数据在 CUDA 设备上。当前设备: {pyg_data.edge_index.device.type} ---")
        return None

    try:
        source_nodes = pyg_data.edge_index[0]
        target_nodes = pyg_data.edge_index[1]

        edge_df = cudf.DataFrame({
            'source': source_nodes,
            'destination': target_nodes
        })

        G = cugraph.Graph(directed=directed)

        # 使用边初始化图，节点会自动添加
        G.from_cudf_edgelist(
            edge_df,
            source="source",
            destination="destination",
            store_transposed=True
        )

        return G

    except Exception as e:
        print(f"--- 警告: PyG 到 cuGraph 的转换过程中发生错误: {e} ---")
        return None


def calculate_pagerank(data: Data) -> Dict[int, float]:
    """
    使用 cuGraph (GPU) 计算图中每个节点的 PageRank 值。
    如果 CUDA 不可用或转换/计算失败，则自动退回到 NetworkX (CPU) 实现。
    """
    print("--- 正在计算 PageRank ---")

    if torch.cuda.is_available() and HAS_CUGRAPH:
        data_gpu = data.to("cuda") if data.edge_index.device.type == "cpu" else data

        G = pyg_to_cugraph(data_gpu, directed=True)

        if G is not None:
            try:
                pr = cugraph.pagerank(
                    G,
                    alpha=0.85,
                    max_iter=100,
                    tol=1e-6
                )

                # cuGraph 返回的是 cuDF Series
                pr_dict = {
                    int(row['vertex']): float(row['pagerank'])
                    for index, row in pr.to_pandas().iterrows()
                }

                print("--- cuGraph PageRank 计算成功 ---")
                return pr_dict

            except Exception as e:
                print(f"--- cuGraph PageRank 失败: {e}，回退 NetworkX ---")

    # NetworkX
    print("--- 使用 NetworkX PageRank (CPU) ---")
    G_nx = to_networkx(data.cpu(), to_undirected=False)
    return nx.pagerank(G_nx, alpha=0.85, max_iter=100, tol=1e-6)


def save_mappings(
        entity_map: Union[EntityMap, NodeMap],
        relation_map: RelationMap,
        entity_map_path: str,
        relation_map_path: str
):
    """
    将实体和关系映射保存到 JSON 文件。
    """
    print(f"--- 正在保存映射文件 ---")
    os.makedirs(os.path.dirname(entity_map_path), exist_ok=True)
    os.makedirs(os.path.dirname(relation_map_path), exist_ok=True)

    with open(entity_map_path, 'w', encoding='utf-8') as f:
        json.dump(entity_map, f, ensure_ascii=False, indent=4)

    with open(relation_map_path, 'w', encoding='utf-8') as f:
        json.dump(relation_map, f, ensure_ascii=False, indent=4)

    print(f"实体/节点映射已保存到: {entity_map_path}")
    print(f"关系映射已保存到: {relation_map_path}")


# --- 针对特定数据格式的处理函数 ---

def process_custom_kg(
        kg_data: KnowledgeGraph,
        existing_node_map: Dict[str, str] = None,
        existing_relation_map: RelationMap = None
) -> Tuple[Data, NodeMap, RelationMap, Dict[int, float]]:
    """
    将来自JSON的自定义知识图谱数据转换为 PyTorch Geometric 的 Data 对象。
    (由 `preprocess_data_for_gnn` 重命名而来)
    """
    print("--- 正在处理自定义知识图谱数据 ---")
    nodes = kg_data['nodes']
    edges = kg_data['edges']

    if existing_node_map:
        node_map = {int(k): v for k, v in existing_node_map.items()}
        name_to_id = {v: k for k, v in node_map.items()}
    else:
        node_names = sorted([node['name'] for node in nodes])
        name_to_id = {name: i for i, name in enumerate(node_names)}
        node_map = {i: name for i, name in enumerate(node_names)}

    num_nodes = len(node_map)
    old_id_to_new_id = {node['id']: name_to_id.get(node['name']) for node in nodes}

    if existing_relation_map:
        relation_map = existing_relation_map
    else:
        unique_relations = sorted(list(set(edge['relation'] for edge in edges)))
        relation_map = {rel: i for i, rel in enumerate(unique_relations)}

    edge_index_list = []
    edge_relations = []
    for edge in edges:
        src_old, tgt_old = edge['source'], edge['target']
        src_new = old_id_to_new_id.get(src_old)
        tgt_new = old_id_to_new_id.get(tgt_old)
        if src_new is not None and tgt_new is not None and edge['relation'] in relation_map:
            edge_index_list.append([src_new, tgt_new])
            edge_relations.append(relation_map[edge['relation']])

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_relations, dtype=torch.long)

    # 仅为计算 PageRank 创建临时图
    temp_data = Data(edge_index=edge_index, num_nodes=num_nodes)
    pagerank_values = calculate_pagerank(temp_data)

    # 创建最终的 Data 对象，但不包含 `x` 特征
    # `x` 特征将在训练脚本中被创建（用于预训练）或加载（用于RL）
    data = Data(x=None, edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes)

    print("--- 自定义知识图谱处理完成 ---")
    return data, node_map, relation_map, pagerank_values


def process_standard_kg(
        train_triplets: List[Tuple[str, str, str]],
        valid_triplets: List[Tuple[str, str, str]],
        test_triplets: List[Tuple[str, str, str]]
) -> Tuple[Data, EntityMap, RelationMap, IntTriplets, IntTriplets, IntTriplets]:
    """
    处理从标准数据集加载的原始三元组列表。

    该函数会：
    1. 从所有三元组中构建实体和关系的完整映射。
    2. 将字符串三元组转换为整数 ID 三元组。
    3. 仅使用训练三元组创建 PyTorch Geometric 的 Data 对象。

    Returns:
        - data (Data): PyTorch Geometric 图数据 (仅含训练边)。
        - entity_map (EntityMap): 实体名到 ID 的映射。
        - relation_map (RelationMap): 关系名到 ID 的映射。
        - train_data (IntTriplets): 整数化的训练三元组。
        - valid_data (IntTriplets): 整数化的验证三元组。
        - test_data (IntTriplets): 整数化的测试三元组。
    """
    print("--- 正在处理标准知识图谱数据 ---")
    # 1. 构建实体和关系的完整映射
    all_triplets = train_triplets + valid_triplets + test_triplets
    entities: Set[str] = set()
    relations: Set[str] = set()
    for h, r, t in all_triplets:
        entities.add(h)
        entities.add(t)
        relations.add(r)

    entity_map = {name: i for i, name in enumerate(sorted(list(entities)))}
    relation_map = {name: i for i, name in enumerate(sorted(list(relations)))}
    num_entities = len(entity_map)
    num_relations = len(relation_map)

    print(f"数据集统计: {num_entities} 个实体, {num_relations} 个关系。")

    # 2. 将字符串三元组转换为整数 ID
    def _to_int_triplets(triplets: List[Tuple[str, str, str]]) -> IntTriplets:
        return [(entity_map[h], relation_map[r], entity_map[t]) for h, r, t in triplets]

    train_data = _to_int_triplets(train_triplets)
    valid_data = _to_int_triplets(valid_triplets)
    test_data = _to_int_triplets(test_triplets)

    print(f"已转换 {len(train_data)} 个训练三元组, "
          f"{len(valid_data)} 个验证三元组, "
          f"{len(test_data)} 个测试三元组。")

    # 3. 创建 PyTorch Geometric Data 对象 (仅使用训练集)
    edge_index_list, edge_type_list = [], []
    for h, r, t in train_data:
        edge_index_list.append([h, t])
        edge_type_list.append(r)

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type_list, dtype=torch.long)

    # `x` 特征将在训练或评估脚本中被处理
    data = Data(
        x=None,
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=num_entities
    )

    print("--- 标准知识图谱处理完成 ---")
    return data, entity_map, relation_map, train_data, valid_data, test_data
