"""
数据加载模块

本模块只负责从磁盘读取原始数据，不进行任何处理或转换。
处理和转换逻辑请参阅 `data_utils.py`。
"""

import os
import json
from typing import List, Tuple, Dict, Any


# --- 类型注释 ---
class NodesData(Dict[str, Any]):
    """单个节点的结构定义"""
    pass


class EdgesData(Dict[str, Any]):
    """单条边的结构定义"""
    pass


class KnowledgeGraph(Dict[str, List]):
    """知识图谱JSON文件的整体结构"""
    pass


# --- 加载器 ---

def load_custom_kg_from_json(file_path: str) -> KnowledgeGraph:
    """
    从 JSON 文件加载自定义格式的知识图谱数据。

    Args:
        file_path (str): kg_data.json 文件的路径。

    Returns:
        KnowledgeGraph: 加载后的数据，以字典形式表示。
    """
    print(f"--- 正在从 {file_path} 加载自定义知识图谱 ---")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_triplets_from_file(file_path: str) -> List[Tuple[str, str, str]]:
    """
    从文本文件加载三元组。
    文件格式应为：头实体\t关系\t尾实体\n
    Args:
        file_path (str): 三元组文件的路径。

    Returns:
        List[Tuple[str, str, str]]: 字符串三元组列表。
    """
    triplets = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # Nell-995 的格式是 concept:h\tconcept:r\tconcept:t
                # 我们在这里直接分割，保持原始字符串
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    triplets.append((parts[0], parts[1], parts[2]))
                else:
                    print(f"警告: 跳过格式不正确的行 -> {line.strip()}")
            except ValueError:
                print(f"警告: 跳过格式不正确的行 -> {line.strip()}")
    return triplets


def load_standard_dataset(dataset_path: str) -> Tuple[
    List[Tuple[str, str, str]],
    List[Tuple[str, str, str]],
    List[Tuple[str, str, str]]
]:
    """
    从目录加载一个标准的知识图谱数据集。
    该函数会查找并加载 train.txt, valid.txt, 和 test.txt 文件。
    """
    print(f"--- 正在从 '{dataset_path}' 加载标准数据集文件 ---")
    print(f"--- 当前工作目录: {os.getcwd()} ---")

    absolute_dataset_path = os.path.abspath(dataset_path)
    print(f"--- 解析后的绝对路径: {absolute_dataset_path} ---")

    if not os.path.isdir(absolute_dataset_path):
        print(f"--- 错误: 路径 '{absolute_dataset_path}' 不是一个有效的目录。 ---")
        raise FileNotFoundError(f"数据集目录不存在: {absolute_dataset_path}")

    print(f"--- 目录 '{absolute_dataset_path}' 中的内容: {os.listdir(absolute_dataset_path)} ---")

    splits = ["train", "valid", "test"]
    loaded_triplets = []

    for split_name in splits:
        # 考虑到某些数据集的文件名可能是 entities.txt, relations.txt
        # 但这里严格按照 train/valid/test.txt 的约定
        file_path = os.path.join(dataset_path, f"{split_name}.txt")

        absolute_file_path = os.path.abspath(file_path)
        print(f"--- 正在检查文件: {absolute_file_path} ---")

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"错误: 在 '{dataset_path}' 中未找到文件 '{os.path.basename(file_path)}'。"
                f"请确保数据集已按要求解压，并且文件名正确 (train.txt, valid.txt, test.txt)。"
            )

        triplets = load_triplets_from_file(file_path)
        print(f"  - 已加载 {len(triplets)} 个三元组从 {os.path.basename(file_path)}")
        loaded_triplets.append(triplets)

    return tuple(loaded_triplets)
