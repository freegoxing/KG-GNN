import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional


def plot_cumulative_rewards(rewards: List[float], title: str = "Cumulative Reward per Episode", xlabel: str = "Episode",
                            ylabel: str = "Cumulative Reward", save_path: Optional[str] = None):
    """
    绘制每个 Episode 的累积奖励。
    Plots the cumulative reward for each episode.

    Args:
        rewards: 每个 Episode 的累积奖励列表。
                 A list of cumulative rewards for each episode.
        title: 图表标题。
               The title of the plot.
        xlabel: X轴标签。
                The label for the X-axis.
        ylabel: Y轴标签。
                The label for the Y-axis.
        save_path: 可选，保存图表的路径。如果不提供，则显示图表。
                   Optional, path to save the plot. If not provided, the plot is displayed.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_metric_over_time(metric_values: List[float], metric_name: str, x_axis_values: List,
                          title: Optional[str] = None, xlabel: str = "Training Episodes",
                          save_path: Optional[str] = None):
    """
    绘制某个指标随训练（或评估）时间的变化。
    Plots a metric over training (or evaluation) time.

    Args:
        metric_values: 指标在每个评估点的值列表。
                       A list of metric values at each evaluation point.
        metric_name: 指标的名称 (例如, "Success Rate")。
                     The name of the metric (e.g., "Success Rate").
        x_axis_values: 用于 x 轴的实际数值列表 (例如, episode 编号)。
                       A list of actual values for the x-axis (e.g., episode numbers).
        title: 图表标题。如果为 None，则自动生成。
               The title of the plot. If None, it's auto-generated.
        xlabel: X轴标签。
                The label for the X-axis.
        save_path: 可选，保存图表的路径。
                   Optional, path to save the plot.
    """
    if title is None:
        title = f"{metric_name} Over {xlabel}"

    plt.figure(figsize=(10, 5))
    plt.plot(x_axis_values, metric_values, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()  # 释放内存
    else:
        plt.show()


def plot_metrics_summary(metrics: dict, title: str = "Evaluation Metrics Summary", save_path: Optional[str] = None):
    """
    将多个评估指标的总结结果绘制成条形图。
    Plots a summary of multiple evaluation metrics as a bar chart.

    Args:
        metrics: 一个包含指标名称和其值的字典。
                 A dictionary containing metric names and their values.
        title: 图表标题。
               The title of the plot.
        save_path: 可选，保存图表的路径。
                   Optional, path to save the plot.
    """
    # 支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    names = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(10, 6))
    bars = plt.barh(names, values)

    plt.xlabel("值 (Value)")
    plt.title(title)

    # 在条形图上显示数值
    for bar in bars:
        plt.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f'{bar.get_width():.2f}',
            va='center',
            ha='left'
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
