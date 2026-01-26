# KG-GNN Learning Path Planning

This repository implements a **knowledge graph–based learning path planning framework**
using a **Relational Graph Convolutional Network (RGCN) encoder** combined with
**reinforcement learning (RL)**.

## Overview
- Encode entities and relations in a knowledge graph using an RGCN-based GNN encoder
- Formulate learning path planning as a sequential decision-making problem
- Apply reinforcement learning to guide path selection and optimization

## Method
- **Encoder**: RGCN for relational structure representation
- **Policy**: Reinforcement learning–based path planning
- **Task**: Learning path inference on knowledge graphs

## Datasets
Experiments are conducted on standard benchmark datasets:
- FB15k-237
- WN18RR
- NELL-995

## Environment
- Python 3.x
- PyTorch
- PyTorch Geometric (or DGL, if applicable)

## Usage
1. Prepare the dataset
2. Train the RGCN encoder
3. Run reinforcement learning for path planning
4. Evaluate model performance

## Notes
This project is intended for **research and experimental purposes**.