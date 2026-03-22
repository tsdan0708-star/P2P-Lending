# Enhancing Default Prediction in P2P Lending with Deep Model Fusion: Exploiting Sequential Structures in Transactional Data

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

## 📌 Overview
This repository contains the official implementation of an advanced Expert System for Peer-to-Peer (P2P) lending default prediction. The research addresses the limitations of traditional credit scoring by fusing **static borrower profiles** with **temporal transaction sequences**.

The system utilizes state-of-the-art deep learning architectures, including **LSTM-Attention** and **Transformer-based Fusion**, to capture long-range dependencies in repayment behavior, leading to more robust and explainable risk assessments.

## 🚀 Key Research Contributions
- **Multi-Modal Data Fusion:** Seamlessly integrates tabular borrower data (MLP stream) and sequential transaction data (RNN/Transformer stream).
- **Temporal Attention Mechanism:** Provides interpretability by identifying critical time steps in a loan's lifecycle that contribute most to default risk.
- **Superior Predictive Power:** Outperforms traditional machine learning baselines (LR, RF, XGBoost) in Precision, Recall, and AUC-ROC.
- **Real-world Validation:** Empirically tested on the large-scale **Bondora P2P dataset** (134,529 records).

## 🏗 System Architecture
The framework consists of two parallel processing streams:
1. **Static Stream:** Encodes demographic and financial attributes (e.g., Income, Credit Score, Education).
2. **Sequential Stream:** Processes 6-month historical sequences of Debt-to-Income ratios and repayment statuses using LSTM or Transformer blocks.
3. **Adaptive Fusion:** Merges the feature vectors through a weighted concatenation layer for final probability estimation.

## 📂 Project Structure
```text
├── src/
│   ├── models.py          # Architecture definitions (Fusion, LSTM, Transformer)
│   ├── data_pipeline.py   # Data preprocessing, scaling, and sequence generation
│   ├── train.py           # Training loops, validation, and testing logic
│   └── config.py          # Centralized hyperparameter & path management
├── notebook/
│   ├── eda/               # Exploratory Data Analysis & Feature Engineering
│   └── model/             # Model tuning, heatmaps, and performance evaluation
├── requirements.txt       # List of required Python packages
├── .gitignore             # Configuration to ignore large CSV files and cache
└── README.md              # Project documentation