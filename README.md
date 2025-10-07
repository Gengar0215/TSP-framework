<div align="center">

# Think-Search-Patch (TSP): A Retrieval-Augmented Reasoning Framework for Repository-Level Code Repair


---

## Table of Contents

- [Overview](#overview)
- [Framework Architecture](#framework-architecture)
- [Key Features](#key-features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [License](#license)

---

## Overview

Think-Search-Patch (TSP) is a retrieval-augmented reasoning framework designed for repository-level code repair. This framework combines advanced retrieval techniques with reasoning capabilities to provide effective solutions for code repair tasks at the repository level.

## Framework Architecture

### TSP Framework Overview
![TSP Framework](assets/fig1.png)

### Index Construction Pipeline
![Index Construction](assets/fig2.png)

## Key Features

- **Retrieval-Augmented Reasoning**: Combines retrieval mechanisms with reasoning capabilities
- **Repository-Level Understanding**: Handles code repair at the repository scale
- **Information Masking**: Advanced information masking techniques in both training frameworks
- **Comprehensive Evaluation**: Search reward calculation and LLM-judge evaluation methods
- **Modular Architecture**: Well-organized components for easy development and deployment

## Installation

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU (recommended)

### Install Required Frameworks

```bash
# Clone the repository
git clone https://github.com/your-username/TSP.git
cd TSP

# Install SFT training framework
cd verl && pip install -e .
cd ..

# Install DPO training framework
cd OpenRLHF && pip install -e .
cd ..

# Install index construction framework
cd SWE-bench && pip install -e .
cd ..
```

## Project Structure

```
TSP/
├── assets/                   # Framework diagrams and images
├── OpenRLHF/                # DPO training framework
├── verl/                    # SFT training framework
├── SWE-bench/               # Index construction framework
└── srcipts/                 # Core TSP components
    ├── inference/           # TSP inference framework
    ├── make_index/          # Repository indexing tools
    ├── process_data/        # SFT data processing
    ├── retrieval/           # Retrieval service
    └── score/               # Evaluation and scoring
```

## Quick Start

### 1. Build Repository Index

```bash
cd srcipts/make_index

# Download target repositories
python download_repo.py

# Build search indices
python make_index.py
```

### 2. Process Training Data

```bash
cd srcipts/process_data

# Generate SFT training data
python process_sftdata.py
```

### 3. Start Retrieval Service

```bash
cd srcipts/retrieval

# Launch indexing service
python retrieval_multi_server.py
```

### 4. Run Inference

```bash
cd srcipts/inference

# Execute TSP inference
python infer_tsp.py
```

## Usage

### Framework Components

#### **SFT Training with VERL**
```bash
cd verl
pip install -e .
# VERL incorporates information masking techniques for enhanced training
```

#### **DPO Training with OpenRLHF**
```bash
cd OpenRLHF
pip install -e .
# OpenRLHF utilizes information masking strategies for improved outcomes
```

#### **Index Construction with SWE-bench**
```bash
cd SWE-bench
pip install -e .
# SWE-bench provides infrastructure for repository-level code analysis
```

### Component Details

| Component | Path | Main Scripts | Description |
|-----------|------|-------------|-------------|
| **Inference** | `srcipts/inference/` | `python infer_tsp.py` | TSP inference framework for code repair |
| **Index Construction** | `srcipts/make_index/` | `python download_repo.py` `python make_index.py` | Repository download and index building |
| **Data Processing** | `srcipts/process_data/` | `python process_sftdata.py` | SFT training data construction |
| **Retrieval Service** | `srcipts/retrieval/` | `python retrieval_multi_server.py` | Indexing service infrastructure |
| **Evaluation** | `srcipts/score/` | `python cal_search_reward.py` `python llm-judge.py` | Search reward calculation and LLM evaluation |

### Evaluation and Scoring

```bash
cd srcipts/score

# Calculate search reward metrics
python cal_search_reward.py

# Run LLM-judge evaluation
python llm-judge.py
```

## License

This project incorporates multiple frameworks, each with their own licenses:

- **OpenRLHF**: Please refer to `OpenRLHF/LICENSE`
- **VERL**: Please refer to `verl/LICENSE`
- **SWE-bench**: Please refer to `SWE-bench/LICENSE`
