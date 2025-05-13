# ID2211 Project

This repository contains the project for ID2211 Data Mining course.

## Project Overview

This project analyzes the temporal evolution of political communities on Reddit from 2008 to 2019. We use graph-based methods to detect communities of subreddits based on user overlap. The primary goals are to:

*   Identify and track political communities over time.
*   Evaluate different community detection algorithms.
*   Understand how community structures change, particularly in response to major political events.ß

## Methodology Summary

We implemented and evaluated three community detection algorithms:

1.  **Label Propagation:** A semi-supervised method using known subreddit political affiliations as seeds.
2.  **Louvain Community Detection:** An unsupervised algorithm that optimizes for modularity, suitable for weighted networks.
3.  **Spectral Clustering:** An unsupervised algorithm using the eigen-decomposition of the graph Laplacian, typically for unweighted networks.

### Algorithm Selection for Temporal Analysis

After a comprehensive evaluation of these algorithms across multiple metrics (modularity, purity, and Normalized Mutual Information - NMI), the **Louvain algorithm consistently demonstrated superior performance**. It yielded more well-defined communities (higher modularity) and clusters that aligned more closely with known political affiliations (higher purity and NMI) compared to Spectral Clustering and Label Propagation's direct output for clustering.

Therefore, for the subsequent temporal analysis and event correlation phases of this project, we will primarily focus on the communities detected by the Louvain algorithm.

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd kth-id2211-project
```

2. Create and activate a virtual environment (recommended):

```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On Unix/MacOS
python -m venv .venv
source .venv/bin/activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```
