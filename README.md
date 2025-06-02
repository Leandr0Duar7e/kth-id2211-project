# ID2211 Data Mining Project: KarmaNet - A Sentiment-Weighted Graph Approach to Analyzing Political Communities on Reddit

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Leandr0Duar7e/kth-id2211-project)

This project introduces **KarmaNet**, a novel sentiment-weighted network construction method, to improve the temporal analysis of political communities on Reddit. Leveraging the "Reddit Politosphere" dataset (2008-2019), we compare KarmaNet networks, where edge weights are determined by comment vote consistency across subreddits, against a baseline co-commenting model. Our analysis, focusing on the 2016 U.S. election period, reveals that KarmaNet generates networks with improved structural stability, higher density, and greater sensitivity to real-world political events.

## Core Methodology & Findings

Our primary work involved:

1.  **Dataset Processing:** Utilizing the Reddit Politosphere dataset, focusing on comments, user metadata, and subreddit metadata. We specifically processed monthly comment data around the 2016 U.S. election period (August 2016 - February 2017).
2.  **KarmaNet Implementation:** Developed a novel edge weighting mechanism where connections between subreddits are weighted based on users who not only comment in both but also exhibit consistent sentiment signals (positive/negative comment scores) in both communities.
3.  **Comparative Network Analysis:** Constructed monthly networks using both the traditional Politosphere co-commenting approach and our KarmaNet method.
4.  **Topological Evaluation:** Assessed networks based on density, average shortest path, clustering coefficient, and temporal stability (Node and Edge Jaccard Indices). KarmaNet networks consistently showed higher density, cohesion, and stability.
5.  **Link Prediction:**
    *   **Supervised Learning:** Employed Logistic Regression with features like Common Neighbors, Jaccard Index, Preferential Attachment, and Adamic-Adar to predict future links. KarmaNet significantly outperformed the baseline Politosphere networks (e.g., AUC of ~0.90 for KarmaNet vs. ~0.34 for Politosphere on uniform sampling).
    *   **Graph Embeddings:** Utilized Node2Vec to generate embeddings and predict links using cosine similarity. KarmaNet again demonstrated superior performance (AUC ~0.76 vs. ~0.49 for Politosphere).
6.  **Key Finding:** Sentiment-weighted ties (KarmaNet) yield a more informative and predictive representation of Reddit\'s political interactions, better capturing community dynamics and sensitivity to real-world events.

## Initial Explorations (Future Work Foundation)

As part of our broader exploration, we initiated work on:

*   **Advanced Sentiment Analysis:** Began implementing a pipeline using RoBERTa (a transformer model optimized for social media text) for fine-grained sentiment classification of all comments.
*   **Topic Modeling:** Started to use Latent Dirichlet Allocation (LDA) to identify thematic groups within comment data.
*   **GPU-Accelerated Processing:** This exploratory work was conducted on a Google Cloud VM equipped with a V100 GPU to handle the large volume of comments.

Due to the computational intensity and time constraints, this comprehensive sentiment and topic analysis pipeline was set aside for future work, with the core of this project focusing on the KarmaNet methodology and its evaluation. The foundational scripts for this are available in the `src` directory (e.g., `run_sentiment_analysis.py`, `run_topic_modeling.py`).

## Key Scripts

*   `src/networks_eda.ipynb`: Jupyter notebook for initial EDA (less relevant for the final KarmaNet focus).
*   `src/step0_analysis.py`: Script for temporal shortest path analysis (part of initial explorations).
*   `src/visualize_networks_by_metadata.py`: For visualizing networks with metadata (used in initial explorations).
*   **[Main Scripts for KarmaNet are integrated within the broader project structure and methodologies described in the report, often involving data processing pipelines and custom functions not isolated to single runnable scripts for the core graph construction and analysis.]** (Self-correction: The user mentioned specific scripts for sentiment/topic, so I should ensure these are highlighted if they were indeed central to the *exploratory* part mentioned).
*   `src/run_sentiment_analysis.py`, `src/sentiment_analysis.py`: Scripts for the sentiment analysis pipeline using RoBERTa.
*   `src/run_topic_modeling.py`, `src/topic_modeling.py`: Scripts for the LDA topic modeling pipeline.

## Full Report

For a comprehensive understanding of our methodology, detailed findings, and in-depth analysis of KarmaNet, please refer to our full project report:
[**ID2211_Final_Project_Report___Group_1.pdf**](Report/ID2211_Final_Project_Report___Group_1.pdf)

## Replicating the Analysis

To replicate the core KarmaNet analysis and the initial explorations:

1.  **Environment Setup:**
    *   Clone the repository.
    *   Create and activate a Python virtual environment (e.g., using `venv` or `conda`).
    *   Install dependencies: `pip install -r requirements.txt`
2.  **Dataset:**
    *   Obtain the Reddit Politosphere dataset \cite{hofmann2022politosphere}. Due to its size and terms of use, it is not included in this repository. Ensure the data is structured as expected by the processing scripts (primarily the comments, user metadata, and network files). The scripts generally expect data to be in a `data/` directory within the project root, with subdirectories like `data/comments`, `data/networks`, `data/metadata`.
3.  **Running Analyses:**
    *   The core KarmaNet graph construction and link prediction experiments are detailed in the report. Replicating these would involve:
        *   Preprocessing the comments data for the specified monthly periods (Aug 2016 - Feb 2017).
        *   Implementing the KarmaNet weighting logic and the baseline Politosphere weighting.
        *   Running the link prediction tasks (feature extraction, model training, evaluation) as described.
    *   For the exploratory sentiment and topic modeling:
        *   Execute `src/run_topic_modeling.py` for LDA topic modeling on comment data.
        *   Execute `src/run_sentiment_analysis.py` for RoBERTa-based sentiment classification.
        *   These scripts may require significant computational resources (GPU recommended) and correctly configured input paths to the comment data.
4.  **Configuration:**
    *   Paths to data files and output directories are often hardcoded or expected in specific locations within the scripts. You may need to adjust these paths according to your local setup.
    *   Some scripts (like `run_sentiment_analysis.py`) might have command-line arguments for specifying devices (CPU/GPU) or batch sizes.

## Installation

1. Clone the repository:

```bash
git clone <repository-url> # Replace <repository-url> with the actual URL
cd <repository-directory-name> # Replace with your local directory name
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

Explore the project interactively on [DeepWiki](https://deepwiki.com/Leandr0Duar7e/kth-id2211-project).
