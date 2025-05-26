import networkx as nx
import numpy as np
import random
from sklearn.metrics import roc_auc_score


def evaluate_link_prediction(
        G: nx.Graph,
        weight_attr: str,
        test_frac: float = 0.2,
        n_splits: int = 5,
        random_seed: int = 42
) -> (float, float):
    """
    Evaluate link prediction performance (AUC) for a given edge-weight attribute.

    Parameters:
    - G: A NetworkX graph where edges have the attribute `weight_attr`.
    - weight_attr: Edge attribute name to use as the score.
    - test_frac: Fraction of edges to hold out as positive test examples per split.
    - n_splits: Number of random train/test splits.
    - random_seed: Seed for reproducibility.

    Returns:
    - mean_auc: Mean AUC over splits.
    - std_auc: Standard deviation of AUC over splits.
    """
    random.seed(random_seed)
    nodes = list(G.nodes())
    edges = list(G.edges())
    n_test = int(len(edges) * test_frac)

    aucs = []
    for _ in range(n_splits):
        # Split positive edges
        test_pos = random.sample(edges, n_test)

        # Build training graph
        G_train = G.copy()
        G_train.remove_edges_from(test_pos)

        # Sample negative edges
        test_neg = set()
        while len(test_neg) < n_test:
            u, v = random.sample(nodes, 2)
            if not G_train.has_edge(u, v):
                test_neg.add((u, v))
        test_neg = list(test_neg)

        # Prepare true labels and scores
        y_true = [1] * n_test + [0] * n_test
        y_scores = []

        # Positive scores
        for u, v in test_pos:
            y_scores.append(G[u][v].get(weight_attr, 0))
        # Negative scores (0 if no edge)
        for u, v in test_neg:
            y_scores.append(G[u][v].get(weight_attr, 0) if G.has_edge(u, v) else 0)

        # Compute AUC
        auc = roc_auc_score(y_true, y_scores)
        aucs.append(auc)

    return np.mean(aucs), np.std(aucs)

# Example usage:
# Assume `G` is your NetworkX graph with edges having attributes 'upvote_similarity'
# and 'co_comment_count'.
# G = nx.read_gpickle('subreddit_graph.gpickle')
# mean_new, std_new = evaluate_link_prediction(G, 'upvote_similarity')
# mean_old, std_old = evaluate_link_prediction(G, 'co_comment_count')
# print(f"New method AUC: {mean_new:.4f} ± {std_new:.4f}")
# print(f"Baseline  AUC: {mean_old:.4f} ± {std_old:.4f}")

