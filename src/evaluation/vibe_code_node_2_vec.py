import pandas as pd
import networkx as nx
import glob
import random
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from node2vec import Node2Vec
import os

# ---- CONFIGURATION ----
DATA_GLOB            = "data/processed/graphs/leandro_graph_*.csv"
NEG_SAMPLE_STRATEGIES = ["uniform", "distance_two"]
# node2vec hyperparams
EMB_DIM     = 64
WALK_LENGTH = 30
NUM_WALKS   = 200
P           = 1.0
Q           = 1.0

# ---- LOADING MONTHLY GRAPHS ----
def load_monthly_graphs():
    base_dir  = os.path.dirname(os.path.abspath(__file__))
    graphs_dir = os.path.join(base_dir, "..", "..", "data", "processed", "graphs")
    path_pattern = os.path.join(graphs_dir, "graph_*.csv")

    graphs = []
    for fn in sorted(glob.glob(path_pattern)):
        month = os.path.basename(fn).replace("graph_", "").replace(".csv", "")
        df = pd.read_csv(fn)
        G  = nx.Graph()
        for u, v, w in df.values:
            G.add_edge(u, v, weight=int(w))
        graphs.append((month, G))
    return graphs

# ---- NEGATIVE SAMPLING ----
def sample_negatives_uniform(G, n):
    nodes, negs = list(G.nodes()), set()
    while len(negs) < n:
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v):
            negs.add((u, v))
    return list(negs)

def sample_negatives_distance_two(G, n):
    cand = [
        (u, v) for u in G for v in G
        if u < v and not G.has_edge(u, v)
           and nx.has_path(G, u, v)
           and nx.shortest_path_length(G, u, v) == 2
    ]
    return random.sample(cand, min(n, len(cand)))

# ---- NODE2VEC EMBEDDING & SCORING ----
def compute_node2vec_scores(G, edge_list):
    # Train node2vec on G
    n2v = Node2Vec(
        G,
        dimensions=EMB_DIM,
        walk_length=WALK_LENGTH,
        num_walks=NUM_WALKS,
        p=P,
        q=Q,
        workers=1,          # adjust for your CPU
        weight_key="weight" # respect weighted edges
    )
    model = n2v.fit(window=10, min_count=1)

    # helper for cosine similarity
    def cos_sim(u, v):
        ui = model.wv.get_vector(str(u))
        vi = model.wv.get_vector(str(v))
        denom = np.linalg.norm(ui) * np.linalg.norm(vi)
        return float(ui.dot(vi) / denom) if denom > 0 else 0.0

    # compute a similarity score for each edge pair
    return np.array([cos_sim(u, v) for u, v in edge_list])

# ---- TEMPORAL EVALUATION ----
def temporal_evaluate(monthly_graphs, neg_strategy="uniform"):
    results = []
    random.seed(42)

    for i in range(1, len(monthly_graphs)):
        # build cumulative training graph
        train_G = nx.Graph()
        for _, g in monthly_graphs[:i]:
            train_G = nx.compose(train_G, g)
        test_month, test_G = monthly_graphs[i]

        # positives & negatives
        pos = list(test_G.edges())
        if neg_strategy == "uniform":
            neg = sample_negatives_uniform(train_G, len(pos))
        elif neg_strategy == "distance_two":
            neg = sample_negatives_distance_two(train_G, len(pos))
        else:
            raise ValueError(f"Unknown neg strategy '{neg_strategy}'")

        # true labels
        y_true = np.array([1]*len(pos) + [0]*len(neg))

        # compute node2vec scores
        scores_pos = compute_node2vec_scores(train_G, pos)
        scores_neg = compute_node2vec_scores(train_G, neg)
        scores_all = np.concatenate([scores_pos, scores_neg])

        # metrics
        auc = roc_auc_score(y_true, scores_all)
        ap  = average_precision_score(y_true, scores_all)

        results.append({
            "test_month":    test_month,
            "neg_strategy":  neg_strategy,
            "AUC_node2vec":  auc,
            "AP_node2vec":   ap
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    monthly = load_monthly_graphs()
    all_res  = []
    for strat in NEG_SAMPLE_STRATEGIES:
        res = temporal_evaluate(monthly, neg_strategy=strat)
        all_res.append(res)
    final_df = pd.concat(all_res, ignore_index=True)
    print(
        final_df
        .groupby("neg_strategy")[["AUC_node2vec","AP_node2vec"]]
        .agg(["mean","std"])
    )
