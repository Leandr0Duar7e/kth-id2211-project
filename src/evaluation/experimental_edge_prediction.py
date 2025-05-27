import networkx as nx
import pandas as pd
import glob
import random
import os
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score
)
from sklearn.linear_model import LogisticRegression


def load_monthly_graphs():
    """
    Load CSVs matching 'leandro_graph_YYYY-MM.csv' in the known graphs directory.
    Returns list of (month, Graph) sorted chronologically.
    """
    # Get absolute path to the directory of the current script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path to the graphs folder
    graphs_dir = os.path.join(base_dir, "..", "..", "data", "processed", "graphs")

    # Construct the full pattern to match leandro_graph_*.csv only
    path_pattern = os.path.join(graphs_dir, "graph_*.csv")

    graphs = []
    for fn in sorted(glob.glob(path_pattern)):
        # Extract the date from the filename (after "leandro_graph_")
        filename = os.path.basename(fn)
        month = filename.replace("graph_", "").replace(".csv", "")

        df = pd.read_csv(fn)
        G = nx.Graph()
        for u, v, w in df.values:
            G.add_edge(u, v, weight=int(w))
        graphs.append((month, G))
    return graphs


def expanding_window_splits(monthly_graphs):
    """
    For each month t > first, trains on all months < t and tests on t.
    Yields (G_train, G_test, month_label).
    """
    for idx in range(1, len(monthly_graphs)):
        # accumulate past graphs
        G_train = nx.Graph()
        for _, G_past in monthly_graphs[:idx]:
            G_train = nx.compose(G_train, G_past)
        month, G_test = monthly_graphs[idx]
        yield G_train, G_test, month


# Negative sampling strategies
def sample_uniform_negatives(G, n):
    nodes = list(G.nodes())
    neg = set()
    while len(neg) < n:
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v):
            neg.add((u, v))
    return list(neg)

def sample_distance2_negatives(G, n):
    neg = set()
    for u in G.nodes():
        for w in G.neighbors(u):
            for v in G.neighbors(w):
                if u < v and not G.has_edge(u, v):
                    neg.add((u, v))
    neg = list(neg)
    return random.sample(neg, min(n, len(neg)))

def precompute_hard_pool(G):
    neighbors = {n: set(G[n]) for n in G}
    pool = []
    for u in G.nodes():
        for v in G.nodes():
            if u < v and not G.has_edge(u, v):
                cn = len(neighbors[u] & neighbors[v])
                pool.append(((u, v), cn))
    pool.sort(key=lambda x: x[1], reverse=True)
    return [uv for uv, score in pool]

def sample_hard_negatives(G, n):
    pool = precompute_hard_pool(G)
    return pool[:n]

# Feature extraction
def extract_features(G, edge_list):
    """
    For each (u, v) in edge_list, compute:
      - cn : common neighbors count
      - jc : Jaccard coef
      - aa : Adamic–Adar
      - pa : preferential attachment
    If u or v didn’t appear in G, treat them as isolated (deg=0, no neighbors).
    Returns an array of shape (len(edge_list), 4).
    """
    # build maps only over existing nodes
    neighbors = {n: set(G[n]) for n in G.nodes()}
    deg       = dict(G.degree())

    feats = []
    for u, v in edge_list:
        # if either endpoint is missing, we'll just get empty sets / zero degrees
        nu = neighbors.get(u, set())
        nv = neighbors.get(v, set())

        cn = len(nu & nv)
        union = len(nu | nv)
        jc = cn / union if union else 0.0

        # only sum over w with deg[w] > 1
        aa = sum(
            1.0 / np.log(deg[w])
            for w in (nu & nv)
            if deg.get(w, 0) > 1
        )

        pa = deg.get(u, 0) * deg.get(v, 0)

        feats.append([cn, jc, aa, pa])

    return np.array(feats)


# Metrics
def precision_at_k(y_true, y_scores, k):
    idx = np.argsort(y_scores)[-k:]
    preds = np.zeros_like(y_true)
    preds[idx] = 1
    return precision_score(y_true, preds)

def recall_at_k(y_true, y_scores, k):
    idx = np.argsort(y_scores)[-k:]
    preds = np.zeros_like(y_true)
    preds[idx] = 1
    return recall_score(y_true, preds)

def evaluate_preds(y_true, y_scores, ks=[50,100]):
    res = {
        'AUC': roc_auc_score(y_true, y_scores),
        'AP': average_precision_score(y_true, y_scores)
    }
    for k in ks:
        res[f'P@{k}'] = precision_at_k(y_true, y_scores, k)
        res[f'R@{k}'] = recall_at_k(y_true, y_scores, k)
    return res

# Main pipeline
if __name__ == "__main__":
    random.seed(42)
    monthly_graphs_raw = load_monthly_graphs()
    thresholds = [5, 10, 20]  # for edge-weight threshold sweep
    samplers = {
        'uniform': sample_uniform_negatives,
        'distance2': sample_distance2_negatives,
        'hard': sample_hard_negatives
    }

    records = []
    for T in thresholds:
        # apply threshold to raw graphs
        monthly_graphs = []
        for month, G_raw in monthly_graphs_raw:
            G_th = nx.Graph((u, v, d) for u, v, d in G_raw.edges(data=True) if d['weight'] >= T)
            monthly_graphs.append((month, G_th))

        for G_train, G_test, month in expanding_window_splits(monthly_graphs):
            pos_test = list(G_test.edges())
            for sampler_name, sampler_fn in samplers.items():
                # generate negatives for testing
                neg_test = sampler_fn(G_train, len(pos_test))

                # Unsupervised: Common Neighbors
                Xp = extract_features(G_train, pos_test)
                Xn = extract_features(G_train, neg_test)
                y_true = np.array([1]*len(pos_test) + [0]*len(neg_test))
                scores_cn = np.concatenate([Xp[:,0], Xn[:,0]])
                metrics_cn = evaluate_preds(y_true, scores_cn)

                # Supervised: train on G_train edges
                pos_train = list(G_train.edges())
                neg_train = sample_uniform_negatives(G_train, len(pos_train))
                X_train = np.vstack([extract_features(G_train, pos_train),
                                     extract_features(G_train, neg_train)])
                y_train = np.array([1]*len(pos_train) + [0]*len(neg_train))
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train, y_train)

                X_test = np.vstack([Xp, Xn])
                y_scores = model.predict_proba(X_test)[:,1]
                metrics_sup = evaluate_preds(y_true, y_scores)

                # record
                records.append({
                    'threshold': T,
                    'month': month,
                    'sampler': sampler_name,
                    'method': 'common_neighbors',
                    **metrics_cn
                })
                records.append({
                    'threshold': T,
                    'month': month,
                    'sampler': sampler_name,
                    'method': 'supervised_logreg',
                    **metrics_sup
                })

    df = pd.DataFrame(records)
    df.to_csv("link_prediction_results.csv", index=False)
    print("Done! Results saved to link_prediction_results.csv")
