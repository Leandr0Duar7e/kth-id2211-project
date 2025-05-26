import networkx as nx
import random
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def train_test_split_graph(G, test_frac=0.2, seed=42):
    random.seed(seed)
    edges = list(G.edges())
    n_test = int(len(edges) * test_frac)
    test_pos = set(random.sample(edges, n_test))

    # build train graph
    G_train = G.copy()
    G_train.remove_edges_from(test_pos)

    # sample negative examples
    nodes = list(G.nodes())
    test_neg = set()
    while len(test_neg) < n_test:
        u, v = random.sample(nodes, 2)
        if not G_train.has_edge(u, v):
            test_neg.add((u, v))
    return G_train, list(test_pos), list(test_neg)


def extract_features(G, edge_list):
    """
    For each (u,v) in edge_list, compute link-prediction heuristics.
    Returns: feature matrix of shape (len(edge_list), n_features)
    """
    # precompute neighbor sets and degrees
    neighbors = {n: set(G[n]) for n in G}
    deg = dict(G.degree())

    features = []
    for u, v in edge_list:
        cn = len(neighbors[u] & neighbors[v])
        # Jaccard
        union_size = len(neighbors[u] | neighbors[v])
        jc = cn / union_size if union_size else 0.0
        # Adamic-Adar
        aa = sum(1.0 / np.log(deg[w]) for w in (neighbors[u] & neighbors[v]) if deg[w] > 1)
        # Preferential Attachment
        pa = deg[u] * deg[v]
        features.append([cn, jc, aa, pa])
    return np.array(features)


def evaluate_link_prediction(G, test_frac=0.2, seed=42):
    # split
    G_train, pos_train, neg_train = train_test_split_graph(G, test_frac, seed)
    # for demonstration we'll use same split for test—but you could do k-fold
    pos_test, neg_test = pos_train, neg_train

    # unsupervised scores on test set
    feats_pos = extract_features(G_train, pos_test)
    feats_neg = extract_features(G_train, neg_test)
    y_true = np.array([1] * len(pos_test) + [0] * len(neg_test))

    # take e.g. Common Neighbors alone
    scores_cn = np.concatenate([feats_pos[:, 0], feats_neg[:, 0]])
    auc_cn = roc_auc_score(y_true, scores_cn)
    print(f"Unsupervised Common Neighbors AUC: {auc_cn:.4f}")

    # --- supervised classifier ---
    # build train data
    X_train = np.vstack([extract_features(G_train, pos_train),
                         extract_features(G_train, neg_train)])
    y_train = np.array([1] * len(pos_train) + [0] * len(neg_train))

    # choose a model:
    model = LogisticRegression(max_iter=1000)
    # model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # predict on test set
    X_test = np.vstack([feats_pos, feats_neg])
    y_scores = model.predict_proba(X_test)[:, 1]
    auc_sup = roc_auc_score(y_true, y_scores)
    print(f"Supervised (LogReg) AUC:      {auc_sup:.4f}")

    return {'cn_auc': auc_cn, 'sup_auc': auc_sup}


if __name__ == "__main__":
    # load your graph—replace with your own loading code
    # e.g. G = nx.read_gpickle('subreddit_graph.gpickle')
    G = nx.karate_club_graph()  # placeholder

    results = evaluate_link_prediction(G)
    print("Done.", results)
