import argparse
import pandas as pd
import networkx as nx
from typing import Literal, Optional

def load_network(
    network_file: str,
    *,
    directed: bool = False,
    view: Literal["weighted", "unweighted", "both"] = "both",
    source_col: str = "node_1",
    target_col: str = "node_2",
    weight_col: str = "weighted",
    unweighted_col: str = "unweighted",
    na_weight: Optional[float] = None,
) -> nx.Graph | nx.DiGraph:
    """
    Load a CSV‐encoded edge list into a NetworkX graph.

    Parameters
    ----------
    network_file : str
        Path to the CSV file.
    directed : bool, default False
        If True returns a DiGraph, otherwise a Graph.
    view : {"weighted", "unweighted", "both"}, default "both"
        * "weighted"   – keep only the numeric weight attribute.
        * "unweighted" – keep only edges whose `unweighted_col == 1`
                         and store no weight attribute at all.
        * "both"       – store both `weight` and `unweighted` attributes
                         on every edge exactly as they appear.
    source_col, target_col, weight_col, unweighted_col : str
        Column names in the CSV.  Change these if your headers differ.
    na_weight : float or None, default None
        If not None, replaces NaNs in the weight column with this value.

    Returns
    -------
    G : nx.Graph or nx.DiGraph
    """
    df = pd.read_csv(network_file)

    # pick graph class once
    G = nx.DiGraph() if directed else nx.Graph()

    if view == "weighted":
        if na_weight is not None:
            df[weight_col] = df[weight_col].fillna(na_weight)

        # `add_weighted_edges_from` is vectorised and fast
        G.add_weighted_edges_from(
            zip(df[source_col], df[target_col], df[weight_col]),
            weight="weight",
        )

    elif view == "unweighted":
        # keep only rows flagged as unweighted
        df_unw = df[df[unweighted_col] == 1]
        G.add_edges_from(zip(df_unw[source_col], df_unw[target_col]))

    elif view == "both":
        # one pass: attach *both* attributes to every edge
        G.add_edges_from(
            (
                u,
                v,
                {
                    "weight": w if pd.notna(w) else na_weight,
                    "unweighted": uw,
                },
            )
            for u, v, w, uw in zip(
                df[source_col],
                df[target_col],
                df[weight_col],
                df[unweighted_col],
            )
        )

    else:
        raise ValueError("view must be 'weighted', 'unweighted', or 'both'")

    return G

