import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import itertools
import os
import sys

# Adjust path to import utils
try:
    from utils import get_project_root, load_network, ensure_dir
    from static_analyzis import (
        GRAPH_METHODS,
        MONTHS,
        GRAPH_DIR as STATIC_GRAPH_DIR,
    )  # For loading graphs
except ImportError:
    current_script_path = Path(__file__).resolve()
    src_path = current_script_path.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from utils import get_project_root, load_network, ensure_dir

    # If static_analyzis is also in src, we can import its constants
    try:
        from static_analyzis import GRAPH_METHODS, MONTHS, GRAPH_DIR as STATIC_GRAPH_DIR
    except ImportError as e:
        print(
            f"Could not import from static_analyzis directly: {e}. Define constants locally or ensure correct PYTHONPATH."
        )
        # Define them locally as a fallback if static_analyzis isn't found easily
        MONTHS = [
            "2016-08",
            "2016-09",
            "2016-10",
            "2016-11",
            "2016-12",
            "2017-01",
            "2017-02",
        ]
        GRAPH_METHODS = {"leandro": "leandro_graph_{}.csv", "t10": "graph_{}_t10.csv"}
        PROJECT_ROOT_FALLBACK = get_project_root()
        STATIC_GRAPH_DIR = PROJECT_ROOT_FALLBACK / "data" / "processed" / "graphs"


PROJECT_ROOT = get_project_root()
STATIC_ANALYSIS_CSV_DIR = (
    PROJECT_ROOT / "analysis_outputs" / "static_analysis" / "csv_data"
)
OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs" / "temporal_analysis"
PLOT_DIR = OUTPUT_DIR / "plots"
CSV_DIR = OUTPUT_DIR / "csv_data"


def load_static_metrics_summary() -> pd.DataFrame:
    """Loads the static metrics summary CSV."""
    summary_file = STATIC_ANALYSIS_CSV_DIR / "static_metrics_summary.csv"
    if not summary_file.exists():
        print(f"Error: Static metrics summary file not found at {summary_file}.")
        print("Please run static_analyzis.py first.")
        return pd.DataFrame()
    return pd.read_csv(summary_file)


def calculate_turnover(
    G1_nodes: set, G1_edges: set, G2_nodes: set, G2_edges: set
) -> dict:
    """Calculates node and edge turnover rates."""
    # Node turnover
    common_nodes = len(G1_nodes.intersection(G2_nodes))
    nodes_appearing = len(G2_nodes - G1_nodes)
    nodes_disappearing = len(G1_nodes - G2_nodes)
    total_unique_nodes = len(G1_nodes.union(G2_nodes))

    node_jaccard_index = (
        common_nodes / total_unique_nodes if total_unique_nodes > 0 else 0
    )
    node_turnover_rate = (
        (nodes_appearing + nodes_disappearing) / total_unique_nodes
        if total_unique_nodes > 0
        else 0
    )

    # Edge turnover
    common_edges = len(G1_edges.intersection(G2_edges))
    edges_appearing = len(G2_edges - G1_edges)
    edges_disappearing = len(G1_edges - G2_edges)
    total_unique_edges = len(G1_edges.union(G2_edges))

    edge_jaccard_index = (
        common_edges / total_unique_edges if total_unique_edges > 0 else 0
    )
    edge_turnover_rate = (
        (edges_appearing + edges_disappearing) / total_unique_edges
        if total_unique_edges > 0
        else 0
    )

    return {
        "node_jaccard_index": node_jaccard_index,
        "node_turnover_rate": node_turnover_rate,
        "nodes_appearing": nodes_appearing,
        "nodes_disappearing": nodes_disappearing,
        "edge_jaccard_index": edge_jaccard_index,
        "edge_turnover_rate": edge_turnover_rate,
        "edges_appearing": edges_appearing,
        "edges_disappearing": edges_disappearing,
    }


def calculate_network_volatility(G1_edges: set, G2_edges: set) -> float:
    """
    Calculates network volatility based on Jaccard dissimilarity of edge sets.
    Volatility = 1 - Jaccard Index(edges_t, edges_t+1)
    """
    intersection_size = len(G1_edges.intersection(G2_edges))
    union_size = len(G1_edges.union(G2_edges))
    if union_size == 0:
        return 0.0  # No edges in either graph
    jaccard_index = intersection_size / union_size
    return 1 - jaccard_index


def plot_temporal_metric(
    df_temporal: pd.DataFrame, metric_name: str, title: str, ylabel: str
):
    """Plots a temporal metric over month transitions."""
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=df_temporal, x="month_transition", y=metric_name, hue="method", marker="o"
    )
    plt.title(title)
    plt.xlabel("Month Transition (t to t+1)")
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = PLOT_DIR / f"{metric_name.lower().replace(' ', '_')}.png"
    plt.savefig(plot_path)
    print(f"Saved plot: {plot_path}")
    plt.close()


def main_temporal_analysis():
    """Main function to perform temporal analysis."""
    ensure_dir(OUTPUT_DIR)
    ensure_dir(PLOT_DIR)
    ensure_dir(CSV_DIR)

    # 1. Metric Evolution (already plotted by static_analyzis.py)
    # We can re-plot or just acknowledge they exist.
    # For this script, focus on new temporal metrics.
    print("Static metric evolution plots are generated by static_analyzis.py.")
    df_static_summary = load_static_metrics_summary()
    if df_static_summary.empty:
        return

    temporal_metrics_list = []
    graphs_cache = {}  # Cache loaded graphs to avoid reloading: (method, month) -> G

    # Helper to load graph and cache it
    def get_graph(method: str, month: str):
        if (method, month) in graphs_cache:
            return graphs_cache[(method, month)]

        file_pattern = GRAPH_METHODS[method]
        file_name = file_pattern.format(month)
        graph_file_path = STATIC_GRAPH_DIR / file_name
        if not graph_file_path.exists():
            print(
                f"Graph file not found: {graph_file_path}. Returning empty graph for temporal analysis."
            )
            return nx.Graph()  # Return empty graph for calculations

        G = load_network(
            network_file_path=graph_file_path,
            weighted=True,
            node_1_col="subreddit_a",
            node_2_col="subreddit_b",
            weight_col="weight",
        )
        graphs_cache[(method, month)] = G
        return G

    for method_name in GRAPH_METHODS.keys():
        for i in range(len(MONTHS) - 1):
            month_t1 = MONTHS[i]
            month_t2 = MONTHS[i + 1]
            month_transition_label = f"{month_t1}_to_{month_t2}"
            print(
                f"\nProcessing temporal metrics for {method_name}: {month_transition_label}"
            )

            G1 = get_graph(method_name, month_t1)
            G2 = get_graph(method_name, month_t2)

            if (
                G1 is None or G2 is None
            ):  # Should be handled by get_graph returning empty graph
                print(
                    f"Skipping {month_transition_label} for {method_name} due to graph loading issues."
                )
                # Add NaN entries
                turnover_data = {
                    "node_jaccard_index": np.nan,
                    "node_turnover_rate": np.nan,
                    "nodes_appearing": 0,
                    "nodes_disappearing": 0,
                    "edge_jaccard_index": np.nan,
                    "edge_turnover_rate": np.nan,
                    "edges_appearing": 0,
                    "edges_disappearing": 0,
                }
                volatility = np.nan
            else:
                G1_nodes = set(G1.nodes())
                # Edges for unweighted comparison (presence/absence)
                # Using frozenset for hashability if storing in sets of sets for complex edge types
                G1_edges = set(map(lambda e: tuple(sorted(e)), G1.edges()))

                G2_nodes = set(G2.nodes())
                G2_edges = set(map(lambda e: tuple(sorted(e)), G2.edges()))

                # 2. Node/Edge Turnover
                turnover_data = calculate_turnover(
                    G1_nodes, G1_edges, G2_nodes, G2_edges
                )

                # 3. Volatility
                volatility = calculate_network_volatility(G1_edges, G2_edges)

            # Get modularity for temporal modularity tracking (from static summary)
            modularity_t1_series = df_static_summary[
                (df_static_summary["method"] == method_name)
                & (df_static_summary["month"] == month_t1)
            ]["modularity"]
            modularity_t2_series = df_static_summary[
                (df_static_summary["method"] == method_name)
                & (df_static_summary["month"] == month_t2)
            ]["modularity"]

            modularity_t1 = (
                modularity_t1_series.iloc[0]
                if not modularity_t1_series.empty
                else np.nan
            )
            modularity_t2 = (
                modularity_t2_series.iloc[0]
                if not modularity_t2_series.empty
                else np.nan
            )

            temporal_entry = {
                "method": method_name,
                "month_transition": month_transition_label,
                "month_t1": month_t1,
                "month_t2": month_t2,
                **turnover_data,
                "volatility": volatility,
                "modularity_t1": modularity_t1,
                "modularity_t2": modularity_t2,
                "modularity_change": (
                    modularity_t2 - modularity_t1
                    if pd.notna(modularity_t1) and pd.notna(modularity_t2)
                    else np.nan
                ),
            }
            temporal_metrics_list.append(temporal_entry)

    df_temporal = pd.DataFrame(temporal_metrics_list)
    temporal_csv_path = CSV_DIR / "temporal_metrics.csv"
    df_temporal.to_csv(temporal_csv_path, index=False)
    print(f"\nSaved temporal metrics to {temporal_csv_path}")
    print(df_temporal.head())

    # 4. Plot temporal metrics
    metrics_to_plot_temporal = [
        ("node_turnover_rate", "Node Turnover Rate Evolution", "Node Turnover Rate"),
        ("edge_turnover_rate", "Edge Turnover Rate Evolution", "Edge Turnover Rate"),
        ("node_jaccard_index", "Node Jaccard Index Evolution", "Node Jaccard Index"),
        ("edge_jaccard_index", "Edge Jaccard Index Evolution", "Edge Jaccard Index"),
        ("volatility", "Network Volatility (Edge Jaccard Dissimilarity)", "Volatility"),
        (
            "modularity_change",
            "Change in Modularity Score Month-over-Month",
            "Modularity Change",
        ),
    ]

    for metric_col, title, ylabel in metrics_to_plot_temporal:
        if metric_col in df_temporal.columns:
            plot_temporal_metric(df_temporal, metric_col, title, ylabel)
        else:
            print(
                f"Warning: Metric column '{metric_col}' not found in temporal DataFrame for plotting."
            )

    print("\nTemporal analysis complete. Plots and CSVs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Temporal Analysis Output Directory: {OUTPUT_DIR}")
    main_temporal_analysis()
