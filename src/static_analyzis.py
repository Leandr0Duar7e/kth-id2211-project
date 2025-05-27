import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import community as community_louvain  # For Louvain algorithm

# Adjust path to import utils
import sys

try:
    from utils import get_project_root, load_network, ensure_dir
except ImportError:
    # Fallback if running script directly from src or if PYTHONPATH isn't set
    # Or try to find project root to add to path
    current_script_path = Path(__file__).resolve()
    project_root_path = current_script_path.parent.parent
    src_path = current_script_path.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    if str(project_root_path) not in sys.path:
        sys.path.insert(0, str(project_root_path))
    from utils import get_project_root, load_network, ensure_dir


# Define constants
MONTHS = ["2016-08", "2016-09", "2016-10", "2016-11", "2016-12", "2017-01", "2017-02"]
GRAPH_METHODS = {"leandro": "leandro_graph_{}.csv", "t10": "graph_{}_t10.csv"}
PROJECT_ROOT = get_project_root()
GRAPH_DIR = PROJECT_ROOT / "data" / "processed" / "graphs"
OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs" / "static_analysis"
PLOT_DIR = OUTPUT_DIR / "plots"
CSV_DIR = OUTPUT_DIR / "csv_data"


def calculate_static_metrics(G: nx.Graph, graph_name: str) -> dict:
    """Calculates static metrics for a given graph."""
    metrics = {"graph_name": graph_name}

    if not G.nodes():
        print(f"Graph {graph_name} is empty. Skipping metrics calculation.")
        metrics.update(
            {
                "density": 0,
                "avg_shortest_path": np.nan,
                "diameter": np.nan,
                "avg_clustering_coefficient": 0,
                "modularity": np.nan,
                "num_nodes": 0,
                "num_edges": 0,
                "avg_degree_centrality": np.nan,
                "avg_betweenness_centrality": np.nan,
                "avg_closeness_centrality": np.nan,
                "avg_eigenvector_centrality": np.nan,
                "is_connected": False,
            }
        )
        return metrics

    metrics["num_nodes"] = G.number_of_nodes()
    metrics["num_edges"] = G.number_of_edges()
    metrics["density"] = nx.density(G)
    metrics["is_connected"] = nx.is_connected(G)

    # Metrics for Largest Connected Component (LCC)
    if not metrics["is_connected"] and G.number_of_nodes() > 0:
        largest_cc_nodes = max(nx.connected_components(G), key=len)
        G_cc = G.subgraph(largest_cc_nodes)
        print(
            f"Graph {graph_name} is not connected. Using LCC for shortest path and diameter ({G_cc.number_of_nodes()} nodes, {G_cc.number_of_edges()} edges)."
        )
    else:
        G_cc = G  # If connected or empty (handled above), use G itself

    if G_cc.number_of_nodes() > 1:
        metrics["avg_shortest_path"] = nx.average_shortest_path_length(
            G_cc, weight="weight" if G_cc.is_weighted() else None
        )
        try:
            metrics["diameter"] = nx.diameter(
                G_cc, weight="weight" if G_cc.is_weighted() else None
            )
        except (
            nx.NetworkXError
        ):  # e.g. if G_cc itself became disconnected after subgraph (should not happen with LCC) or other issues
            metrics["diameter"] = np.nan
            print(f"Could not calculate diameter for LCC of {graph_name}")
    else:
        metrics["avg_shortest_path"] = (
            0 if G_cc.number_of_nodes() == 1 else np.nan
        )  # 0 for single node, nan if G_cc is empty
        metrics["diameter"] = 0 if G_cc.number_of_nodes() == 1 else np.nan

    # Clustering coefficient (typically unweighted, but check if weighted makes sense for your context)
    # Using unweighted as per user's example: nx.average_clustering(G)
    metrics["avg_clustering_coefficient"] = nx.average_clustering(G)

    # Modularity (using Louvain)
    if G.number_of_edges() > 0:
        partition = community_louvain.best_partition(G, weight="weight")
        metrics["modularity"] = community_louvain.modularity(
            partition, G, weight="weight"
        )
    else:
        metrics["modularity"] = np.nan

    # Centrality measures (average values)
    # Degree (unweighted is common, G.degree() directly gives degree, not centrality as fraction)
    # nx.degree_centrality is unweighted.
    if G.number_of_nodes() > 0:
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(
            G, weight="weight", normalized=True
        )
        # Closeness centrality needs connected graph for all nodes. Calculate on LCC if not connected.
        # If G is not connected, closeness is typically computed for each component.
        # For an overall average, we can average over LCC or all components. Let's use LCC for now.
        if G_cc.number_of_nodes() > 0:
            closeness_centrality = nx.closeness_centrality(G_cc, distance="weight")
        else:  # G is empty
            closeness_centrality = {}

        try:
            # Eigenvector centrality can fail on graphs with multiple components or specific structures.
            # Use G_cc for eigenvector as well if G is not connected, as it's often more stable.
            if G_cc.number_of_nodes() > 0:
                eigenvector_centrality = nx.eigenvector_centrality(
                    G_cc, weight="weight", max_iter=1000, tol=1e-03
                )  # increased tol
            else:  # G is empty
                eigenvector_centrality = {}
        except (nx.NetworkXError, nx.PowerIterationFailedConvergence) as e:
            print(
                f"Eigenvector centrality failed for {graph_name} (or LCC): {e}. Setting to NaN."
            )
            eigenvector_centrality = {node: np.nan for node in G_cc.nodes()}

        metrics["avg_degree_centrality"] = (
            np.mean(list(degree_centrality.values())) if degree_centrality else np.nan
        )
        metrics["avg_betweenness_centrality"] = (
            np.mean(list(betweenness_centrality.values()))
            if betweenness_centrality
            else np.nan
        )
        metrics["avg_closeness_centrality"] = (
            np.mean(list(closeness_centrality.values()))
            if closeness_centrality
            else np.nan
        )
        metrics["avg_eigenvector_centrality"] = (
            np.nanmean(list(eigenvector_centrality.values()))
            if eigenvector_centrality
            else np.nan
        )  # nanmean for potential nans from failed calc

        # Store all centralities
        centrality_df = pd.DataFrame(
            {
                "node": list(G.nodes()),
                "degree_centrality": [
                    degree_centrality.get(n, np.nan) for n in G.nodes()
                ],
                "betweenness_centrality": [
                    betweenness_centrality.get(n, np.nan) for n in G.nodes()
                ],
                # For closeness and eigenvector, map from G_cc back to G if necessary
                "closeness_centrality": [
                    closeness_centrality.get(n, np.nan) for n in G.nodes()
                ],
                "eigenvector_centrality": [
                    eigenvector_centrality.get(n, np.nan) for n in G.nodes()
                ],
            }
        )
        metrics["centralities_df"] = centrality_df
    else:  # G is empty
        metrics.update(
            {
                "avg_degree_centrality": np.nan,
                "avg_betweenness_centrality": np.nan,
                "avg_closeness_centrality": np.nan,
                "avg_eigenvector_centrality": np.nan,
                "centralities_df": pd.DataFrame(
                    columns=[
                        "node",
                        "degree_centrality",
                        "betweenness_centrality",
                        "closeness_centrality",
                        "eigenvector_centrality",
                    ]
                ),
            }
        )

    # Edge weights
    if G.number_of_edges() > 0:
        metrics["edge_weights"] = [
            d.get("weight", 1) for u, v, d in G.edges(data=True)
        ]  # Default to 1 if no weight
    else:
        metrics["edge_weights"] = []

    return metrics


def plot_metric_comparison(
    df_summary: pd.DataFrame, metric_name: str, title: str, ylabel: str
):
    """Plots a comparison of a metric between methods over months."""
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_summary, x="month", y=metric_name, hue="method", marker="o")
    plt.title(title)
    plt.xlabel("Month")
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = PLOT_DIR / f"{metric_name.lower().replace(' ', '_')}_evolution.png"
    plt.savefig(plot_path)
    print(f"Saved plot: {plot_path}")
    plt.close()


def plot_edge_weight_distribution(all_metrics_data: list, month: str):
    """Plots edge weight distributions for a given month, comparing methods."""
    plt.figure(figsize=(12, 6))
    for method_name in GRAPH_METHODS.keys():
        method_data = next(
            (
                m
                for m in all_metrics_data
                if m["month"] == month and m["method"] == method_name
            ),
            None,
        )
        if method_data and method_data["edge_weights"]:
            sns.histplot(
                method_data["edge_weights"],
                label=f"{method_name} (Median: {np.median(method_data['edge_weights']):.2f})",
                kde=True,
                stat="density",
                common_norm=False,
                element="step",
            )

    plt.title(f"Edge Weight Distribution - {month}")
    plt.xlabel("Edge Weight")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plot_path = PLOT_DIR / f"edge_weight_dist_{month}.png"
    plt.savefig(plot_path)
    print(f"Saved plot: {plot_path}")
    plt.close()


def plot_centrality_distributions(
    all_metrics_data: list, month: str, centrality_type: str
):
    """Plots centrality distributions for a given month, comparing methods."""
    plot_data = []
    for method_name in GRAPH_METHODS.keys():
        method_month_data = next(
            (
                m
                for m in all_metrics_data
                if m["month"] == month and m["method"] == method_name
            ),
            None,
        )
        if method_month_data:
            centralities_df = method_month_data.get("centralities_df")
            if (
                centralities_df is not None
                and not centralities_df.empty
                and centrality_type in centralities_df.columns
            ):
                for val in centralities_df[centrality_type].dropna():
                    plot_data.append({"method": method_name, "centrality_value": val})

    if not plot_data:
        print(f"No data to plot for {centrality_type} distribution for {month}")
        return

    df_plot = pd.DataFrame(plot_data)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_plot, x="method", y="centrality_value")
    plt.title(f"{centrality_type.replace('_', ' ').title()} Distribution - {month}")
    plt.xlabel("Method")
    plt.ylabel(centrality_type.replace("_", " ").title())
    plt.tight_layout()
    plot_path = PLOT_DIR / f"{centrality_type}_dist_{month}.png"
    plt.savefig(plot_path)
    print(f"Saved plot: {plot_path}")
    plt.close()


def main_static_analysis():
    """Main function to perform static analysis."""
    ensure_dir(OUTPUT_DIR)
    ensure_dir(PLOT_DIR)
    ensure_dir(CSV_DIR)

    all_metrics_data = []  # To store dicts from calculate_static_metrics for each graph
    summary_metrics_list = (
        []
    )  # To store simplified metrics for summary CSV and line plots

    for month in MONTHS:
        for method_name, file_pattern in GRAPH_METHODS.items():
            file_name = file_pattern.format(month)
            graph_file_path = GRAPH_DIR / file_name

            print(f"\nProcessing: {method_name} - {month} from {graph_file_path}")

            if not graph_file_path.exists():
                print(f"File not found: {graph_file_path}. Skipping.")
                # Add NaN entries for this graph to keep dataframes aligned if needed
                metrics_summary = {
                    "month": month,
                    "method": method_name,
                    "density": np.nan,
                    "avg_shortest_path": np.nan,
                    "diameter": np.nan,
                    "avg_clustering_coefficient": np.nan,
                    "modularity": np.nan,
                    "num_nodes": 0,
                    "num_edges": 0,
                    "is_connected": False,
                    "avg_degree_centrality": np.nan,
                    "avg_betweenness_centrality": np.nan,
                    "avg_closeness_centrality": np.nan,
                    "avg_eigenvector_centrality": np.nan,
                }
                summary_metrics_list.append(metrics_summary)
                all_metrics_data.append(
                    {
                        "month": month,
                        "method": method_name,
                        "graph_name": f"{method_name}_{month}",
                        "edge_weights": [],
                        "centralities_df": pd.DataFrame(),
                    }
                )  # Add placeholder for plotting functions
                continue

            # Use load_network from utils.py
            G = load_network(
                network_file_path=graph_file_path,
                weighted=True,  # As per user: "edges are nr of shared users but calculated in a different way" -> implies weight
                node_1_col="subreddit_a",  # As per user
                node_2_col="subreddit_b",  # As per user
                weight_col="weight",  # As per user
            )

            if (
                G is None
            ):  # load_network might return None on error, though current utils.py returns empty graph
                print(f"Failed to load graph {graph_file_path}. Skipping.")
                continue

            metrics = calculate_static_metrics(G, f"{method_name}_{month}")

            # Store for overall data and detailed CSVs
            current_data_entry = {"month": month, "method": method_name, **metrics}
            all_metrics_data.append(current_data_entry)

            # Store for summary CSV and line plots
            metrics_summary = {
                "month": month,
                "method": method_name,
                "density": metrics.get("density", np.nan),
                "avg_shortest_path": metrics.get("avg_shortest_path", np.nan),
                "diameter": metrics.get("diameter", np.nan),
                "avg_clustering_coefficient": metrics.get(
                    "avg_clustering_coefficient", np.nan
                ),
                "modularity": metrics.get("modularity", np.nan),
                "num_nodes": metrics.get("num_nodes", 0),
                "num_edges": metrics.get("num_edges", 0),
                "is_connected": metrics.get("is_connected", False),
                "avg_degree_centrality": metrics.get("avg_degree_centrality", np.nan),
                "avg_betweenness_centrality": metrics.get(
                    "avg_betweenness_centrality", np.nan
                ),
                "avg_closeness_centrality": metrics.get(
                    "avg_closeness_centrality", np.nan
                ),
                "avg_eigenvector_centrality": metrics.get(
                    "avg_eigenvector_centrality", np.nan
                ),
            }
            summary_metrics_list.append(metrics_summary)

            # Save centralities per graph
            centralities_df = metrics.get("centralities_df")
            if centralities_df is not None and not centralities_df.empty:
                centrality_csv_path = (
                    CSV_DIR / f"centralities_{method_name}_{month}.csv"
                )
                centralities_df.to_csv(centrality_csv_path, index=False)
                print(f"Saved centralities to {centrality_csv_path}")

    # Create and save summary DataFrame
    df_summary = pd.DataFrame(summary_metrics_list)
    summary_csv_path = CSV_DIR / "static_metrics_summary.csv"
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"\nSaved summary static metrics to {summary_csv_path}")
    print(df_summary.head())

    # Generate plots
    # 1. Plot evolution of key metrics over time
    metrics_to_plot = [
        ("density", "Network Density Evolution", "Density"),
        (
            "avg_shortest_path",
            "Average Shortest Path Evolution (LCC)",
            "Avg. Shortest Path",
        ),
        ("diameter", "Network Diameter Evolution (LCC)", "Diameter"),
        (
            "avg_clustering_coefficient",
            "Avg. Clustering Coefficient Evolution",
            "Avg. Clustering Coeff.",
        ),
        ("modularity", "Modularity Evolution", "Modularity Score"),
        ("num_nodes", "Number of Nodes Evolution", "Node Count"),
        ("num_edges", "Number of Edges Evolution", "Edge Count"),
        (
            "avg_degree_centrality",
            "Avg. Degree Centrality Evolution",
            "Avg. Degree Centrality",
        ),
        (
            "avg_betweenness_centrality",
            "Avg. Betweenness Centrality Evolution",
            "Avg. Betweenness Centrality",
        ),
        (
            "avg_closeness_centrality",
            "Avg. Closeness Centrality Evolution (LCC)",
            "Avg. Closeness Centrality",
        ),
        (
            "avg_eigenvector_centrality",
            "Avg. Eigenvector Centrality Evolution (LCC)",
            "Avg. Eigenvector Centrality",
        ),
    ]
    for metric_col, title, ylabel in metrics_to_plot:
        if metric_col in df_summary.columns:
            plot_metric_comparison(df_summary, metric_col, title, ylabel)
        else:
            print(
                f"Warning: Metric column '{metric_col}' not found in summary DataFrame for plotting."
            )

    # 2. Plot distributions for each month
    for month in MONTHS:
        plot_edge_weight_distribution(all_metrics_data, month)
        for centrality_col_name in [
            "degree_centrality",
            "betweenness_centrality",
            "closeness_centrality",
            "eigenvector_centrality",
        ]:
            plot_centrality_distributions(all_metrics_data, month, centrality_col_name)

    print("\nStatic analysis complete. Plots and CSVs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    # Simple test for get_project_root if utils is loaded correctly
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Graph Directory: {GRAPH_DIR}")
    print(f"Output Directory for Static Analysis: {OUTPUT_DIR}")

    main_static_analysis()
