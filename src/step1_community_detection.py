import pandas as pd
import numpy as np
from pathlib import Path
import glob
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

from utils import (  # Assuming utils.py is in the same directory (src)
    load_network,
    load_subreddit_metadata,
    ensure_dir,
    get_project_root,
    calculate_modularity,
    calculate_purity,
    calculate_nmi,
)
from label_propagation import LabelPropagator
from louvain_clustering import LouvainCommunityDetector
from spectral_clustering import SpectralClusteringDetector

# Constants for output subdirectories
LP_SUBDIR = "label_propagation"
LOUVAIN_SUBDIR = "louvain_clustering"
SPECTRAL_SUBDIR = "spectral_clustering"
STEP1_EVAL_SUBDIR = "step1_evaluation_summary"
VISUAL_SUBDIR = (
    "visualization_comparisons"  # New directory for comparison visualizations
)


def process_year(
    year: str,
    network_file: Path,
    metadata_file: Path,
    base_output_dir: Path,
    lp_seed_key: str = "party",
):
    """
    Processes a single year: loads data, runs algorithms, evaluates, and saves results.
    """
    print(f"\n--- Processing Year: {year} ---")
    ensure_dir(base_output_dir)

    # --- 1. Load ground truth labels (party affiliation from metadata) ---
    print("Loading subreddit metadata for ground truth labels...")
    true_labels_map = load_subreddit_metadata(metadata_file, seed_key=lp_seed_key)
    if not true_labels_map:
        print(
            f"Warning: No ground truth labels loaded for year {year}. External evaluation will be skipped or limited."
        )
        true_labels_series = pd.Series(dtype=str)  # Empty series
    else:
        true_labels_series = pd.Series(true_labels_map, name="true_label")
        print(f"Loaded {len(true_labels_series)} ground truth labels.")

    # --- 2. Label Propagation (uses unweighted network) ---
    print("\nRunning Label Propagation...")
    lp_output_dir = base_output_dir / LP_SUBDIR / year
    ensure_dir(lp_output_dir)

    G_unweighted_lp = load_network(network_file, weighted=False)
    propagator = LabelPropagator(random_state=42)
    propagated_labels_df = pd.DataFrame()  # Default to empty

    if not G_unweighted_lp.nodes():
        print(f"Year {year}: Graph for Label Propagation is empty. Skipping LP.")
    else:
        propagator.fit(G_unweighted_lp, true_labels_map)  # Use true_labels_map as seeds
        propagated_labels_df = propagator.predict()
        if not propagated_labels_df.empty:
            propagator.save_results(lp_output_dir / f"propagated_labels_{year}.csv")
            propagator.visualize_results(
                lp_output_dir / f"network_visualization_{year}.png"
            )
            print(
                f"Label Propagation for {year} complete. {len(propagated_labels_df)} labels predicted/propagated."
            )
        else:
            print(f"Label Propagation for {year} produced no results.")

    # For evaluation, we might use a combination of original seeds and LP predictions.
    # For now, let's consider the original `true_labels_series` as the primary ground truth for Purity/NMI.
    # And `propagated_labels_df` can be used to get a fuller set of labels for nodes, if needed for coverage.

    # --- 3. Louvain Community Detection (uses weighted network) ---
    print("\nRunning Louvain Community Detection...")
    louvain_output_dir = base_output_dir / LOUVAIN_SUBDIR / year
    ensure_dir(louvain_output_dir)

    G_weighted_louvain = load_network(network_file, weighted=True)
    louvain_detector = LouvainCommunityDetector(random_state=42, num_runs=10)
    louvain_partition = None
    louvain_modularity = None
    louvain_communities = None

    if not G_weighted_louvain.nodes():
        print(f"Year {year}: Graph for Louvain is empty. Skipping Louvain.")
    else:
        louvain_detector.detect_communities(G_weighted_louvain)
        louvain_partition = louvain_detector.get_partition()
        louvain_modularity = louvain_detector.get_modularity()
        louvain_communities = louvain_detector.get_community_map()

        if louvain_partition:
            louvain_detector.save_results(
                louvain_output_dir / f"communities_louvain_{year}.csv"
            )
            louvain_detector.visualize_communities(
                louvain_output_dir / f"louvain_visualization_{year}.png"
            )
            print(
                f"Louvain for {year} complete. Modularity: {louvain_modularity if louvain_modularity is not None else 'N/A'}"
            )
        else:
            print(f"Louvain for {year} produced no partition.")

    # --- 4. Spectral Clustering (uses unweighted network) ---
    print("\nRunning Spectral Clustering...")
    spectral_output_dir = base_output_dir / SPECTRAL_SUBDIR / year
    ensure_dir(spectral_output_dir)

    G_unweighted_spectral = load_network(network_file, weighted=False)
    # Using normalized Laplacian as it's often preferred, and eigengap for k
    spectral_detector = SpectralClusteringDetector(
        max_clusters=15, laplacian_type="normalized", random_state=42
    )
    spectral_partition = None
    spectral_modularity = None
    spectral_k = None
    spectral_communities = None

    if not G_unweighted_spectral.nodes():
        print(
            f"Year {year}: Graph for Spectral Clustering is empty. Skipping Spectral."
        )
    else:
        spectral_detector.detect_communities(
            G_unweighted_spectral
        )  # k determined by eigengap
        spectral_partition = spectral_detector.get_partition()
        spectral_modularity = spectral_detector.get_modularity()
        spectral_k = spectral_detector.get_optimal_k()
        spectral_communities = spectral_detector.get_community_map()

        if spectral_partition:
            spectral_detector.save_results(
                spectral_output_dir / f"communities_spectral_k{spectral_k}_{year}.csv"
            )
            spectral_detector.plot_eigengap(
                spectral_output_dir / f"eigengap_spectral_k{spectral_k}_{year}.png"
            )
            spectral_detector.visualize_communities(
                spectral_output_dir / f"visualization_spectral_k{spectral_k}_{year}.png"
            )
            print(
                f"Spectral Clustering for {year} complete. Optimal k={spectral_k}. Modularity: {spectral_modularity if spectral_modularity is not None else 'N/A'}"
            )
        else:
            print(f"Spectral Clustering for {year} produced no partition.")

    # --- 5. Evaluation ---
    # Store evaluation results for this year
    year_eval_results = {
        "year": year,
        "louvain_modularity": louvain_modularity,
        "spectral_modularity_k_auto": spectral_modularity,
        "spectral_optimal_k": spectral_k,
        "louvain_purity": None,
        "louvain_nmi": None,
        "spectral_purity_k_auto": None,
        "spectral_nmi_k_auto": None,
        "nodes_count": len(G_unweighted_lp.nodes()),
        "edges_count": len(G_unweighted_lp.edges()),
        "louvain_community_count": (
            len(louvain_communities) if louvain_communities else 0
        ),
        "spectral_community_count": (
            len(spectral_communities) if spectral_communities else 0
        ),
    }

    # Generate community-party breakdown for Louvain
    louvain_community_party_breakdown = {}
    spectral_community_party_breakdown = {}

    if true_labels_series.empty:
        print(
            f"Year {year}: Skipping Purity and NMI calculations due to missing ground truth labels."
        )
    else:
        # For Louvain
        if louvain_partition:
            louvain_pred_series = pd.Series(
                louvain_partition, name="louvain_cluster"
            ).reindex(G_weighted_louvain.nodes())
            year_eval_results["louvain_purity"] = calculate_purity(
                true_labels_series, louvain_pred_series
            )
            year_eval_results["louvain_nmi"] = calculate_nmi(
                true_labels_series, louvain_pred_series
            )
            print(
                f"  Louvain - Purity: {year_eval_results['louvain_purity']:.4f}, NMI: {year_eval_results['louvain_nmi']:.4f}"
            )

            # Create community-party breakdown for Louvain
            for community_id, nodes in louvain_communities.items():
                community_party_counts = Counter()
                for node in nodes:
                    if node in true_labels_map:
                        community_party_counts[true_labels_map[node]] += 1
                louvain_community_party_breakdown[community_id] = dict(
                    community_party_counts
                )

            # Save community-party breakdown for Louvain
            if louvain_community_party_breakdown:
                louvain_breakdown_df = pd.DataFrame.from_dict(
                    louvain_community_party_breakdown, orient="index"
                ).fillna(0)
                louvain_breakdown_df.to_csv(
                    louvain_output_dir / f"community_party_breakdown_{year}.csv"
                )
                visualize_community_party_breakdown(
                    louvain_community_party_breakdown,
                    louvain_output_dir / f"community_party_breakdown_{year}.png",
                    f"Louvain Community-Party Breakdown ({year})",
                )

        # For Spectral
        if spectral_partition:
            spectral_pred_series = pd.Series(
                spectral_partition, name="spectral_cluster"
            ).reindex(G_unweighted_spectral.nodes())
            year_eval_results["spectral_purity_k_auto"] = calculate_purity(
                true_labels_series, spectral_pred_series
            )
            year_eval_results["spectral_nmi_k_auto"] = calculate_nmi(
                true_labels_series, spectral_pred_series
            )
            print(
                f"  Spectral (k={spectral_k}) - Purity: {year_eval_results['spectral_purity_k_auto']:.4f}, NMI: {year_eval_results['spectral_nmi_k_auto']:.4f}"
            )

            # Create community-party breakdown for Spectral
            for community_id, nodes in spectral_communities.items():
                community_party_counts = Counter()
                for node in nodes:
                    if node in true_labels_map:
                        community_party_counts[true_labels_map[node]] += 1
                spectral_community_party_breakdown[community_id] = dict(
                    community_party_counts
                )

            # Save community-party breakdown for Spectral
            if spectral_community_party_breakdown:
                spectral_breakdown_df = pd.DataFrame.from_dict(
                    spectral_community_party_breakdown, orient="index"
                ).fillna(0)
                spectral_breakdown_df.to_csv(
                    spectral_output_dir / f"community_party_breakdown_{year}.csv"
                )
                visualize_community_party_breakdown(
                    spectral_community_party_breakdown,
                    spectral_output_dir / f"community_party_breakdown_{year}.png",
                    f"Spectral Community-Party Breakdown (k={spectral_k}, {year})",
                )

    # Generate comparative analysis text report
    generate_comparative_report(
        year,
        year_eval_results,
        louvain_community_party_breakdown,
        spectral_community_party_breakdown,
        base_output_dir / STEP1_EVAL_SUBDIR / f"comparison_report_{year}.txt",
    )

    return year_eval_results


def visualize_community_party_breakdown(breakdown_dict, output_path, title):
    """Visualize community-party breakdown as a stacked bar chart."""
    if not breakdown_dict:
        return

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame.from_dict(breakdown_dict, orient="index").fillna(0)

    # Check if there are any parties in the breakdown
    if df.empty or df.shape[1] == 0:
        return

    plt.figure(figsize=(12, 6))
    df.plot(kind="bar", stacked=True, colormap="viridis")
    plt.title(title)
    plt.xlabel("Community ID")
    plt.ylabel("Number of Subreddits")
    plt.legend(title="Party")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Make sure directory exists
    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_comparative_report(
    year, metrics, louvain_breakdown, spectral_breakdown, output_path
):
    """Generate a textual comparative report for the year."""
    ensure_dir(output_path.parent)

    with open(output_path, "w") as f:
        f.write(f"=== COMMUNITY DETECTION COMPARATIVE REPORT ({year}) ===\n\n")

        # Network Statistics
        f.write("--- NETWORK STATISTICS ---\n")
        f.write(f"Number of nodes (subreddits): {metrics['nodes_count']}\n")
        f.write(f"Number of edges (connections): {metrics['edges_count']}\n\n")

        # Algorithm Performance
        f.write("--- ALGORITHM PERFORMANCE ---\n")
        f.write(
            f"Louvain Modularity: {metrics['louvain_modularity'] if metrics['louvain_modularity'] is not None else 'N/A'}\n"
        )
        f.write(
            f"Spectral Modularity: {metrics['spectral_modularity_k_auto'] if metrics['spectral_modularity_k_auto'] is not None else 'N/A'}\n"
        )
        f.write(
            f"Louvain Purity: {metrics['louvain_purity'] if metrics['louvain_purity'] is not None else 'N/A'}\n"
        )
        f.write(
            f"Spectral Purity: {metrics['spectral_purity_k_auto'] if metrics['spectral_purity_k_auto'] is not None else 'N/A'}\n"
        )
        f.write(
            f"Louvain NMI: {metrics['louvain_nmi'] if metrics['louvain_nmi'] is not None else 'N/A'}\n"
        )
        f.write(
            f"Spectral NMI: {metrics['spectral_nmi_k_auto'] if metrics['spectral_nmi_k_auto'] is not None else 'N/A'}\n\n"
        )

        # Community Structure
        f.write("--- COMMUNITY STRUCTURE ---\n")
        f.write(f"Louvain Communities: {metrics['louvain_community_count']}\n")
        f.write(
            f"Spectral Communities: {metrics['spectral_community_count']} (k={metrics['spectral_optimal_k']})\n\n"
        )

        # Community-Party Breakdown
        f.write("--- COMMUNITY-PARTY BREAKDOWN ---\n")

        f.write("Louvain Communities:\n")
        if louvain_breakdown:
            for community_id, party_counts in louvain_breakdown.items():
                f.write(f"  Community {community_id}: {party_counts}\n")
        else:
            f.write("  No community-party breakdown available\n")

        f.write("\nSpectral Communities:\n")
        if spectral_breakdown:
            for community_id, party_counts in spectral_breakdown.items():
                f.write(f"  Community {community_id}: {party_counts}\n")
        else:
            f.write("  No community-party breakdown available\n")

        # Conclusion
        f.write("\n--- CONCLUSION ---\n")
        if (
            metrics["louvain_modularity"] is not None
            and metrics["spectral_modularity_k_auto"] is not None
        ):
            better_modularity = (
                "Louvain"
                if metrics["louvain_modularity"] > metrics["spectral_modularity_k_auto"]
                else "Spectral"
            )
            f.write(f"Better modularity: {better_modularity}\n")

        if (
            metrics["louvain_purity"] is not None
            and metrics["spectral_purity_k_auto"] is not None
        ):
            better_purity = (
                "Louvain"
                if metrics["louvain_purity"] > metrics["spectral_purity_k_auto"]
                else "Spectral"
            )
            f.write(f"Better purity: {better_purity}\n")

        if (
            metrics["louvain_nmi"] is not None
            and metrics["spectral_nmi_k_auto"] is not None
        ):
            better_nmi = (
                "Louvain"
                if metrics["louvain_nmi"] > metrics["spectral_nmi_k_auto"]
                else "Spectral"
            )
            f.write(f"Better NMI: {better_nmi}\n")


def plot_evaluation_trends(evaluation_df, output_dir):
    """Plot trends for evaluation metrics over the years."""
    ensure_dir(output_dir)

    # Modularity Comparison
    plt.figure(figsize=(12, 6))
    plt.plot(
        evaluation_df.index, evaluation_df["louvain_modularity"], "bo-", label="Louvain"
    )
    plt.plot(
        evaluation_df.index,
        evaluation_df["spectral_modularity_k_auto"],
        "ro-",
        label="Spectral",
    )
    plt.title("Modularity (Q) Comparison Across Years")
    plt.xlabel("Year")
    plt.ylabel("Modularity (Q)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "modularity_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Purity Comparison
    plt.figure(figsize=(12, 6))
    plt.plot(
        evaluation_df.index, evaluation_df["louvain_purity"], "bo-", label="Louvain"
    )
    plt.plot(
        evaluation_df.index,
        evaluation_df["spectral_purity_k_auto"],
        "ro-",
        label="Spectral",
    )
    plt.title("Purity Comparison Across Years")
    plt.xlabel("Year")
    plt.ylabel("Purity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "purity_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # NMI Comparison
    plt.figure(figsize=(12, 6))
    plt.plot(evaluation_df.index, evaluation_df["louvain_nmi"], "bo-", label="Louvain")
    plt.plot(
        evaluation_df.index,
        evaluation_df["spectral_nmi_k_auto"],
        "ro-",
        label="Spectral",
    )
    plt.title("Normalized Mutual Information (NMI) Comparison Across Years")
    plt.xlabel("Year")
    plt.ylabel("NMI")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "nmi_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Network Size and Community Count
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot network size
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Number of Nodes/Edges", color="tab:blue")
    ax1.plot(evaluation_df.index, evaluation_df["nodes_count"], "b-", label="Nodes")
    ax1.plot(evaluation_df.index, evaluation_df["edges_count"], "b--", label="Edges")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Create a secondary y-axis for community counts
    ax2 = ax1.twinx()
    ax2.set_ylabel("Number of Communities", color="tab:red")
    ax2.plot(
        evaluation_df.index,
        evaluation_df["louvain_community_count"],
        "r-",
        label="Louvain Communities",
    )
    ax2.plot(
        evaluation_df.index,
        evaluation_df["spectral_community_count"],
        "r--",
        label="Spectral Communities",
    )
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("Network Size and Community Structure Evolution")
    plt.grid(True, alpha=0.3)
    plt.savefig(
        output_dir / "network_community_evolution.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def generate_overall_summary_report(evaluation_df, output_path):
    """Generate a comprehensive summary report of all years' analysis."""
    with open(output_path, "w") as f:
        f.write("=== REDDIT POLITICAL COMMUNITIES ANALYSIS (2008-2019) ===\n\n")

        # Overall Statistics
        f.write("--- OVERALL STATISTICS ---\n")
        f.write(f"Years analyzed: {len(evaluation_df)}\n")
        f.write(f"Average nodes per year: {evaluation_df['nodes_count'].mean():.1f}\n")
        f.write(f"Average edges per year: {evaluation_df['edges_count'].mean():.1f}\n")
        f.write(
            f"Network growth rate: {(evaluation_df['nodes_count'].iloc[-1] / evaluation_df['nodes_count'].iloc[0]):.2f}x increase in nodes\n\n"
        )

        # Algorithm Comparison
        f.write("--- ALGORITHM COMPARISON ---\n")
        f.write(
            f"Average Louvain modularity: {evaluation_df['louvain_modularity'].mean():.4f}\n"
        )
        f.write(
            f"Average Spectral modularity: {evaluation_df['spectral_modularity_k_auto'].mean():.4f}\n"
        )
        f.write(
            f"Average Louvain purity: {evaluation_df['louvain_purity'].mean():.4f}\n"
        )
        f.write(
            f"Average Spectral purity: {evaluation_df['spectral_purity_k_auto'].mean():.4f}\n"
        )
        f.write(f"Average Louvain NMI: {evaluation_df['louvain_nmi'].mean():.4f}\n")
        f.write(
            f"Average Spectral NMI: {evaluation_df['spectral_nmi_k_auto'].mean():.4f}\n\n"
        )

        # Key Findings
        f.write("--- KEY FINDINGS ---\n")
        # Determine the algorithm with better overall performance
        better_modularity = (
            "Louvain"
            if evaluation_df["louvain_modularity"].mean()
            > evaluation_df["spectral_modularity_k_auto"].mean()
            else "Spectral"
        )
        better_purity = (
            "Louvain"
            if evaluation_df["louvain_purity"].mean()
            > evaluation_df["spectral_purity_k_auto"].mean()
            else "Spectral"
        )
        better_nmi = (
            "Louvain"
            if evaluation_df["louvain_nmi"].mean()
            > evaluation_df["spectral_nmi_k_auto"].mean()
            else "Spectral"
        )

        f.write(f"Better modularity algorithm: {better_modularity}\n")
        f.write(f"Better purity algorithm: {better_purity}\n")
        f.write(f"Better NMI algorithm: {better_nmi}\n\n")

        # Year-by-Year Notable Observations
        f.write("--- YEAR-BY-YEAR NOTABLE OBSERVATIONS ---\n")
        for year, row in evaluation_df.iterrows():
            f.write(f"Year {year}:\n")
            f.write(
                f"  Network: {row['nodes_count']} nodes, {row['edges_count']} edges\n"
            )

            # Check for significant changes from previous year
            if year != evaluation_df.index[0]:  # Not the first year
                prev_year = evaluation_df.index[evaluation_df.index.get_loc(year) - 1]
                prev_row = evaluation_df.loc[prev_year]

                node_change = (
                    (row["nodes_count"] - prev_row["nodes_count"])
                    / prev_row["nodes_count"]
                    * 100
                )
                f.write(f"  Node change from {prev_year}: {node_change:.1f}%\n")

                louvain_mod_change = (
                    (row["louvain_modularity"] - prev_row["louvain_modularity"])
                    / prev_row["louvain_modularity"]
                    * 100
                )
                f.write(f"  Louvain modularity change: {louvain_mod_change:.1f}%\n")

            # Note political event correlations (example)
            if year == "2012":
                f.write(
                    "  Political event: US Presidential Election (Obama vs. Romney)\n"
                )
            elif year == "2016":
                f.write(
                    "  Political event: US Presidential Election (Trump vs. Clinton)\n"
                )

            f.write("\n")

        # Conclusion
        f.write("--- CONCLUSION ---\n")
        f.write(
            "Based on the comprehensive analysis of political communities on Reddit from 2008 to 2019:\n\n"
        )

        # Compare Louvain vs Spectral overall
        if (
            evaluation_df["louvain_modularity"].mean()
            > evaluation_df["spectral_modularity_k_auto"].mean()
            and evaluation_df["louvain_purity"].mean()
            > evaluation_df["spectral_purity_k_auto"].mean()
            and evaluation_df["louvain_nmi"].mean()
            > evaluation_df["spectral_nmi_k_auto"].mean()
        ):
            f.write(
                "Louvain algorithm consistently outperforms Spectral Clustering across all metrics.\n"
            )
        elif (
            evaluation_df["spectral_modularity_k_auto"].mean()
            > evaluation_df["louvain_modularity"].mean()
            and evaluation_df["spectral_purity_k_auto"].mean()
            > evaluation_df["louvain_purity"].mean()
            and evaluation_df["spectral_nmi_k_auto"].mean()
            > evaluation_df["louvain_nmi"].mean()
        ):
            f.write(
                "Spectral Clustering consistently outperforms Louvain across all metrics.\n"
            )
        else:
            f.write(
                "The algorithms show mixed performance across different metrics, suggesting that\n"
            )
            f.write(
                "the choice of community detection algorithm should depend on the specific aspect\n"
            )
            f.write("of community structure that is most important for the analysis.\n")

        # Comment on evolution of communities
        if (
            evaluation_df["louvain_modularity"].iloc[-1]
            > evaluation_df["louvain_modularity"].iloc[0]
        ):
            f.write(
                "\nPolitical communities have become more distinct (higher modularity) over time,\n"
            )
            f.write(
                "suggesting increased polarization in the Reddit political landscape.\n"
            )
        else:
            f.write(
                "\nPolitical communities have not shown a clear trend toward increasing distinctness,\n"
            )
            f.write(
                "suggesting the polarization pattern on Reddit is more complex than a simple increase over time.\n"
            )


def main():
    """Main script to run Step 1: Community Detection and Evaluation."""
    project_root = get_project_root()
    base_data_dir = project_root / "data"
    networks_dir = base_data_dir / "networks"
    metadata_file = base_data_dir / "metadata" / "subreddits_metadata.json"
    output_dir = base_data_dir / "processed"  # Main processed data directory

    ensure_dir(output_dir)
    eval_summary_dir = output_dir / STEP1_EVAL_SUBDIR
    ensure_dir(eval_summary_dir)

    # Create visualization directory for comparison plots
    visualization_dir = output_dir / VISUAL_SUBDIR
    ensure_dir(visualization_dir)

    # Find all yearly network files
    # Assuming format like networks_YYYY.csv
    network_files_pattern = str(networks_dir / "networks_*.csv")
    network_files = sorted(glob.glob(network_files_pattern))

    if not network_files:
        print(f"No network files found matching pattern: {network_files_pattern}")
        return

    print(f"Found {len(network_files)} network files to process.")

    all_years_evaluation_results = []
    start_time_total = time.time()

    for net_file_path_str in network_files:
        net_file_path = Path(net_file_path_str)
        try:
            year_str = net_file_path.stem.split("_")[-1]  # Assumes networks_YYYY.csv
            # Validate if year_str is a number, could be more robust
            if not year_str.isdigit() or not (1900 < int(year_str) < 2100):
                print(
                    f"Could not reliably extract year from '{net_file_path.name}'. Expected format 'networks_YYYY.csv'. Skipping."
                )
                continue
        except IndexError:
            print(
                f"Could not extract year from filename: {net_file_path.name}. Skipping."
            )
            continue

        year_start_time = time.time()
        eval_results = process_year(year_str, net_file_path, metadata_file, output_dir)
        all_years_evaluation_results.append(eval_results)
        year_duration = time.time() - year_start_time
        print(
            f"--- Year {year_str} processing completed in {year_duration:.2f} seconds. ---"
        )

    # Save overall evaluation summary
    if all_years_evaluation_results:
        summary_df = pd.DataFrame(all_years_evaluation_results)
        summary_df = summary_df.set_index("year").sort_index()
        summary_file_path = (
            eval_summary_dir / "community_detection_evaluation_summary.csv"
        )
        summary_df.to_csv(summary_file_path)
        print(f"\nOverall evaluation summary saved to: {summary_file_path}")
        print(summary_df)

        # Generate trend plots
        print("\nGenerating evaluation trend visualizations...")
        plot_evaluation_trends(summary_df, visualization_dir)

        # Generate overall summary report
        overall_report_path = eval_summary_dir / "overall_analysis_summary.txt"
        generate_overall_summary_report(summary_df, overall_report_path)
        print(f"Overall analysis summary saved to: {overall_report_path}")

    else:
        print("\nNo evaluation results to summarize.")

    total_duration = time.time() - start_time_total
    print(f"\nTotal script execution time: {total_duration:.2f} seconds.")

    # Print guide to the output files
    print("\n=== RESULTS GUIDE ===")
    print("The analysis results are organized as follows:")
    print(
        f"1. Raw community detection results: {output_dir}/{LP_SUBDIR}/YYYY/, {output_dir}/{LOUVAIN_SUBDIR}/YYYY/, {output_dir}/{SPECTRAL_SUBDIR}/YYYY/"
    )
    print(
        f"2. Evaluation metrics summary: {eval_summary_dir}/community_detection_evaluation_summary.csv"
    )
    print(f"3. Trend visualizations: {visualization_dir}/*.png")
    print(
        f"4. Yearly comparative reports: {eval_summary_dir}/comparison_report_YYYY.txt"
    )
    print(
        f"5. Overall analysis summary: {eval_summary_dir}/overall_analysis_summary.txt"
    )
    print(
        "\nThese results provide a comprehensive view of political community structure and evolution on Reddit."
    )


if __name__ == "__main__":
    main()
