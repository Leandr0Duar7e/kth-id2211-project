import pandas as pd
import numpy as np
from pathlib import Path
import glob
import json
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

from utils import get_project_root, ensure_dir, load_network, load_subreddit_metadata

# --- Constants ---
LOUVAIN_DIR_NAME = "louvain_clustering"
STEP2_OUTPUT_DIR_NAME = "step2_temporal_analysis"
STEP3_OUTPUT_DIR_NAME = "step3_advanced_analysis"
VISUALIZATIONS_SUBDIR = "visualizations"
RAW_NETWORKS_DIR_NAME = "networks"


# --- Helper function from Step 2 (or could be moved to utils) ---
def load_louvain_communities_for_year(
    base_processed_dir: Path, year: str
) -> dict[int, set[str]]:
    """Loads Louvain communities for a specific year into a {community_id: {nodes}} format."""
    louvain_communities_path = (
        base_processed_dir / LOUVAIN_DIR_NAME / year / f"communities_louvain_{year}.csv"
    )
    communities = defaultdict(set)
    if not louvain_communities_path.exists():
        print(
            f"Warning: Louvain communities file not found for year {year}: {louvain_communities_path}"
        )
        return communities
    try:
        df = pd.read_csv(louvain_communities_path)
        for _, row in df.iterrows():
            communities[int(row["community_id"])].add(row["subreddit"])
    except Exception as e:
        print(
            f"Error loading Louvain communities for year {year} from {louvain_communities_path}: {e}"
        )
    return communities


# --- Part 1: Deep Dive into Specific Lineages ---
def plot_lineage_political_evolution(
    lineage_details_df: pd.DataFrame, lineage_ids: list[int], output_dir: Path
):
    """Plots political evolution for specified lineages."""
    ensure_dir(output_dir)
    for lineage_id in lineage_ids:
        lineage_data = lineage_details_df[
            lineage_details_df["lineage_id"] == lineage_id
        ].sort_values(by="year")
        if lineage_data.empty:
            print(f"No data found for lineage {lineage_id}. Skipping plot.")
            continue

        plt.figure(figsize=(12, 7))
        plt.plot(
            lineage_data["year"],
            lineage_data["homogeneity_score"],
            marker="o",
            label="Homogeneity Score",
        )
        plt.plot(
            lineage_data["year"],
            lineage_data["dem_ratio"],
            marker="s",
            linestyle="--",
            label="Democrat Ratio",
        )
        plt.plot(
            lineage_data["year"],
            lineage_data["rep_ratio"],
            marker="^",
            linestyle=":",
            label="Republican Ratio",
        )
        plt.plot(
            lineage_data["year"],
            lineage_data["size"] / lineage_data["size"].max(),
            marker="x",
            linestyle="-.",
            label="Normalized Size (Max=1)",
        )

        plt.title(f"Political Evolution and Size of Lineage {lineage_id}")
        plt.xlabel("Year")
        plt.ylabel("Ratio / Score / Normalized Size")
        plt.legend()
        plt.grid(True, alpha=0.4)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / f"lineage_{lineage_id}_evolution.png", dpi=300)
        plt.close()
        print(f"Saved evolution plot for lineage {lineage_id}")


# --- Part 2: Inter-Community Polarization Analysis ---


def get_community_political_profile(
    lineage_details_df: pd.DataFrame, year: str, lineage_id: int
) -> tuple[float, float] | None:
    """Fetches (dem_ratio, rep_ratio) for a lineage in a specific year."""
    profile_data = lineage_details_df[
        (lineage_details_df["year"] == year)
        & (lineage_details_df["lineage_id"] == lineage_id)
    ]
    if not profile_data.empty:
        return profile_data.iloc[0]["dem_ratio"], profile_data.iloc[0]["rep_ratio"]
    return None


def calculate_ideological_distance(
    profile1: tuple[float, float], profile2: tuple[float, float]
) -> float:
    """Calculates Euclidean distance between two (dem_ratio, rep_ratio) profiles."""
    return np.sqrt((profile1[0] - profile2[0]) ** 2 + (profile1[1] - profile2[1]) ** 2)


def calculate_inter_community_linkage(
    graph, community1_nodes: set[str], community2_nodes: set[str]
) -> int:
    """Counts the number of edges between two disjoint sets of nodes in a graph."""
    linkage_count = 0
    # Ensure sets are disjoint for accurate inter-community edge counting
    # Though communities from Louvain should be disjoint by definition.
    # Iterate over nodes of the smaller community for efficiency
    if len(community1_nodes) > len(community2_nodes):
        community1_nodes, community2_nodes = community2_nodes, community1_nodes

    for node1 in community1_nodes:
        if node1 in graph:
            for neighbor in graph.neighbors(node1):
                if neighbor in community2_nodes:
                    linkage_count += (
                        1  # Assumes unweighted graph for inter-community edges
                    )
                    # If weighted, graph[node1][neighbor].get('weight', 1)
    return linkage_count


def analyze_yearly_inter_community_metrics(
    year: str,
    graph,
    active_lineages_nodes: dict[int, set[str]],
    lineage_details_df: pd.DataFrame,
) -> dict | None:
    """
    Calculates inter-community linkage and ideological distance for a given year.
    Returns a dict with yearly summary metrics.
    active_lineages_nodes: {lineage_id: {nodes}} for the current year.
    """
    if len(active_lineages_nodes) < 2:
        return {
            "year": year,
            "num_active_lineages": len(active_lineages_nodes),
            "total_inter_community_links": 0,
            "avg_ideological_distance_linked_communities": None,  # No pairs to measure
            "num_linked_pairs": 0,
        }

    pair_metrics = []
    lineage_ids = list(active_lineages_nodes.keys())

    for i in range(len(lineage_ids)):
        for j in range(i + 1, len(lineage_ids)):
            lineage_id1 = lineage_ids[i]
            lineage_id2 = lineage_ids[j]

            nodes1 = active_lineages_nodes[lineage_id1]
            nodes2 = active_lineages_nodes[lineage_id2]

            linkage = calculate_inter_community_linkage(graph, nodes1, nodes2)

            profile1 = get_community_political_profile(
                lineage_details_df, year, lineage_id1
            )
            profile2 = get_community_political_profile(
                lineage_details_df, year, lineage_id2
            )

            if profile1 and profile2:
                ideological_dist = calculate_ideological_distance(profile1, profile2)
                pair_metrics.append(
                    {
                        "lineage1": lineage_id1,
                        "lineage2": lineage_id2,
                        "linkage": linkage,
                        "ideological_distance": ideological_dist,
                    }
                )
            else:
                print(
                    f"Warning: Missing political profile for lineage pair ({lineage_id1}, {lineage_id2}) in {year}."
                )

    if not pair_metrics:
        return {
            "year": year,
            "num_active_lineages": len(active_lineages_nodes),
            "total_inter_community_links": 0,
            "avg_ideological_distance_linked_communities": None,
            "num_linked_pairs": 0,
        }

    pair_metrics_df = pd.DataFrame(pair_metrics)
    linked_pairs_df = pair_metrics_df[pair_metrics_df["linkage"] > 0]

    total_links = pair_metrics_df[
        "linkage"
    ].sum()  # This is sum of links for all pairs, could double count edges if not careful with definition
    # The way calculate_inter_community_linkage is defined, it counts edges between pairs once.
    # So this sum is total edges between any two distinct communities.

    avg_ideological_distance_linked = None
    if (
        not linked_pairs_df.empty and linked_pairs_df["linkage"].sum() > 0
    ):  # Ensure there are links to weight by
        avg_ideological_distance_linked = np.average(
            linked_pairs_df["ideological_distance"], weights=linked_pairs_df["linkage"]
        )

    return {
        "year": year,
        "num_active_lineages": len(active_lineages_nodes),
        "total_inter_community_links": total_links,
        "avg_ideological_distance_linked_communities": avg_ideological_distance_linked,
        "num_linked_pairs": len(linked_pairs_df),
    }


def plot_inter_community_trends(
    yearly_inter_community_summary_df: pd.DataFrame, output_dir: Path
):
    ensure_dir(output_dir)
    if yearly_inter_community_summary_df.empty:
        print("No inter-community summary data to plot.")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(
        yearly_inter_community_summary_df["year"],
        yearly_inter_community_summary_df[
            "avg_ideological_distance_linked_communities"
        ],
        marker="o",
        label="Avg. Ideological Distance of Linked Communities (Weighted by Linkage)",
    )
    plt.title("Inter-Community Ideological Separation Over Time")
    plt.xlabel("Year")
    plt.ylabel("Average Ideological Distance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "inter_community_ideological_distance_trend.png", dpi=300)
    plt.close()
    print("Saved inter-community ideological distance trend plot.")

    plt.figure(figsize=(12, 6))
    plt.plot(
        yearly_inter_community_summary_df["year"],
        yearly_inter_community_summary_df["total_inter_community_links"],
        marker="s",
        label="Total Inter-Community Links",
    )
    plt.title("Total Inter-Community Links Over Time")
    plt.xlabel("Year")
    plt.ylabel("Number of Links")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "total_inter_community_links_trend.png", dpi=300)
    plt.close()
    print("Saved total inter-community links trend plot.")


# --- Main Function ---
def main():
    project_root = get_project_root()
    base_data_dir = project_root / "data"
    base_processed_dir = base_data_dir / "processed"
    step2_output_dir = base_processed_dir / STEP2_OUTPUT_DIR_NAME
    step3_output_dir = base_processed_dir / STEP3_OUTPUT_DIR_NAME
    ensure_dir(step3_output_dir)
    visualizations_output_dir = step3_output_dir / VISUALIZATIONS_SUBDIR
    ensure_dir(visualizations_output_dir)

    print("Starting Step 3: Advanced Temporal Analysis")

    # --- Load Data ---
    lineage_details_file = step2_output_dir / "tracked_community_lineage_details.csv"
    if not lineage_details_file.exists():
        print(f"Critical: Tracked lineage details not found at {lineage_details_file}")
        return
    lineage_details_df = pd.read_csv(lineage_details_file)
    # Convert year to string for consistent indexing/plotting if it's not already
    lineage_details_df["year"] = lineage_details_df["year"].astype(str)

    # --- Part 1: Plot evolution of specific long-lived lineages ---
    # From step2 community_stability_metrics.csv, lineages 0 and 1 are long-lived.
    # You might want to inspect that file to pick interesting lineages.
    long_lived_lineages_to_plot = [0, 1]
    # Add more based on stability_metrics.csv, e.g. those with high total_nodes_seen or lifespan
    stability_file = step2_output_dir / "community_stability_metrics.csv"
    if stability_file.exists():
        stability_df = pd.read_csv(stability_file)
        # Example: plot top 2 longest and top 2 largest (by total nodes seen), ensuring uniqueness
        top_lifespan = stability_df.nlargest(2, "lifespan_years")["lineage_id"].tolist()
        top_size = stability_df.nlargest(2, "total_nodes_seen")["lineage_id"].tolist()
        interesting_lineages = sorted(
            list(set(long_lived_lineages_to_plot + top_lifespan + top_size))
        )
    else:
        interesting_lineages = long_lived_lineages_to_plot

    print(f"Plotting political evolution for lineages: {interesting_lineages}")
    plot_lineage_political_evolution(
        lineage_details_df, interesting_lineages, visualizations_output_dir
    )

    # --- Part 2: Inter-Community Polarization Analysis ---
    print("\nStarting Inter-Community Polarization Analysis...")
    yearly_inter_community_summaries = []

    # Get sorted list of unique years from lineage_details_df
    available_years = sorted(lineage_details_df["year"].unique().tolist())

    for year in available_years:
        print(f"  Analyzing inter-community metrics for year: {year}")
        raw_network_file = (
            base_data_dir / RAW_NETWORKS_DIR_NAME / f"networks_{year}.csv"
        )
        if not raw_network_file.exists():
            print(
                f"    Warning: Raw network file not found for {year} at {raw_network_file}. Skipping year."
            )
            continue
        # Load unweighted graph for counting inter-community edges, as per project plan
        # Or weighted if you want to consider edge weights in linkage strength
        graph = load_network(raw_network_file, weighted=False)

        # Get active lineages and their node sets for the current year
        lineage_details_this_year = lineage_details_df[
            lineage_details_df["year"] == year
        ]
        if lineage_details_this_year.empty:
            print(f"    No lineage details for year {year}. Skipping.")
            continue

        louvain_partition_this_year = load_louvain_communities_for_year(
            base_processed_dir, year
        )
        if not louvain_partition_this_year:
            print(
                f"    No Louvain partitions loaded for year {year}. Skipping inter-community analysis for this year."
            )
            continue

        active_lineages_nodes_this_year = {}
        for _, row in lineage_details_this_year.iterrows():
            lineage_id = int(row["lineage_id"])
            raw_community_id = int(row["raw_community_id"])
            if raw_community_id in louvain_partition_this_year:
                active_lineages_nodes_this_year[lineage_id] = (
                    louvain_partition_this_year[raw_community_id]
                )
            else:
                # This case should be rare if data is consistent from step 2
                print(
                    f"    Warning: Raw community ID {raw_community_id} for lineage {lineage_id} in year {year} not found in Louvain partitions."
                )

        if not active_lineages_nodes_this_year:
            print(
                f"    No active lineages with node sets found for year {year}. Skipping."
            )
            continue

        yearly_summary = analyze_yearly_inter_community_metrics(
            year, graph, active_lineages_nodes_this_year, lineage_details_df
        )
        if yearly_summary:
            yearly_inter_community_summaries.append(yearly_summary)
            print(
                f"    Finished for {year}. Avg ideological distance linked: {yearly_summary.get('avg_ideological_distance_linked_communities', 'N/A')}"
            )

    if yearly_inter_community_summaries:
        inter_community_summary_df = pd.DataFrame(yearly_inter_community_summaries)
        inter_community_summary_df.to_csv(
            step3_output_dir / "yearly_inter_community_summary.csv", index=False
        )
        print(
            f"\nSaved yearly inter-community summary to {step3_output_dir / 'yearly_inter_community_summary.csv'}"
        )
        plot_inter_community_trends(
            inter_community_summary_df, visualizations_output_dir
        )
    else:
        print("\nNo inter-community summary data generated.")

    print("\nStep 3: Advanced Temporal Analysis complete.")
    print(f"All advanced analysis outputs are in: {step3_output_dir}")


if __name__ == "__main__":
    main()
