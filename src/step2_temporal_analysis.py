import pandas as pd
import numpy as np
from pathlib import Path
import glob
import json
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

from utils import get_project_root, ensure_dir, load_subreddit_metadata

# --- Constants ---
JACCARD_THRESHOLD = 0.2  # Threshold for considering communities as continuations
LOUVAIN_DIR_NAME = "louvain_clustering"
METADATA_FILE_NAME = "subreddits_metadata.json"
OUTPUT_DIR_NAME = "step2_temporal_analysis"
VISUALIZATIONS_SUBDIR = "visualizations"

# --- Helper Functions ---


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


def jaccard_similarity(set1: set, set2: set) -> float:
    """Calculates Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


def calculate_community_political_composition(
    community_nodes: set[str], subreddit_political_labels: dict[str, str]
) -> dict:
    """
    Calculates the political composition of a community.
    Returns a dictionary with dem_nodes, rep_nodes, other_nodes, dem_ratio, rep_ratio, other_ratio,
    dominant_party, homogeneity_score (ratio of dominant party), and total_nodes.
    """
    party_counts = Counter()
    for node in community_nodes:
        party = subreddit_political_labels.get(node, "unknown")
        party_counts[party] += 1

    total_nodes = len(community_nodes)
    if total_nodes == 0:
        return {
            "dem_nodes": 0,
            "rep_nodes": 0,
            "other_nodes": 0,
            "unknown_nodes": 0,
            "dem_ratio": 0.0,
            "rep_ratio": 0.0,
            "other_ratio": 0.0,
            "unknown_ratio": 0.0,
            "dominant_party": "none",
            "homogeneity_score": 0.0,
            "total_nodes": 0,
        }

    composition = {
        "dem_nodes": party_counts.get("dem", 0),
        "rep_nodes": party_counts.get("rep", 0),
        "other_nodes": sum(
            count
            for party, count in party_counts.items()
            if party not in ["dem", "rep", "unknown"]
        ),
        "unknown_nodes": party_counts.get("unknown", 0),
        "total_nodes": total_nodes,
    }

    composition["dem_ratio"] = composition["dem_nodes"] / total_nodes
    composition["rep_ratio"] = composition["rep_nodes"] / total_nodes
    # 'other' includes all non-dem, non-rep, non-unknown parties from metadata
    composition["other_ratio"] = composition["other_nodes"] / total_nodes
    composition["unknown_ratio"] = composition["unknown_nodes"] / total_nodes

    # Determine dominant party and homogeneity
    party_ratios_for_dominance = {
        "dem": composition["dem_ratio"],
        "rep": composition["rep_ratio"],
        "other": composition["other_ratio"],  # Consider if 'other' can be dominant
    }
    if not any(party_ratios_for_dominance.values()):  # All unknown or empty
        composition["dominant_party"] = "unknown"
        composition["homogeneity_score"] = 0.0  # or 1.0 if all are 'unknown'
    else:
        dominant_party = max(
            party_ratios_for_dominance, key=party_ratios_for_dominance.get
        )
        # Check if max is actually > 0, otherwise pick based on counts or default to "mixed" / "unknown"
        if party_ratios_for_dominance[dominant_party] > 0:
            composition["dominant_party"] = dominant_party
            composition["homogeneity_score"] = party_ratios_for_dominance[
                dominant_party
            ]
        elif composition["unknown_ratio"] == 1.0:  # if all nodes are unknown
            composition["dominant_party"] = "unknown"
            composition["homogeneity_score"] = 1.0  # Homogeneously unknown
        else:  # Truly mixed or small, non-distinct parties
            composition["dominant_party"] = "mixed"
            composition["homogeneity_score"] = max(
                party_ratios_for_dominance.values()
            )  # still take max ratio

    return composition


# --- Main Temporal Analysis Logic ---


class TemporalCommunityTracker:
    def __init__(self, jaccard_threshold: float):
        self.jaccard_threshold = jaccard_threshold
        self.next_lineage_id = 0
        self.tracked_lineages = defaultdict(
            list
        )  # lineage_id -> list of yearly snapshots
        # Snapshot: {'year', 'raw_community_id', 'nodes', 'size', 'political_composition'}
        self.yearly_transitions = []  # Store details of transitions

    def _get_new_lineage_id(self) -> int:
        new_id = self.next_lineage_id
        self.next_lineage_id += 1
        return new_id

    def process_year(
        self,
        year: str,
        current_year_communities: dict[int, set[str]],
        subreddit_political_labels: dict[str, str],
        previous_year_lineages: dict[int, set[str]] | None = None,
    ):
        """
        Processes communities for a single year, tracking from previous year if available.
        previous_year_lineages: {lineage_id: nodes_set}
        Returns a dictionary of {lineage_id: nodes_set} for the current year.
        """
        print(f"  Processing temporal tracking for {year}...")
        current_year_tracked_nodes = (
            {}
        )  # {lineage_id: nodes_set} for this year's active lineages

        # Calculate political composition for all raw communities this year
        current_year_compositions = {
            raw_id: calculate_community_political_composition(
                nodes, subreddit_political_labels
            )
            for raw_id, nodes in current_year_communities.items()
        }

        if previous_year_lineages is None:  # First year
            for raw_id, nodes in current_year_communities.items():
                lineage_id = self._get_new_lineage_id()
                composition_info = current_year_compositions[raw_id]
                self.tracked_lineages[lineage_id].append(
                    {
                        "year": year,
                        "raw_community_id": raw_id,
                        "nodes": nodes,
                        "size": len(nodes),
                        "political_composition": composition_info,
                    }
                )
                current_year_tracked_nodes[lineage_id] = nodes
                self.yearly_transitions.append(
                    {
                        "year_prev": None,
                        "year_curr": year,
                        "prev_lineage_id": None,
                        "curr_raw_id": raw_id,
                        "curr_lineage_id": lineage_id,
                        "jaccard": 1.0,
                        "type": "creation",
                        "size_prev": None,
                        "size_curr": len(nodes),
                    }
                )
        else:
            # Match current communities to previous lineages
            # prev_lineage_id -> best_current_raw_id, jaccard
            prev_to_curr_matches = {}
            for prev_lineage_id, prev_nodes in previous_year_lineages.items():
                best_jaccard = -1.0
                best_curr_raw_id = -1
                for curr_raw_id, curr_nodes in current_year_communities.items():
                    sim = jaccard_similarity(prev_nodes, curr_nodes)
                    if sim > best_jaccard:
                        best_jaccard = sim
                        best_curr_raw_id = curr_raw_id
                if best_curr_raw_id != -1 and best_jaccard >= self.jaccard_threshold:
                    prev_to_curr_matches[prev_lineage_id] = (
                        best_curr_raw_id,
                        best_jaccard,
                    )

            # Match current_raw_ids that were successfully matched by a prev_lineage_id
            matched_current_raw_ids = {
                match[0] for match in prev_to_curr_matches.values()
            }

            # Handle continuations
            for prev_lineage_id, (curr_raw_id, jaccard) in prev_to_curr_matches.items():
                nodes = current_year_communities[curr_raw_id]
                composition_info = current_year_compositions[curr_raw_id]
                self.tracked_lineages[prev_lineage_id].append(
                    {
                        "year": year,
                        "raw_community_id": curr_raw_id,
                        "nodes": nodes,
                        "size": len(nodes),
                        "political_composition": composition_info,
                    }
                )
                current_year_tracked_nodes[prev_lineage_id] = nodes
                prev_size = self.tracked_lineages[prev_lineage_id][-2][
                    "size"
                ]  # size from previous year
                self.yearly_transitions.append(
                    {
                        "year_prev": str(int(year) - 1),
                        "year_curr": year,
                        "prev_lineage_id": prev_lineage_id,
                        "curr_raw_id": curr_raw_id,
                        "curr_lineage_id": prev_lineage_id,
                        "jaccard": jaccard,
                        "type": "continuation",
                        "size_prev": prev_size,
                        "size_curr": len(nodes),
                    }
                )

            # Handle creations (current communities not matched)
            for curr_raw_id, nodes in current_year_communities.items():
                if curr_raw_id not in matched_current_raw_ids:
                    lineage_id = self._get_new_lineage_id()
                    composition_info = current_year_compositions[curr_raw_id]
                    self.tracked_lineages[lineage_id].append(
                        {
                            "year": year,
                            "raw_community_id": curr_raw_id,
                            "nodes": nodes,
                            "size": len(nodes),
                            "political_composition": composition_info,
                        }
                    )
                    current_year_tracked_nodes[lineage_id] = nodes
                    self.yearly_transitions.append(
                        {
                            "year_prev": str(int(year) - 1),
                            "year_curr": year,
                            "prev_lineage_id": None,
                            "curr_raw_id": curr_raw_id,
                            "curr_lineage_id": lineage_id,
                            "jaccard": 0.0,
                            "type": "creation",
                            "size_prev": None,
                            "size_curr": len(nodes),
                        }
                    )

            # Handle dissolutions (previous lineages not continued)
            for prev_lineage_id, prev_nodes in previous_year_lineages.items():
                if (
                    prev_lineage_id not in current_year_tracked_nodes
                ):  # Check if it found a continuation
                    self.yearly_transitions.append(
                        {
                            "year_prev": str(int(year) - 1),
                            "year_curr": year,
                            "prev_lineage_id": prev_lineage_id,
                            "curr_raw_id": None,
                            "curr_lineage_id": None,
                            "jaccard": 0.0,
                            "type": "dissolution",
                            "size_prev": len(prev_nodes),
                            "size_curr": None,
                        }
                    )

        return current_year_tracked_nodes

    def get_lineage_details_df(self) -> pd.DataFrame:
        """Converts tracked_lineages into a flat DataFrame."""
        records = []
        for lineage_id, snapshots in self.tracked_lineages.items():
            for snap in snapshots:
                record = {
                    "lineage_id": lineage_id,
                    "year": snap["year"],
                    "raw_community_id": snap["raw_community_id"],
                    "size": snap["size"],
                    **snap["political_composition"],  # Unpack dem_ratio, rep_ratio etc.
                }
                # Exclude 'nodes' set from CSV for brevity, can be reconstructed if needed
                records.append(record)
        return pd.DataFrame(records)

    def analyze_community_stability(self) -> pd.DataFrame:
        """Analyzes stability of all tracked lineages."""
        stability_metrics = []
        for lineage_id, snapshots in self.tracked_lineages.items():
            if not snapshots:
                continue
            lifespan = len(snapshots)
            avg_size = np.mean([s["size"] for s in snapshots])

            # Persistence: average Jaccard with itself in next year (if continued)
            # This needs transition data. For now, just lifespan and avg_size.
            # Or, a simpler persistence: how many years it lasted.

            # Find first and last year
            min_year = min(s["year"] for s in snapshots)
            max_year = max(s["year"] for s in snapshots)

            stability_metrics.append(
                {
                    "lineage_id": lineage_id,
                    "lifespan_years": lifespan,
                    "start_year": min_year,
                    "end_year": max_year,
                    "average_size": avg_size,
                    "total_nodes_seen": len(
                        set.union(*[s["nodes"] for s in snapshots])
                    ),  # Unique nodes over lifespan
                }
            )
        return pd.DataFrame(stability_metrics)


def calculate_yearly_polarization_summary(
    lineage_details_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculates yearly average polarization metrics."""
    if lineage_details_df.empty:
        return pd.DataFrame()

    # Example: Yearly average homogeneity_score (weighted by community size)
    def weighted_avg_homogeneity(group):
        return np.average(group["homogeneity_score"], weights=group["size"])

    yearly_summary = (
        lineage_details_df.groupby("year")
        .apply(
            lambda g: pd.Series(
                {
                    "num_communities": g["lineage_id"].nunique(),
                    "avg_community_size": g["size"].mean(),
                    "total_nodes_in_communities": g["size"].sum(),
                    "avg_homogeneity_unweighted": g["homogeneity_score"].mean(),
                    "avg_homogeneity_weighted": weighted_avg_homogeneity(g),
                    "avg_dem_ratio_weighted": np.average(
                        g["dem_ratio"], weights=g["size"]
                    ),
                    "avg_rep_ratio_weighted": np.average(
                        g["rep_ratio"], weights=g["size"]
                    ),
                }
            )
        )
        .reset_index()
    )
    return yearly_summary


# --- Plotting Functions ---
def plot_temporal_trends(yearly_metrics_df: pd.DataFrame, output_plot_dir: Path):
    ensure_dir(output_plot_dir)

    # Number of communities and average size
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    ax1.plot(
        yearly_metrics_df["year"],
        yearly_metrics_df["num_communities"],
        color="blue",
        marker="o",
        label="Number of Communities",
    )
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Number of Communities", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.plot(
        yearly_metrics_df["year"],
        yearly_metrics_df["avg_community_size"],
        color="red",
        marker="x",
        label="Average Community Size",
    )
    ax2.set_ylabel("Average Community Size", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    plt.title("Community Evolution: Count and Average Size Over Time")
    # ax1.legend(loc="upper left") # Create combined legend
    # ax2.legend(loc="upper right")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper center")
    plt.grid(True, alpha=0.3)
    plt.savefig(
        output_plot_dir / "community_count_size_trends.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Average Homogeneity (Polarization proxy)
    plt.figure(figsize=(10, 6))
    plt.plot(
        yearly_metrics_df["year"],
        yearly_metrics_df["avg_homogeneity_weighted"],
        marker="o",
        label="Avg. Weighted Homogeneity",
    )
    plt.plot(
        yearly_metrics_df["year"],
        yearly_metrics_df["avg_homogeneity_unweighted"],
        marker="x",
        linestyle="--",
        label="Avg. Unweighted Homogeneity",
    )
    plt.title("Average Community Political Homogeneity Over Time")
    plt.xlabel("Year")
    plt.ylabel("Average Homogeneity Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)  # Homogeneity is a ratio
    plt.savefig(
        output_plot_dir / "avg_homogeneity_trends.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Average Dem/Rep Ratios
    plt.figure(figsize=(10, 6))
    plt.plot(
        yearly_metrics_df["year"],
        yearly_metrics_df["avg_dem_ratio_weighted"],
        marker="o",
        color="blue",
        label="Avg. Weighted Dem Ratio",
    )
    plt.plot(
        yearly_metrics_df["year"],
        yearly_metrics_df["avg_rep_ratio_weighted"],
        marker="o",
        color="red",
        label="Avg. Weighted Rep Ratio",
    )
    plt.title("Average Weighted Political Leaning Ratios in Communities Over Time")
    plt.xlabel("Year")
    plt.ylabel("Average Ratio")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.savefig(
        output_plot_dir / "avg_party_ratio_trends.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_lifespan_distribution(stability_df: pd.DataFrame, output_plot_dir: Path):
    if stability_df.empty or "lifespan_years" not in stability_df.columns:
        return
    plt.figure(figsize=(10, 6))
    sns.histplot(
        stability_df["lifespan_years"],
        kde=False,
        bins=max(1, stability_df["lifespan_years"].max()),
    )
    plt.title("Distribution of Community Lineage Lifespans")
    plt.xlabel("Lifespan (Years)")
    plt.ylabel("Number of Community Lineages")
    plt.grid(True, alpha=0.3)
    plt.savefig(
        output_plot_dir / "community_lifespan_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# --- Main ---
def main():
    project_root = get_project_root()
    base_data_dir = project_root / "data"
    base_processed_dir = base_data_dir / "processed"
    louvain_base_dir = base_processed_dir / LOUVAIN_DIR_NAME

    output_main_dir = base_processed_dir / OUTPUT_DIR_NAME
    ensure_dir(output_main_dir)
    output_plot_dir = output_main_dir / VISUALIZATIONS_SUBDIR
    ensure_dir(output_plot_dir)

    print("Starting Step 2: Temporal Analysis of Louvain Communities")

    # 1. Load subreddit political labels (metadata)
    metadata_file_path = base_data_dir / "metadata" / METADATA_FILE_NAME
    subreddit_political_labels = load_subreddit_metadata(
        metadata_file_path, seed_key="party"
    )
    if not subreddit_political_labels:
        print(
            f"Critical: Could not load subreddit metadata from {metadata_file_path}. Polarization analysis will be limited."
        )
        # return # Or proceed with dummy labels for structural analysis

    # 2. Discover available years for Louvain results
    year_dirs = sorted(
        [d.name for d in louvain_base_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    )
    if not year_dirs:
        print(f"No yearly Louvain community data found in {louvain_base_dir}")
        return
    print(f"Found Louvain data for years: {year_dirs}")

    # 3. Initialize Tracker and process year by year
    tracker = TemporalCommunityTracker(jaccard_threshold=JACCARD_THRESHOLD)

    # Store {lineage_id: nodes_set} for the previously processed year's active lineages
    # This will be updated in each iteration.
    active_lineages_prev_year = None

    for year in year_dirs:
        print(f"Processing year: {year}")
        current_year_raw_communities = load_louvain_communities_for_year(
            base_processed_dir, year
        )
        if not current_year_raw_communities:
            print(
                f"  No Louvain communities loaded for {year}. Skipping temporal processing for this year."
            )
            # If a year is skipped, the chain of active_lineages_prev_year is broken.
            # For simplicity, we reset it, meaning lineages might restart after a gap.
            # A more complex logic could try to bridge gaps.
            active_lineages_prev_year = None
            continue

        active_lineages_prev_year = tracker.process_year(
            year,
            current_year_raw_communities,
            subreddit_political_labels,
            active_lineages_prev_year,
        )
        print(
            f"  Finished processing for {year}. Active lineages: {len(active_lineages_prev_year) if active_lineages_prev_year else 0}"
        )

    # 4. Consolidate and Save Results
    print("Consolidating and saving temporal analysis results...")

    # Lineage details (yearly snapshots of each tracked community)
    lineage_details_df = tracker.get_lineage_details_df()
    if not lineage_details_df.empty:
        lineage_details_df.to_csv(
            output_main_dir / "tracked_community_lineage_details.csv", index=False
        )
        print(
            f"Saved tracked community lineage details to {output_main_dir / 'tracked_community_lineage_details.csv'}"
        )
    else:
        print("No lineage details to save.")

    # Community stability metrics
    stability_df = tracker.analyze_community_stability()
    if not stability_df.empty:
        stability_df.to_csv(
            output_main_dir / "community_stability_metrics.csv", index=False
        )
        print(
            f"Saved community stability metrics to {output_main_dir / 'community_stability_metrics.csv'}"
        )
    else:
        print("No stability metrics to save.")

    # Yearly transitions summary
    transitions_df = pd.DataFrame(tracker.yearly_transitions)
    if not transitions_df.empty:
        transitions_df.to_csv(
            output_main_dir / "yearly_community_transitions.csv", index=False
        )
        print(
            f"Saved yearly community transitions to {output_main_dir / 'yearly_community_transitions.csv'}"
        )
    else:
        print("No transition data to save.")

    # Yearly polarization summary
    yearly_polarization_df = calculate_yearly_polarization_summary(lineage_details_df)
    if not yearly_polarization_df.empty:
        yearly_polarization_df.to_csv(
            output_main_dir / "yearly_polarization_summary.csv", index=False
        )
        print(
            f"Saved yearly polarization summary to {output_main_dir / 'yearly_polarization_summary.csv'}"
        )

        # 5. Generate and Save Plots
        print("Generating temporal trend visualizations...")
        plot_temporal_trends(yearly_polarization_df, output_plot_dir)
        if not stability_df.empty:
            plot_lifespan_distribution(stability_df, output_plot_dir)
        print(f"Saved plots to {output_plot_dir}")
    else:
        print("No yearly polarization summary to save or plot.")

    print("Step 2: Temporal Analysis complete.")
    print(f"All temporal analysis outputs are in: {output_main_dir}")


if __name__ == "__main__":
    main()
