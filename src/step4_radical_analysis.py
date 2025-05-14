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
STEP4_OUTPUT_DIR_NAME = (
    "step4_radical_analysis"  # New directory for this step's outputs
)
VISUALIZATIONS_SUBDIR = "visualizations"
RAW_NETWORKS_DIR_NAME = "networks"  # If needed for direct connectivity


# --- Helper function (potentially from utils or step3) ---
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


def load_all_data(project_root: Path, base_processed_dir: Path) -> dict:
    """Loads all necessary data for the radical analysis."""
    data = {}

    # Subreddit Metadata (for banned status and political party)
    metadata_file = project_root / "data" / "metadata" / "subreddits_metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Subreddit metadata file not found: {metadata_file}")

    # Load metadata: We need a more flexible way to load this if it's truly JSON and not JSONL
    # For now, assuming load_subreddit_metadata can handle it or we adapt it.
    # Let's assume for now it returns a dict {subreddit: {details}}
    # Or a list of dicts that we process into a DataFrame.
    # For banned_status, let's assume the key is 'banned_status' and True means banned.
    # For party, key is 'political_party' or similar.

    raw_metadata = []
    with open(metadata_file, "r") as f:
        for line in f:
            try:
                raw_metadata.append(json.loads(line))
            except (
                json.JSONDecodeError
            ):  # Handle if it's a single JSON object not JSONL
                f.seek(0)
                raw_metadata = json.load(f)
                if isinstance(raw_metadata, dict):  # If it's a dict of subreddits
                    raw_metadata = [
                        {**{"subreddit": k}, **v} for k, v in raw_metadata.items()
                    ]
                break

    metadata_df = pd.DataFrame(raw_metadata)
    # Ensure 'subreddit' column exists
    if "name" in metadata_df.columns and "subreddit" not in metadata_df.columns:
        metadata_df.rename(columns={"name": "subreddit"}, inplace=True)
    if "subreddit" not in metadata_df.columns:
        raise ValueError("Could not find 'subreddit' or 'name' column in metadata.")

    # Standardize banned_status based on the provided sample
    # Key is 'banned', value is 1 for banned, 0 for not banned.
    if "banned" in metadata_df.columns:
        metadata_df["is_banned"] = metadata_df["banned"].astype(int) == 1
        # Clean up old logic branches if they are no longer relevant or to avoid confusion
        # If 'banned_status' was a fallback, we might not need it if 'banned' is reliably present.
        if "banned_status" in metadata_df.columns and "banned" != "banned_status":
            print(
                "Prioritizing 'banned' key for banned status over potential 'banned_status' key."
            )
            # Optionally, remove the now redundant 'banned_status' if it causes issues or is just noise
            # metadata_df.drop(columns=['banned_status'], inplace=True, errors='ignore')
    elif (
        "banned_status" in metadata_df.columns
    ):  # Fallback if 'banned' is not present but 'banned_status' is
        print(
            "Warning: 'banned' key not found. Attempting to use 'banned_status' as a fallback."
        )
        # This fallback logic attempts to interpret various ways 'banned_status' might be represented
        if metadata_df["banned_status"].dtype == "bool":
            metadata_df["is_banned"] = metadata_df["banned_status"]
        else:  # Try to convert from int (e.g., 1/0) or common strings ('true', 'banned')
            try:
                metadata_df["is_banned"] = metadata_df["banned_status"].astype(int) == 1
            except ValueError:
                metadata_df["is_banned"] = (
                    metadata_df["banned_status"]
                    .astype(str)
                    .str.lower()
                    .isin(["true", "1", "yes", "banned"])
                )
    else:
        print(
            "Warning: Critical - Neither 'banned' nor 'banned_status' key found in metadata. Cannot identify banned subreddits."
        )
        metadata_df["is_banned"] = (
            False  # Default to not banned, script functionality will be limited
        )

    data["metadata_df"] = metadata_df.set_index("subreddit")

    # Tracked Community Lineage Details (from Step 2)
    step2_dir = base_processed_dir / STEP2_OUTPUT_DIR_NAME
    lineage_details_file = step2_dir / "tracked_community_lineage_details.csv"
    if not lineage_details_file.exists():
        raise FileNotFoundError(
            f"Tracked lineage details not found: {lineage_details_file}"
        )
    data["lineage_details_df"] = pd.read_csv(lineage_details_file)
    data["lineage_details_df"]["year"] = data["lineage_details_df"]["year"].astype(str)

    # We will load yearly Louvain partitions and raw networks inside functions as needed.
    print("Initial data loaded: subreddit metadata, lineage details.")
    return data


def identify_radical_and_mainstream_lineages(
    loaded_data: dict, base_processed_dir: Path, project_root: Path
):
    """
    Identifies lineages containing radical subreddits and mainstream (Dem/Rep) lineages.
    """
    metadata_df = loaded_data["metadata_df"]
    lineage_details_df = loaded_data["lineage_details_df"]

    radical_subreddits = set(metadata_df[metadata_df["is_banned"]].index)
    print(
        f"Identified {len(radical_subreddits)} banned subreddits: {list(radical_subreddits)[:20]}..."
    )

    lineage_radical_sub_map = defaultdict(
        lambda: defaultdict(set)
    )  # {lineage_id: {year: {radical_subs}}}
    all_radical_lineage_ids = set()

    available_years = sorted(lineage_details_df["year"].unique())

    for year in available_years:
        yearly_louvain_partition = load_louvain_communities_for_year(
            base_processed_dir, year
        )
        if not yearly_louvain_partition:
            print(
                f"Skipping year {year} for radical lineage identification due to missing Louvain data."
            )
            continue

        # Create a reverse map from raw_community_id to subreddits for the current year
        raw_comm_to_subs = (
            yearly_louvain_partition  # This is already {raw_comm_id: {subs}}
        )

        # Filter lineage details for the current year
        lineages_this_year = lineage_details_df[lineage_details_df["year"] == year]

        for _, row in lineages_this_year.iterrows():
            lineage_id = int(row["lineage_id"])
            raw_community_id = int(row["raw_community_id"])

            if raw_community_id in raw_comm_to_subs:
                community_subreddits = raw_comm_to_subs[raw_community_id]
                found_radical_subs_in_comm = radical_subreddits.intersection(
                    community_subreddits
                )

                if found_radical_subs_in_comm:
                    all_radical_lineage_ids.add(lineage_id)
                    lineage_radical_sub_map[lineage_id][year].update(
                        found_radical_subs_in_comm
                    )

    print(
        f"Identified {len(all_radical_lineage_ids)} lineages containing at least one banned subreddit at some point: {list(all_radical_lineage_ids)[:20]}..."
    )

    # Identify mainstream Dem/Rep lineages (excluding radical ones)
    mainstream_lineages = {"dem": set(), "rep": set()}
    # Adjusted thresholds to be less strict
    dem_threshold_ratio = 0.5  # Lowered from 0.6
    rep_threshold_ratio = 0.5  # Lowered from 0.6
    other_max_ratio = 0.4  # Increased from 0.3

    # Group by lineage_id and calculate average political ratios over its lifespan
    # This gives a more stable political identity for a lineage
    avg_lineage_politics = lineage_details_df.groupby("lineage_id")[
        ["dem_ratio", "rep_ratio", "other_ratio"]
    ].mean()

    # Debug prints for top Democratic and Republican lineages
    print("\nTop 5 lineages by average Democratic ratio:")
    top_dem_lineages = avg_lineage_politics.sort_values(
        "dem_ratio", ascending=False
    ).head(5)
    for lineage_id, ratios in top_dem_lineages.iterrows():
        print(
            f"Lineage {lineage_id}: dem_ratio={ratios['dem_ratio']:.4f}, rep_ratio={ratios['rep_ratio']:.4f}, other_ratio={ratios['other_ratio']:.4f}"
        )

    print("\nTop 5 lineages by average Republican ratio:")
    top_rep_lineages = avg_lineage_politics.sort_values(
        "rep_ratio", ascending=False
    ).head(5)
    for lineage_id, ratios in top_rep_lineages.iterrows():
        print(
            f"Lineage {lineage_id}: dem_ratio={ratios['dem_ratio']:.4f}, rep_ratio={ratios['rep_ratio']:.4f}, other_ratio={ratios['other_ratio']:.4f}"
        )

    for lineage_id, ratios in avg_lineage_politics.iterrows():
        if lineage_id in all_radical_lineage_ids:
            continue  # Skip radical lineages for mainstream classification

        if (
            ratios["dem_ratio"] >= dem_threshold_ratio
            and ratios["rep_ratio"] <= other_max_ratio
        ):
            mainstream_lineages["dem"].add(lineage_id)
        elif (
            ratios["rep_ratio"] >= rep_threshold_ratio
            and ratios["dem_ratio"] <= other_max_ratio
        ):
            mainstream_lineages["rep"].add(lineage_id)

    print(
        f"Identified {len(mainstream_lineages['dem'])} mainstream Democrat lineages: {list(mainstream_lineages['dem'])[:10]}..."
    )
    print(
        f"Identified {len(mainstream_lineages['rep'])} mainstream Republican lineages: {list(mainstream_lineages['rep'])[:10]}..."
    )

    return all_radical_lineage_ids, lineage_radical_sub_map, mainstream_lineages


def plot_radical_lineage_properties(
    radical_lineage_ids: set,
    lineage_radical_sub_map: dict,  # {lineage_id: {year: {radical_subs}}}
    lineage_details_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    base_processed_dir: Path,
    output_dir: Path,
):
    """Plots properties of radical-containing lineages over time."""
    print(
        f"\nPlotting properties for {len(radical_lineage_ids)} radical-containing lineages..."
    )
    ensure_dir(output_dir)

    # Get political party mapping from metadata, handling missing 'party' key
    if "party" in metadata_df.columns:
        subreddit_to_party = metadata_df["party"].to_dict()
    else:
        print(
            "Warning: 'party' column not found in metadata_df. Cannot determine Dem/Rep labeled percentages."
        )
        subreddit_to_party = {}

    for lineage_id in radical_lineage_ids:
        lineage_data = lineage_details_df[
            lineage_details_df["lineage_id"] == lineage_id
        ].copy()  # Use .copy() to avoid SettingWithCopyWarning
        if lineage_data.empty:
            continue

        # Calculate % of banned, dem-labeled, and rep-labeled subreddits
        percent_banned_list = []
        percent_dem_labeled_list = []
        percent_rep_labeled_list = []

        # We need all subreddits in the lineage for each year
        for year_str in lineage_data["year"]:
            yearly_louvain_partition = load_louvain_communities_for_year(
                base_processed_dir, year_str
            )
            lineage_year_info = lineage_data[lineage_data["year"] == year_str]

            current_community_subreddits = set()
            if not lineage_year_info.empty and yearly_louvain_partition:
                raw_comm_id = lineage_year_info.iloc[0]["raw_community_id"]
                current_community_subreddits = yearly_louvain_partition.get(
                    raw_comm_id, set()
                )

            num_banned_in_year = len(
                lineage_radical_sub_map.get(lineage_id, {}).get(year_str, set())
            )

            num_dem_labeled = 0
            num_rep_labeled = 0
            if current_community_subreddits:
                for sub_name in current_community_subreddits:
                    party = subreddit_to_party.get(sub_name, "")
                    if party == "dem":
                        num_dem_labeled += 1
                    elif party == "rep":
                        num_rep_labeled += 1

            total_size_in_year = (
                lineage_year_info["size"].iloc[0] if not lineage_year_info.empty else 0
            )

            percent_banned = (
                (num_banned_in_year / total_size_in_year * 100)
                if total_size_in_year > 0
                else 0
            )
            percent_dem_labeled = (
                (num_dem_labeled / total_size_in_year * 100)
                if total_size_in_year > 0
                else 0
            )
            percent_rep_labeled = (
                (num_rep_labeled / total_size_in_year * 100)
                if total_size_in_year > 0
                else 0
            )
            percent_banned_list.append(percent_banned)
            percent_dem_labeled_list.append(percent_dem_labeled)
            percent_rep_labeled_list.append(percent_rep_labeled)

        lineage_data.loc[:, "percent_banned"] = percent_banned_list
        lineage_data.loc[:, "percent_dem_labeled"] = percent_dem_labeled_list
        lineage_data.loc[:, "percent_rep_labeled"] = percent_rep_labeled_list

        fig, axs = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

        # Plot 1: Size and Homogeneity
        axs[0].plot(
            lineage_data["year"], lineage_data["size"], marker="o", label="Size"
        )
        axs[0].set_ylabel("Community Size", color="blue")
        axs[0].tick_params(axis="y", labelcolor="blue")

        ax0_twin = axs[0].twinx()
        ax0_twin.plot(
            lineage_data["year"],
            lineage_data["homogeneity_score"],
            marker="x",
            linestyle="--",
            color="red",
            label="Homogeneity Score",
        )
        ax0_twin.set_ylabel("Homogeneity Score", color="red")
        ax0_twin.tick_params(axis="y", labelcolor="red")
        axs[0].set_title(
            f"Size & Homogeneity for Radical-Containing Lineage {lineage_id}"
        )
        fig.legend(
            loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=axs[0].transAxes
        )

        # Plot 2: Political Ratios and % Banned
        axs[1].plot(
            lineage_data["year"],
            lineage_data["dem_ratio"],
            marker="s",
            label="Dem Ratio (Overall)",
            alpha=0.7,
        )
        axs[1].plot(
            lineage_data["year"],
            lineage_data["rep_ratio"],
            marker="^",
            label="Rep Ratio (Overall)",
            alpha=0.7,
        )
        axs[1].plot(
            lineage_data["year"],
            lineage_data["other_ratio"],
            marker="d",
            label="Other Ratio (Overall)",
            alpha=0.7,
        )
        axs[1].plot(
            lineage_data["year"],
            lineage_data["percent_banned"],
            marker="*",
            linestyle=":",
            color="purple",
            label="% Banned Subreddits",
        )
        # New plots for labeled percentages
        axs[1].plot(
            lineage_data["year"],
            lineage_data["percent_dem_labeled"],
            marker="P",  # Plus sign
            linestyle="--",
            color="blue",
            label="% Dem Labeled Subs",
        )
        axs[1].plot(
            lineage_data["year"],
            lineage_data["percent_rep_labeled"],
            marker="X",  # X mark
            linestyle="--",
            color="red",
            label="% Rep Labeled Subs",
        )
        axs[1].set_title(
            f"Political Composition, Banned %, and Labeled % for Lineage {lineage_id}"
        )
        axs[1].set_xlabel("Year")
        axs[1].set_ylabel("Ratio / Percentage")
        axs[1].legend()
        axs[1].grid(True, alpha=0.4)

        plt.xticks(rotation=45)
        plt.tight_layout(
            rect=[0, 0, 1, 0.96]
        )  # Adjust layout to make space for main title if needed
        # fig.suptitle(f"Evolution of Radical-Containing Lineage {lineage_id}", fontsize=16) # Optional main title

        plot_path = output_dir / f"radical_lineage_{lineage_id}_evolution.png"
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)
        print(
            f"Saved evolution plot for radical-containing lineage {lineage_id} to {plot_path}"
        )


def analyze_radical_mainstream_connectivity(
    radical_lineage_ids: set,
    mainstream_lineages: dict,  # {'dem': set(), 'rep': set()}
    lineage_details_df: pd.DataFrame,
    base_processed_dir: Path,
    project_root: Path,
    output_dir: Path,
):
    """Analyzes connectivity between radical and mainstream lineages."""
    print("\nAnalyzing connectivity between radical and mainstream lineages...")
    ensure_dir(output_dir)

    yearly_connectivity_data = []
    available_years = sorted(lineage_details_df["year"].unique())

    # Helper to get subreddits for a lineage in a year
    def get_lineage_nodes_for_year(
        lineage_id, year, lineage_details_df, yearly_louvain_partition
    ):
        lineage_year_info = lineage_details_df[
            (lineage_details_df["lineage_id"] == lineage_id)
            & (lineage_details_df["year"] == year)
        ]
        if lineage_year_info.empty:
            return set()
        raw_comm_id = lineage_year_info.iloc[0]["raw_community_id"]
        return yearly_louvain_partition.get(raw_comm_id, set())

    for year in available_years:
        print(f"  Processing connectivity for year: {year}")
        yearly_louvain_partition = load_louvain_communities_for_year(
            base_processed_dir, year
        )
        raw_network_file = (
            project_root / "data" / RAW_NETWORKS_DIR_NAME / f"networks_{year}.csv"
        )

        if not raw_network_file.exists():
            print(
                f"    Raw network for {year} not found. Skipping connectivity analysis for this year."
            )
            continue
        graph = load_network(
            raw_network_file, weighted=False
        )  # Use unweighted for edge counting

        if not yearly_louvain_partition or not graph.nodes():
            print(f"    Louvain partition or graph missing for {year}. Skipping.")
            continue

        total_links_radical_to_dem = 0
        total_links_radical_to_rep = 0
        num_radical_mainstream_dem_pairs = 0
        num_radical_mainstream_rep_pairs = 0

        active_radical_lineages_this_year = {
            lid
            for lid in radical_lineage_ids
            if not lineage_details_df[
                (lineage_details_df["lineage_id"] == lid)
                & (lineage_details_df["year"] == year)
            ].empty
        }
        active_mainstream_dem_lineages = {
            lid
            for lid in mainstream_lineages["dem"]
            if not lineage_details_df[
                (lineage_details_df["lineage_id"] == lid)
                & (lineage_details_df["year"] == year)
            ].empty
        }
        active_mainstream_rep_lineages = {
            lid
            for lid in mainstream_lineages["rep"]
            if not lineage_details_df[
                (lineage_details_df["lineage_id"] == lid)
                & (lineage_details_df["year"] == year)
            ].empty
        }

        # Radical to Dem
        for rad_LID in active_radical_lineages_this_year:
            rad_nodes = get_lineage_nodes_for_year(
                rad_LID, year, lineage_details_df, yearly_louvain_partition
            )
            if not rad_nodes:
                continue
            for dem_LID in active_mainstream_dem_lineages:
                if rad_LID == dem_LID:
                    continue  # Should not happen if sets are distinct
                dem_nodes = get_lineage_nodes_for_year(
                    dem_LID, year, lineage_details_df, yearly_louvain_partition
                )
                if not dem_nodes:
                    continue

                linkage = calculate_inter_community_linkage(
                    graph, rad_nodes, dem_nodes
                )  # Assumes this helper is available
                total_links_radical_to_dem += linkage
                if linkage > 0:
                    num_radical_mainstream_dem_pairs += 1

        # Radical to Rep
        for rad_LID in active_radical_lineages_this_year:
            rad_nodes = get_lineage_nodes_for_year(
                rad_LID, year, lineage_details_df, yearly_louvain_partition
            )
            if not rad_nodes:
                continue
            for rep_LID in active_mainstream_rep_lineages:
                if rad_LID == rep_LID:
                    continue
                rep_nodes = get_lineage_nodes_for_year(
                    rep_LID, year, lineage_details_df, yearly_louvain_partition
                )
                if not rep_nodes:
                    continue

                linkage = calculate_inter_community_linkage(graph, rad_nodes, rep_nodes)
                total_links_radical_to_rep += linkage
                if linkage > 0:
                    num_radical_mainstream_rep_pairs += 1

        yearly_connectivity_data.append(
            {
                "year": year,
                "total_links_radical_to_dem": total_links_radical_to_dem,
                "total_links_radical_to_rep": total_links_radical_to_rep,
                "num_radical_mainstream_dem_pairs_linked": num_radical_mainstream_dem_pairs,
                "num_radical_mainstream_rep_pairs_linked": num_radical_mainstream_rep_pairs,
                "num_active_radical_lineages": len(active_radical_lineages_this_year),
                "num_active_mainstream_dem_lineages": len(
                    active_mainstream_dem_lineages
                ),
                "num_active_mainstream_rep_lineages": len(
                    active_mainstream_rep_lineages
                ),
            }
        )

    if not yearly_connectivity_data:
        print("No yearly connectivity data generated.")
        return

    connectivity_df = pd.DataFrame(yearly_connectivity_data)
    connectivity_df.to_csv(
        output_dir / "radical_mainstream_connectivity_summary.csv", index=False
    )
    print(
        f"Saved radical-mainstream connectivity summary to {output_dir / 'radical_mainstream_connectivity_summary.csv'}"
    )

    # Plotting trends
    plt.figure(figsize=(12, 6))
    plt.plot(
        connectivity_df["year"],
        connectivity_df["total_links_radical_to_dem"],
        marker="o",
        label="Radical to Mainstream Dem Links",
    )
    plt.plot(
        connectivity_df["year"],
        connectivity_df["total_links_radical_to_rep"],
        marker="s",
        label="Radical to Mainstream Rep Links",
    )
    plt.xlabel("Year")
    plt.ylabel("Total Inter-Community Links")
    plt.title("Connectivity of Radical Lineages to Mainstream Political Lineages")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "radical_mainstream_connectivity_trends.png", dpi=300)
    plt.close()
    print(
        f"Saved radical-mainstream connectivity trend plot to {output_dir / 'radical_mainstream_connectivity_trends.png'}"
    )


def calculate_inter_community_linkage(
    graph, community1_nodes: set[str], community2_nodes: set[str]
) -> int:
    """Counts the number of edges between two disjoint sets of nodes in a graph."""
    linkage_count = 0
    # Iterate over nodes of the smaller community for efficiency
    iter_nodes, check_nodes = (
        (community1_nodes, community2_nodes)
        if len(community1_nodes) <= len(community2_nodes)
        else (community2_nodes, community1_nodes)
    )

    for node1 in iter_nodes:
        if node1 in graph:  # Ensure node from community is in the graph (it should be)
            for neighbor in graph.neighbors(node1):
                if neighbor in check_nodes:
                    linkage_count += (
                        1  # Assumes unweighted graph for inter-community edges
                    )
    return linkage_count


def main():
    project_root = get_project_root()
    base_data_dir = project_root / "data"
    base_processed_dir = base_data_dir / "processed"

    step4_output_path = base_processed_dir / STEP4_OUTPUT_DIR_NAME
    visualizations_output_path = step4_output_path / VISUALIZATIONS_SUBDIR
    ensure_dir(step4_output_path)
    ensure_dir(visualizations_output_path)

    print(
        f"Starting Step 4: Radical Community Analysis. Outputs will be in {step4_output_path}"
    )

    try:
        loaded_data = load_all_data(project_root, base_processed_dir)

        radical_lineage_ids, lineage_radical_sub_map, mainstream_lineages = (
            identify_radical_and_mainstream_lineages(
                loaded_data, base_processed_dir, project_root
            )
        )

        if radical_lineage_ids:
            plot_radical_lineage_properties(
                radical_lineage_ids,
                lineage_radical_sub_map,
                loaded_data["lineage_details_df"],
                loaded_data["metadata_df"],
                base_processed_dir,
                visualizations_output_path,
            )

            # Only proceed with connectivity if mainstream lineages are found
            if mainstream_lineages["dem"] or mainstream_lineages["rep"]:
                analyze_radical_mainstream_connectivity(
                    radical_lineage_ids,
                    mainstream_lineages,
                    loaded_data["lineage_details_df"],
                    base_processed_dir,
                    project_root,
                    step4_output_path,
                )
            else:
                print(
                    "\nSkipping connectivity analysis as no mainstream Dem/Rep lineages were identified with current thresholds."
                )
        else:
            print(
                "No radical-containing lineages identified. Skipping further analysis."
            )

    except FileNotFoundError as e:
        print(f"Error: A required data file was not found: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
        return

    print(
        f"\nStep 4: Radical Community Analysis complete. Outputs are in {step4_output_path}"
    )


if __name__ == "__main__":
    main()
