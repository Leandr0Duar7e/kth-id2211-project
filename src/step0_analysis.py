import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
from utils import load_metadata_dataframe_from_jsonl, load_yearly_graph_from_csv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# relative paths from the script's location (src/)
RELATIVE_BASE_DATA_DIR = "../data"
RELATIVE_METADATA_SUBDIR = "metadata"
RELATIVE_NETWORKS_SUBDIR = "networks"
RELATIVE_PROCESSED_SUBDIR = "processed"
RELATIVE_SHORTEST_PATHS_SUBDIR = "shortest_paths"

# Construct absolute paths
ABSOLUTE_BASE_DATA_DIR = os.path.normpath(
    os.path.join(SCRIPT_DIR, RELATIVE_BASE_DATA_DIR)
)

ABSOLUTE_METADATA_FILE_PATH = os.path.join(
    ABSOLUTE_BASE_DATA_DIR, RELATIVE_METADATA_SUBDIR, "subreddits_metadata.json"
)
ABSOLUTE_NETWORK_DATA_DIR = os.path.join(
    ABSOLUTE_BASE_DATA_DIR, RELATIVE_NETWORKS_SUBDIR
)
ABSOLUTE_OUTPUT_DIR = os.path.join(
    ABSOLUTE_BASE_DATA_DIR, RELATIVE_PROCESSED_SUBDIR, RELATIVE_SHORTEST_PATHS_SUBDIR
)


MAIN_SUBREDDITS = [
    "politics",
    "The_Donald",
    "SandersForPresident",
    "Libertarian",
    "Conservative",
]
YEARS = range(2008, 2019 + 1)


def get_target_nodes(metadata_df, group_type):
    """
    Returns a set of subreddit names for the specified target group.
    """
    if metadata_df.empty:
        return set()
    if group_type == "dem" and "party" in metadata_df.columns:
        return set(metadata_df[metadata_df["party"] == "dem"]["subreddit"])
    elif group_type == "rep" and "party" in metadata_df.columns:
        return set(metadata_df[metadata_df["party"] == "rep"]["subreddit"])
    elif group_type == "gun" and "gun" in metadata_df.columns:
        return set(metadata_df[metadata_df["gun"] == 1]["subreddit"])
    return set()


def calculate_avg_shortest_path(graph, source_node_str, target_node_set):
    """
    Calculates the average shortest path length from a source node to a set of target nodes.
    Returns np.nan if source_node is not in graph, no valid targets, or no paths found.
    """
    if source_node_str not in graph:
        return np.nan

    valid_target_nodes = [
        n_str
        for n_str in target_node_set
        if n_str in graph and n_str != source_node_str
    ]

    if not valid_target_nodes:
        return np.nan

    path_lengths = []
    for target_node_str in valid_target_nodes:
        try:
            length = nx.shortest_path_length(
                graph, source=source_node_str, target=target_node_str
            )
            path_lengths.append(length)
        except nx.NetworkXNoPath:
            pass
        except nx.NodeNotFound:
            pass

    if not path_lengths:
        return np.nan
    return np.mean(path_lengths)


def get_subreddit_plot_start_year(main_sub_name):
    """Returns the required start year for plotting for a given subreddit."""
    if main_sub_name in ["The_Donald", "SandersForPresident"]:
        return 2015
    elif main_sub_name in ["politics", "Libertarian", "Conservative"]:
        return 2012
    return min(
        YEARS
    )  # Default to earliest year if no specific rule (should not be hit with current MAIN_SUBREDDITS)


def main():
    if not os.path.exists(ABSOLUTE_OUTPUT_DIR):
        os.makedirs(ABSOLUTE_OUTPUT_DIR)
        print(f"Created output directory: {ABSOLUTE_OUTPUT_DIR}")

    fields_to_include = ["subreddit", "party", "banned", "gun"]
    field_defaults = {"party": "", "banned": 0, "gun": 0}
    metadata_path_obj = Path(ABSOLUTE_METADATA_FILE_PATH)
    subreddit_metadata = load_metadata_dataframe_from_jsonl(
        metadata_path_obj, fields_to_include, field_defaults
    )

    if subreddit_metadata.empty:
        print("Metadata is empty (or failed to load). Cannot proceed with analysis.")
        return

    all_main_sub_data = {
        sub: {
            "years": list(YEARS),
            "dem": [np.nan] * len(YEARS),
            "rep": [np.nan] * len(YEARS),
            "gun": [np.nan] * len(YEARS),
        }
        for sub in MAIN_SUBREDDITS
    }

    target_groups_nodes_global = {
        "dem": get_target_nodes(subreddit_metadata, "dem"),
        "rep": get_target_nodes(subreddit_metadata, "rep"),
        "gun": get_target_nodes(subreddit_metadata, "gun"),
    }

    global_min_y = float("inf")
    global_max_y = float("-inf")

    for i, year in enumerate(YEARS):
        print(f"Processing year: {year}")
        graph_year = load_yearly_graph_from_csv(Path(ABSOLUTE_NETWORK_DATA_DIR), year)

        if not graph_year.nodes():
            print(
                f"Graph for year {year} is empty or failed to load. Skipping calculations for this year."
            )
            continue

        current_main_subreddits_str = [str(s) for s in MAIN_SUBREDDITS]

        for main_sub_orig, main_sub_str in zip(
            MAIN_SUBREDDITS, current_main_subreddits_str
        ):
            if main_sub_str not in graph_year:
                continue

            for group_type, nodes_set in target_groups_nodes_global.items():
                avg_dist = calculate_avg_shortest_path(
                    graph_year, main_sub_str, nodes_set
                )
                all_main_sub_data[main_sub_orig][group_type][i] = avg_dist
                if pd.notna(avg_dist):
                    global_min_y = min(global_min_y, avg_dist)
                    global_max_y = max(global_max_y, avg_dist)

    # Plotting
    master_years_list = list(YEARS)
    year_to_index = {year: i for i, year in enumerate(master_years_list)}

    for main_sub_orig in MAIN_SUBREDDITS:
        data_for_sub = all_main_sub_data[main_sub_orig]

        required_start_year = get_subreddit_plot_start_year(main_sub_orig)
        plot_worthy = True
        for year_to_check in range(required_start_year, 2019 + 1):
            idx = year_to_index.get(year_to_check)
            if idx is None:  # Should not occur
                plot_worthy = False
                break
            has_data_for_year = (
                pd.notna(data_for_sub["dem"][idx])
                or pd.notna(data_for_sub["rep"][idx])
                or pd.notna(data_for_sub["gun"][idx])
            )
            if not has_data_for_year:
                plot_worthy = False
                break

        if not plot_worthy:
            print(
                f"Skipping plot for r/{main_sub_orig} due to insufficient yearly data coverage in range {required_start_year}-2019."
            )
            continue

        # Slice data for plotting according to required_start_year
        start_index = year_to_index.get(required_start_year, 0)

        plot_years = master_years_list[start_index:]
        plot_dem_data = data_for_sub["dem"][start_index:]
        plot_rep_data = data_for_sub["rep"][start_index:]
        plot_gun_data = data_for_sub["gun"][start_index:]

        plt.figure(figsize=(12, 7))
        plt.plot(
            plot_years,
            plot_dem_data,
            label="Avg. dist to Democrat subs",
            marker="o",
            linestyle="-",
            color="blue",
        )
        plt.plot(
            plot_years,
            plot_rep_data,
            label="Avg. dist to Republican subs",
            marker="x",
            linestyle="-",
            color="red",
        )
        plt.plot(
            plot_years,
            plot_gun_data,
            label="Avg. dist to Gun Control subs",
            marker="^",
            linestyle="-",
            color="black",
        )

        plt.xlabel("Year")
        plt.ylabel("Average Shortest Path Length")
        plt.title(f"Temporal Avg. Shortest Path Distances for r/{main_sub_orig}")
        plt.legend()
        plt.grid(True)
        plt.xticks(plot_years, rotation=45)  # Use plot_years for x-ticks
        plt.tight_layout()

        # Set consistent Y-axis limits for all plots
        if global_min_y != float("inf") and global_max_y != float("-inf"):
            plt.ylim(max(0, global_min_y - 0.5), global_max_y + 0.5)
        else:
            # Default if no data was found across all plots (should be rare)
            plt.ylim(0, 10)

        plot_filename = os.path.join(
            ABSOLUTE_OUTPUT_DIR, f"{main_sub_orig.replace('/', '_')}_shortest_paths.png"
        )
        try:
            plt.savefig(plot_filename)
            print(f"Saved plot: {plot_filename}")
        except Exception as e:
            print(f"Error saving plot {plot_filename}: {e}")
        plt.close()


if __name__ == "__main__":
    main()
