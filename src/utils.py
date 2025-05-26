import pandas as pd
import networkx as nx
import json
from pathlib import Path
import os
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from collections import Counter, defaultdict

# Try to import community_louvain for a consistent modularity calculation
try:
    import community as community_louvain
except ImportError:
    community_louvain = None
    print(
        "Info: `community` package not found in utils.py. Standalone modularity calculation will rely on NetworkX if available, or be disabled."
    )


def ensure_dir(directory_path: Path) -> None:
    """
    Ensures that a directory exists, creating it if necessary.

    Args:
        directory_path (Path): The path to the directory.
    """
    directory_path.mkdir(parents=True, exist_ok=True)


def load_network(
    network_file_path: Path,
    weighted: bool = False,
    node_1_col: str = "node_1",
    node_2_col: str = "node_2",
    weight_col: str = "weighted",
    unweighted_col: str = "unweighted",
) -> nx.Graph:
    """
    Loads a network from a CSV file into a NetworkX graph.

    Args:
        network_file_path (Path): The path to the network CSV file.
        weighted (bool): If True, loads the graph using the 'weight_col' for edge weights.
                         If False, loads the graph using the 'unweighted_col' as binary indicators of edges,
                         and if an edge exists (value is 1), it's added without a specific weight attribute
                         unless 'weight_col' is also present and non-zero for that edge (then it's used).
                         Effectively, for unweighted=True, we primarily care about edge existence based on unweighted_col.
        node_1_col (str): Name of the column for the first node.
        node_2_col (str): Name of the column for the second node.
        weight_col (str): Name of the column for edge weights (used if weighted=True).
        unweighted_col (str): Name of the column indicating unweighted edge existence (1 for exists, 0 for not).

    Returns:
        nx.Graph: The loaded graph.
    """
    df = pd.read_csv(network_file_path)
    G = nx.Graph()

    for _, row in df.iterrows():
        node1 = row[node_1_col]
        node2 = row[node_2_col]

        if weighted:
            weight = float(row[weight_col])
            if weight > 0:  # Only add edge if weight is positive
                G.add_edge(node1, node2, weight=weight)
        else:
            # For unweighted, check the unweighted column
            if int(row[unweighted_col]) == 1:
                # Add edge. If a weight column is also present and non-zero, capture it.
                # This aligns with how unweighted networks from backboning might still have original weights.
                edge_attributes = {}
                if (
                    weight_col in df.columns
                    and pd.notna(row[weight_col])
                    and float(row[weight_col]) > 0
                ):
                    edge_attributes["weight"] = float(row[weight_col])
                G.add_edge(node1, node2, **edge_attributes)

    return G


def load_subreddit_metadata(metadata_file_path: Path, seed_key: str = "party") -> dict:
    """
    Loads subreddit metadata from a JSONL file.

    Args:
        metadata_file_path (Path): Path to the JSONL metadata file.
        seed_key (str): The key in the JSON objects that contains the seed label (e.g., 'party').

    Returns:
        dict: A dictionary mapping subreddit names to their seed labels.
              Returns empty dict if file not found or error occurs.
    """
    seed_labels = {}
    if not metadata_file_path.exists():
        print(f"Warning: Metadata file not found at {metadata_file_path}")
        return seed_labels

    with open(metadata_file_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if (
                    seed_key in data and data[seed_key]
                ):  # Ensure the key exists and has a value
                    seed_labels[data["subreddit"]] = data[seed_key]
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from line: {line.strip()}")
                continue
    return seed_labels


def get_project_root() -> Path:
    """Returns the project root directory. Assumes this script is in src/"""
    return Path(__file__).parent.parent


def calculate_modularity(
    G: nx.Graph, partition: dict[str, int], weight: str = "weight"
) -> float | None:
    """
    Calculates the modularity of a given partition of a graph.

    Args:
        G (nx.Graph): The graph.
        partition (dict): A dictionary mapping node names to community IDs.
        weight (str): The key for edge weights. Defaults to 'weight'.

    Returns:
        float | None: The modularity score, or None if calculation is not possible.
    """
    if not G.edges() or not partition:
        return 0.0  # Or None, depending on desired behavior for empty/trivial cases

    if community_louvain:
        return community_louvain.modularity(partition, G, weight=weight)
    elif hasattr(nx, "community") and hasattr(nx.community, "modularity"):
        # NetworkX modularity expects a list of sets/frozensets for communities
        communities_map = defaultdict(set)
        for node, comm_id in partition.items():
            communities_map[comm_id].add(node)
        list_of_communities = [frozenset(c) for c in communities_map.values()]
        if not list_of_communities:
            return 0.0
        try:
            return nx.community.modularity(G, list_of_communities, weight=weight)
        except Exception as e:
            print(f"Error calculating modularity with NetworkX: {e}")
            return None
    else:
        print(
            "Warning: Neither 'community' package nor NetworkX modularity function is available."
        )
        return None


def calculate_purity(
    true_labels: pd.Series, predicted_clusters: pd.Series
) -> float | None:
    """
    Calculates the Purity score for clustering.

    Purity is the sum of correctly classified data points divided by the total number of data points.
    Assumes true_labels and predicted_clusters are pandas Series aligned by index (node names).

    Args:
        true_labels (pd.Series): Series of true labels (e.g., party affiliation), indexed by node name.
        predicted_clusters (pd.Series): Series of predicted cluster IDs, indexed by node name.

    Returns:
        float | None: The purity score, or None if inputs are invalid.
    """
    if not isinstance(true_labels, pd.Series) or not isinstance(
        predicted_clusters, pd.Series
    ):
        print("Error: true_labels and predicted_clusters must be pandas Series.")
        return None

    # Align labels and clusters based on common nodes (index)
    common_nodes = true_labels.index.intersection(predicted_clusters.index)
    if common_nodes.empty:
        print(
            "Warning: No common nodes between true labels and predicted clusters for purity calculation."
        )
        return 0.0  # Or None

    true_labels_aligned = true_labels.loc[common_nodes]
    predicted_clusters_aligned = predicted_clusters.loc[common_nodes]

    if true_labels_aligned.empty or predicted_clusters_aligned.empty:
        print("Warning: Alignment resulted in empty series for purity calculation.")
        return 0.0

    contingency_matrix = pd.crosstab(predicted_clusters_aligned, true_labels_aligned)

    # Sum of the max values in each row (for each cluster, find the count of the dominant true label)
    dominant_label_sum = contingency_matrix.max(axis=1).sum()
    total_samples = len(true_labels_aligned)

    if total_samples == 0:
        return 0.0  # Avoid division by zero if no samples after alignment

    purity = dominant_label_sum / total_samples
    return purity


def calculate_nmi(
    true_labels: pd.Series, predicted_clusters: pd.Series
) -> float | None:
    """
    Calculates the Normalized Mutual Information (NMI) for clustering.
    Assumes true_labels and predicted_clusters are pandas Series aligned by index (node names).

    Args:
        true_labels (pd.Series): Series of true labels, indexed by node name.
        predicted_clusters (pd.Series): Series of predicted cluster IDs, indexed by node name.

    Returns:
        float | None: The NMI score, or None if inputs are invalid.
    """
    if not isinstance(true_labels, pd.Series) or not isinstance(
        predicted_clusters, pd.Series
    ):
        print("Error: true_labels and predicted_clusters must be pandas Series.")
        return None

    common_nodes = true_labels.index.intersection(predicted_clusters.index)
    if common_nodes.empty:
        print(
            "Warning: No common nodes between true labels and predicted clusters for NMI calculation."
        )
        return 0.0  # Or None

    true_labels_aligned = true_labels.loc[common_nodes].astype(
        str
    )  # NMI expects consistent type, often string/int
    predicted_clusters_aligned = predicted_clusters.loc[common_nodes].astype(str)

    if true_labels_aligned.empty or predicted_clusters_aligned.empty:
        print("Warning: Alignment resulted in empty series for NMI calculation.")
        return 0.0

    # sklearn's NMI can handle string labels directly
    try:
        nmi = normalized_mutual_info_score(
            true_labels_aligned, predicted_clusters_aligned, average_method="arithmetic"
        )
        return nmi
    except ValueError as e:
        print(
            f"Error calculating NMI: {e}. This might happen if one set of labels is constant."
        )
        # If one set of labels is constant (e.g., all nodes in one cluster, or all true labels are the same)
        # NMI can be undefined or 0. Returning 0 in such cases or if other errors occur.
        return 0.0


def load_yearly_graph_from_csv(
    network_data_dir: Path,
    year: int,
    node_1_col: str = "node_1",
    node_2_col: str = "node_2",
    unweighted_col: str = "unweighted",
) -> nx.Graph:
    """
    Loads the unweighted network for a given year from a CSV file.
    Filters for edges where the unweighted_col == 1.
    Node labels from node_1_col and node_2_col are read as strings.

    Args:
        network_data_dir (Path): The directory containing the network CSV files.
        year (int): The year for which to load the graph.
        node_1_col (str): Name of the column for the first node.
        node_2_col (str): Name of the column for the second node.
        unweighted_col (str): Name of the column indicating unweighted edge existence.

    Returns:
        nx.Graph: The loaded graph, or an empty graph if file not found or error occurs.
    """
    try:
        graph_file_name = f"networks_{year}.csv"
        graph_path = network_data_dir / graph_file_name

        if not graph_path.exists():
            print(
                f"Warning: Graph file for year {year} not found at {graph_path}. Returning empty graph."
            )
            return nx.Graph()

        edge_df = pd.read_csv(graph_path, dtype={node_1_col: str, node_2_col: str})

        if unweighted_col not in edge_df.columns:
            print(
                f"Warning: '{unweighted_col}' column not found in {graph_path}. Attempting to use all edges."
            )
            # If unweighted_col is missing, we might assume all edges are part of the intended graph.
            # This part of the logic depends on how you want to handle missing 'unweighted' column.
            # For now, create graph with all edges if column is missing.
            unweighted_edges_df = edge_df
        else:
            # Filter for edges that are part of the unweighted backbone
            unweighted_edges_df = edge_df[edge_df[unweighted_col] == 1]

        if unweighted_edges_df.empty:
            # print(f"No unweighted edges found or file was empty for year {year} in {graph_path}.")
            return nx.Graph()

        graph_year = nx.from_pandas_edgelist(
            unweighted_edges_df, node_1_col, node_2_col
        )
        return graph_year

    except pd.errors.EmptyDataError:
        print(
            f"Warning: Graph file for year {year} ({graph_path}) is empty. Returning empty graph."
        )
        return nx.Graph()
    except Exception as e:
        print(
            f"Error loading graph for year {year} from {graph_path}: {e}. Returning empty graph."
        )
        return nx.Graph()


def load_metadata_dataframe_from_jsonl(
    file_path: Path, fields_to_keep: list[str], default_values: dict = None
) -> pd.DataFrame:
    """
    Loads metadata from a JSONL file into a Pandas DataFrame, keeping specified fields.

    Args:
        file_path (Path): Path to the JSONL metadata file.
        fields_to_keep (list[str]): A list of field names to extract from the JSON objects.
                                    'subreddit' field is mandatory and will always be included.
        default_values (dict, optional): A dictionary mapping field names to default values
                                         to use if a field is missing in a JSON object.
                                         Defaults to None, meaning missing fields will be NaN/None initially.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted metadata.
    """
    if not file_path.exists():
        print(f"Error: Metadata file '{file_path}' not found.")
        return pd.DataFrame()

    if default_values is None:
        default_values = {}

    processed_fields_to_keep = ["subreddit"] + [
        f for f in fields_to_keep if f != "subreddit"
    ]

    data_list = []
    with open(file_path, "r") as f:
        for line_number, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
            except json.JSONDecodeError:
                print(
                    f"Warning: Could not decode JSON from line {line_number} in {file_path}: {line.strip()}"
                )
                continue

            if "subreddit" not in record or not record["subreddit"]:
                print(
                    f"Warning: 'subreddit' field missing or empty in line {line_number} in {file_path}. Skipping record."
                )
                continue

            row_data = {"subreddit": str(record["subreddit"])}

            for field in processed_fields_to_keep:
                if field == "subreddit":
                    continue
                row_data[field] = record.get(field, default_values.get(field))

            data_list.append(row_data)

    if not data_list:
        return pd.DataFrame(columns=processed_fields_to_keep)

    df = pd.DataFrame(data_list)

    if "party" in df.columns:
        df["party"] = (
            df["party"]
            .astype(str)
            .replace(["nan", "None"], default_values.get("party", ""))
        )
        if default_values.get("party") is None:
            df["party"] = df["party"].replace("None", "")

    if "banned" in df.columns:
        df["banned"] = (
            pd.to_numeric(df["banned"], errors="coerce")
            .fillna(default_values.get("banned", 0))
            .astype(int)
        )

    if "gun" in df.columns:
        df["gun"] = (
            pd.to_numeric(df["gun"], errors="coerce")
            .fillna(default_values.get("gun", 0))
            .astype(int)
        )

    for field in processed_fields_to_keep:
        if field not in df.columns:
            df[field] = default_values.get(field, None)

    final_columns = [col for col in processed_fields_to_keep if col in df.columns]
    return df[final_columns]


def load_bot_user_ids(users_metadata_file: Path) -> set[str]:
    """
    Loads bot user IDs from a JSONL metadata file.
    Each line in the file is expected to be a JSON object.
    It looks for an 'author' (or 'Author') key for the user ID and a 'bot' key with value 1.

    Args:
        users_metadata_file (Path): Path to the users_metadata.jsonl file.

    Returns:
        set[str]: A set of author IDs identified as bots.
    """
    bot_ids = set()
    if not users_metadata_file.exists():
        print(
            f"Warning: Users metadata file not found at {users_metadata_file}. No bots will be filtered."
        )
        return bot_ids

    try:
        with open(users_metadata_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    is_bot = data.get("bot") == 1
                    author_id = None
                    if "author" in data:  # Preferred key based on comments data
                        author_id = data["author"]

                    if is_bot and author_id:
                        bot_ids.add(str(author_id))  # Ensure ID is string
                except json.JSONDecodeError:
                    print(
                        f"Warning: JSON decode error on line {line_num} in {users_metadata_file}: {line.strip()}"
                    )
                    continue
    except Exception as e:
        print(
            f"Error reading or processing users metadata file {users_metadata_file}: {e}"
        )

    if bot_ids:
        print(f"Loaded {len(bot_ids)} bot user IDs from {users_metadata_file.name}.")
    else:
        print(
            f"No bot user IDs loaded from {users_metadata_file.name}. This could be due to file format, content, or the file being empty."
        )
    return bot_ids


if __name__ == "__main__":
    # Example Usage (assuming you have the data in the expected structure relative to project root)
    project_root = get_project_root()
    print(f"Project root: {project_root}")

    # Define paths relative to the detected project root
    data_dir = project_root / "data"
    networks_dir = data_dir / "networks"
    metadata_dir = data_dir / "metadata"
    processed_dir = data_dir / "processed"

    ensure_dir(networks_dir)
    ensure_dir(metadata_dir)
    ensure_dir(processed_dir)

    # Dummy file paths
    dummy_network_file = networks_dir / "dummy_network_2019.csv"
    dummy_metadata_file = metadata_dir / "dummy_subreddits_metadata.jsonl"

    # Create dummy network file if it doesn't exist
    if not dummy_network_file.exists():
        print(f"Creating dummy network file: {dummy_network_file}")
        network_content = """node_1,node_2,weighted,unweighted
subA,subB,5,1
subB,subC,10,1
subA,subC,0,0
subC,subD,3,1
subD,subE,2,1
subE,subF,7,1
subG,subH,4,1
"""
        with open(dummy_network_file, "w") as f:
            f.write(network_content)

    # Create dummy metadata file if it doesn't exist
    if not dummy_metadata_file.exists():
        print(f"Creating dummy metadata file: {dummy_metadata_file}")
        metadata_content = """{"subreddit": "subA", "party": "dem"}
{"subreddit": "subB", "party": "dem"}
{"subreddit": "subC", "party": "rep"}
{"subreddit": "subD", "party": "rep"}
{"subreddit": "subE", "party": "rep"}
{"subreddit": "subF", "party": "dem"}
{"subreddit": "subG", "party": "neutral"}
{"subreddit": "subH", "party": "neutral"}
{"subreddit": "subNonExistent", "party": "dem"}
"""
        with open(dummy_metadata_file, "w") as f:
            f.write(metadata_content)

    print(f"\n--- Testing load_network (unweighted) from {dummy_network_file} ---")
    G_unweighted = load_network(dummy_network_file, weighted=False)
    print(f"Unweighted Graph Nodes: {G_unweighted.nodes()}")

    print(f"\n--- Testing load_subreddit_metadata from {dummy_metadata_file} ---")
    subreddit_truth = load_subreddit_metadata(dummy_metadata_file, seed_key="party")
    print(f"Metadata (True Labels): {subreddit_truth}")

    # Create a dummy partition for testing evaluation metrics
    # Nodes: subA, subB, subC, subD, subE, subF, subG, subH
    # True:  dem,  dem,  rep,  rep,  rep,  dem,  neutral, neutral
    dummy_partition_data = {
        "subA": 0,
        "subB": 0,
        "subC": 1,
        "subD": 1,
        "subE": 0,
        "subF": 1,
        "subG": 2,
        "subH": 2,
        "subMissingFromTruth": 0,  # A node in partition but not in true_labels for testing alignment
    }
    dummy_predicted_clusters_series = pd.Series(
        dummy_partition_data, name="predicted_cluster"
    )
    dummy_true_labels_series = pd.Series(subreddit_truth, name="true_label")

    print(f"\n--- Testing calculate_modularity ---")
    # For modularity, G_unweighted can be used, assuming unweighted scenario for this partition
    # or load a weighted one if partition is from a weighted algorithm.
    # For this dummy test, let's use G_unweighted. If it has weights, they will be used if specified.
    modularity_score = calculate_modularity(
        G_unweighted, dummy_partition_data, weight="weight"
    )  # `load_network` adds weight if present
    if modularity_score is not None:
        print(f"Modularity Score: {modularity_score:.4f}")
    else:
        print("Modularity calculation failed or not available.")

    print(f"\n--- Testing calculate_purity ---")
    purity_score = calculate_purity(
        dummy_true_labels_series, dummy_predicted_clusters_series
    )
    if purity_score is not None:
        print(f"Purity Score: {purity_score:.4f}")
    else:
        print("Purity calculation failed.")

    print(f"\n--- Testing calculate_nmi ---")
    nmi_score = calculate_nmi(dummy_true_labels_series, dummy_predicted_clusters_series)
    if nmi_score is not None:
        print(f"NMI Score: {nmi_score:.4f}")
    else:
        print("NMI calculation failed.")

    # Test with perfect alignment (subset of nodes)
    print("\n--- Testing Purity & NMI with perfectly aligned known nodes ---")
    known_nodes_true = pd.Series({"subA": "dem", "subB": "dem", "subC": "rep"})
    known_nodes_pred = pd.Series({"subA": 0, "subB": 0, "subC": 1})
    purity_perfect = calculate_purity(known_nodes_true, known_nodes_pred)
    nmi_perfect = calculate_nmi(known_nodes_true, known_nodes_pred)
    print(f"Purity (perfect subset): {purity_perfect}")  # Expected: 1.0
    print(f"NMI (perfect subset): {nmi_perfect}")  # Expected: 1.0

    print("\nUtils.py evaluation functions testing complete.")
