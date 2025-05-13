import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Assuming utils.py is in the same directory or src is in PYTHONPATH
from utils import load_network, ensure_dir, get_project_root

# Try to import the community detection package for Louvain
# This is often provided by `python-louvain` or `community` package
try:
    import community as community_louvain
except ImportError:
    print(
        "Warning: `community` package not found. Louvain clustering will not be available."
    )
    print("Please install it, e.g., by running: pip install python-louvain")
    community_louvain = None


class LouvainCommunityDetector:
    """
    Detects communities in a graph using the Louvain algorithm.
    This method is well-suited for weighted networks and optimizes modularity.
    """

    def __init__(self, random_state: int | None = None):
        """
        Initializes the LouvainCommunityDetector.

        Args:
            random_state (int | None): Seed for reproducibility of the Louvain algorithm if supported
                                     by the specific implementation (networkx's version uses it).
        """
        if community_louvain is None:
            raise ImportError("The 'community' package (for Louvain) is not installed.")
        self.random_state = random_state
        self.partition_ = None  # Stores the best partition found {node: community_id}
        self.communities_ = (
            None  # Stores communities as a list of sets {community_id: {nodes}}
        )
        self.modularity_ = None
        self.G_ = None  # Store the graph for visualization or other purposes

    def detect_communities(self, G: nx.Graph, weight_key: str = "weight"):
        """
        Detects communities in the graph using the Louvain algorithm.
        The Louvain algorithm inherently uses weights if they are present on edges
        and named 'weight'.

        Args:
            G (nx.Graph): The input graph. Must be a weighted graph for Louvain to be most effective.
            weight_key (str): The edge attribute key for weights. Defaults to 'weight'.
                              The `community_louvain` package expects this to be 'weight'.
        """
        if not G.nodes():
            print("Warning: Input graph is empty. No communities to detect.")
            self.partition_ = {}
            self.communities_ = defaultdict(set)
            self.modularity_ = None
            self.G_ = G
            return

        self.G_ = G
        # The community_louvain.best_partition function directly uses the 'weight' attribute if present.
        # Ensure your graph loading sets this attribute correctly for weighted edges.
        self.partition_ = community_louvain.best_partition(
            G, weight=weight_key, random_state=self.random_state
        )

        if not self.partition_:
            print("Warning: Louvain algorithm did not return a partition.")
            self.communities_ = defaultdict(set)
            self.modularity_ = None
            return

        # Organize communities into a more usable format: list of sets of nodes
        self.communities_ = defaultdict(set)
        for node, community_id in self.partition_.items():
            self.communities_[community_id].add(node)

        # Calculate modularity of the obtained partition
        self.modularity_ = community_louvain.modularity(
            self.partition_, G, weight=weight_key
        )
        print(
            f"Louvain algorithm detected {len(self.communities_)} communities with modularity: {self.modularity_:.4f}"
        )

    def get_partition(self) -> dict | None:
        """Returns the partition (dictionary mapping node to community ID)."""
        return self.partition_

    def get_communities(self) -> list[set]:
        """Returns the communities as a list of sets of nodes."""
        if not self.communities_:
            return []
        return list(self.communities_.values())  # Return as list of sets

    def get_community_map(self) -> dict[int, set]:
        """Returns communities as a dictionary {community_id: {nodes}}."""
        return self.communities_

    def get_modularity(self) -> float | None:
        """Returns the modularity of the detected partition."""
        return self.modularity_

    def save_results(self, output_file_path: Path):
        """
        Saves the community assignments to a CSV file.
        Each row will be: subreddit, community_id.

        Args:
            output_file_path (Path): The path to save the CSV file.
        """
        if self.partition_ is None or not self.partition_:
            print(f"No partition to save for {output_file_path.name}.")
            return

        results_df = pd.DataFrame(
            list(self.partition_.items()), columns=["subreddit", "community_id"]
        )
        results_df = results_df.sort_values(
            by=["community_id", "subreddit"]
        ).reset_index(drop=True)

        ensure_dir(output_file_path.parent)
        results_df.to_csv(output_file_path, index=False)
        print(f"Louvain community assignments saved to {output_file_path}")

    def visualize_communities(self, output_file_path: Path, **kwargs):
        """
        Creates and saves a visualization of the network with nodes colored by detected communities.

        Args:
            output_file_path (Path): The path to save the visualization.
            **kwargs: Additional arguments for layout and drawing functions.
        """
        if (
            self.G_ is None
            or not self.G_.nodes()
            or self.partition_ is None
            or not self.partition_
        ):
            print(
                f"Cannot visualize: Graph or partition not available. Skipping visualization for {output_file_path.name}"
            )
            return

        plt.figure(figsize=kwargs.get("figsize", (15, 12)))

        # Create a color map for communities
        # Ensure consistent coloring even if community IDs are not contiguous or start from non-zero
        community_ids = sorted(list(set(self.partition_.values())))
        palette = sns.color_palette("husl", len(community_ids))
        community_color_map = {
            community_id: palette[i] for i, community_id in enumerate(community_ids)
        }

        node_colors = [
            community_color_map[self.partition_[node]] for node in self.G_.nodes()
        ]

        # Layout parameters
        pos_k = (
            kwargs.get("k", 0.15 / np.sqrt(len(self.G_))) if len(self.G_) > 0 else 0.15
        )
        iterations = kwargs.get("iterations", 30)
        node_size = kwargs.get("node_size", 50 if len(self.G_) < 500 else 20)
        alpha_nodes = kwargs.get("alpha_nodes", 0.8)
        alpha_edges = kwargs.get("alpha_edges", 0.3)
        edge_width = kwargs.get("edge_width", 0.6)

        # For Louvain, it can be insightful to draw with a layout that respects communities if possible,
        # but a general spring layout is a good start.
        try:
            pos = nx.spring_layout(
                self.G_,
                k=pos_k,
                iterations=iterations,
                weight="weight",
                seed=self.random_state,
            )
        except Exception as e:
            print(
                f"Spring layout failed ({e}), attempting random layout for {output_file_path.name}."
            )
            pos = nx.random_layout(self.G_, seed=self.random_state)

        nx.draw_networkx_nodes(
            self.G_, pos, node_color=node_colors, node_size=node_size, alpha=alpha_nodes
        )
        nx.draw_networkx_edges(self.G_, pos, alpha=alpha_edges, width=edge_width)

        # Create a legend (optional, can be crowded for many communities)
        # For simplicity, not adding a legend here as number of Louvain communities can be large.
        # If needed, a representative sample or dominant communities could be shown.

        title = kwargs.get(
            "title",
            f"Network with Louvain Communities ({output_file_path.stem})\nModularity: {self.modularity_:.4f}",
        )
        plt.title(title, fontsize=14)
        plt.axis("off")

        ensure_dir(output_file_path.parent)
        plt.savefig(output_file_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Louvain community visualization saved to {output_file_path}")


if __name__ == "__main__":
    if community_louvain is None:
        print(
            "Louvain community detection cannot be tested as `community` package is not installed."
        )
    else:
        project_root = get_project_root()
        print(
            f"Running LouvainCommunityDetector example from project root: {project_root}"
        )

        data_dir = project_root / "data"
        dummy_networks_dir = data_dir / "networks"
        output_louvain_dir = data_dir / "processed" / "louvain_clustering"
        ensure_dir(output_louvain_dir)

        dummy_network_file = dummy_networks_dir / "dummy_network_2024_louvain.csv"
        ensure_dir(dummy_networks_dir)

        if not dummy_network_file.exists():
            print(
                f"Creating dummy weighted network file for Louvain testing: {dummy_network_file}"
            )
            # More complex graph for better community structure
            network_content = """node_1,node_2,weighted,unweighted
subA,subB,5,1
subA,subC,4,1
subB,subC,6,1
subD,subE,5,1
subD,subF,4,1
subE,subF,6,1
subG,subH,5,1
subG,subI,4,1
subH,subI,6,1
subA,subD,1,1 # Weak link between community 1 and 2
subC,subE,0.5,1 # Another weak link
subF,subG,1,1 # Weak link between community 2 and 3
subB,subH,0.2,1 # Very weak link
subX,subY,10,1 # Isolated community
subY,subZ,12,1
subX,subZ,11,1
"""
            with open(dummy_network_file, "w") as f:
                f.write(network_content)

        year_tag = "2024_dummy_louvain_test"
        print(
            f"\n--- Testing LouvainCommunityDetector for year {year_tag} (dummy data) ---"
        )

        # Louvain uses WEIGHTED networks
        print(f"Loading weighted network: {dummy_network_file}")
        graph = load_network(dummy_network_file, weighted=True)
        print(f"Graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges.")

        if not graph.nodes():
            print("Loaded graph is empty. Louvain test cannot proceed.")
        else:
            current_year_output_dir = output_louvain_dir / year_tag
            ensure_dir(current_year_output_dir)

            detector = LouvainCommunityDetector(random_state=42)

            print("\nDetecting communities with Louvain...")
            detector.detect_communities(
                graph
            )  # Default weight_key='weight' is used by load_network

            partition = detector.get_partition()
            modularity = detector.get_modularity()

            if partition and modularity is not None:
                print(
                    f"\nDetected partition (first 10 nodes): {dict(list(partition.items())[:10])}"
                )
                print(f"Modularity: {modularity:.4f}")

                output_csv = (
                    current_year_output_dir / f"louvain_communities_{year_tag}.csv"
                )
                detector.save_results(output_csv)

                output_png = (
                    current_year_output_dir / f"louvain_visualization_{year_tag}.png"
                )
                detector.visualize_communities(output_png, k=0.3, node_size=70)
            else:
                print(
                    "Louvain detection did not yield a valid partition or modularity."
                )

        print("\nLouvainCommunityDetector example test complete.")
