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

# Try to import mpld3 for interactive visualizations
try:
    import mpld3
    from mpld3 import plugins

    MPLD3_AVAILABLE = True
    print("mpld3 is available. Interactive HTML visualizations will be generated.")
except ImportError:
    MPLD3_AVAILABLE = False
    print("Warning: `mpld3` package not found. Visualizations will be static PNGs.")
    print("To enable interactive HTML visualizations, run: pip install mpld3")


class LouvainCommunityDetector:
    """
    Detects communities in a graph using the Louvain algorithm.
    This method is well-suited for weighted networks and optimizes modularity.
    """

    def __init__(self, random_state: int | None = None, num_runs: int = 1):
        """
        Initializes the LouvainCommunityDetector.

        Args:
            random_state (int | None): Seed for reproducibility of the Louvain algorithm.
                                     If num_runs > 1 and random_state is an int, this serves as the base seed.
                                     If num_runs > 1 and random_state is None, each run is non-deterministic.
            num_runs (int): Number of times to run the Louvain algorithm to find the best partition.
                            Defaults to 1 (single run).
        """
        if community_louvain is None:
            raise ImportError("The 'community' package (for Louvain) is not installed.")
        self.random_state = random_state
        self.num_runs = num_runs
        self.partition_ = None  # Stores the best partition found {node: community_id}
        self.communities_ = (
            None  # Stores communities as a list of sets {community_id: {nodes}}
        )
        self.modularity_ = None
        self.G_ = None  # Store the graph for visualization or other purposes
        self.best_random_state_ = (
            None  # Store the random state that yielded the best modularity
        )

    def detect_communities(self, G: nx.Graph, weight_key: str = "weight"):
        """
        Detects communities in the graph using the Louvain algorithm.
        If num_runs > 1, it runs the algorithm multiple times and selects the partition
        with the highest modularity.

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
            self.best_random_state_ = self.random_state
            return

        self.G_ = G

        best_partition = None
        best_modularity = -float("inf")  # Initialize with a very small number
        current_best_random_state = self.random_state

        if self.num_runs <= 0:
            print("Warning: num_runs must be positive. Defaulting to 1 run.")
            self.num_runs = 1

        for i in range(self.num_runs):
            current_seed = None
            if self.random_state is not None:
                current_seed = self.random_state + i
            # If self.random_state is None, current_seed remains None, leading to non-deterministic runs by the library

            # The community_louvain.best_partition function directly uses the 'weight' attribute if present.
            partition = community_louvain.best_partition(
                G, weight=weight_key, random_state=current_seed
            )

            if not partition:
                if self.num_runs > 1:
                    print(
                        f"Warning: Louvain run {i+1}/{self.num_runs} (seed: {current_seed}) did not return a partition. Skipping this run."
                    )
                    continue
                else:
                    print(
                        f"Warning: Louvain algorithm (seed: {current_seed}) did not return a partition."
                    )
                    self.communities_ = defaultdict(set)
                    self.modularity_ = None
                    self.partition_ = None
                    self.best_random_state_ = current_seed
                    return

            modularity = community_louvain.modularity(partition, G, weight=weight_key)

            if modularity > best_modularity:
                best_modularity = modularity
                best_partition = partition
                current_best_random_state = current_seed

            if self.num_runs > 1:
                print(
                    f"  Run {i+1}/{self.num_runs} (seed: {current_seed}), Modularity: {modularity:.4f}"
                )

        if best_partition is None:
            print("Warning: All Louvain runs failed to produce a partition.")
            self.partition_ = {}
            self.communities_ = defaultdict(set)
            self.modularity_ = None
            self.best_random_state_ = self.random_state  # Fallback
            return

        self.partition_ = best_partition
        self.modularity_ = best_modularity
        self.best_random_state_ = current_best_random_state

        # Organize communities into a more usable format: list of sets of nodes
        self.communities_ = defaultdict(set)
        for node, community_id in self.partition_.items():
            self.communities_[community_id].add(node)

        num_detected_communities = len(self.communities_)
        if self.num_runs > 1:
            print(
                f"Louvain algorithm (best of {self.num_runs} runs, best seed: {self.best_random_state_}) detected {num_detected_communities} communities with modularity: {self.modularity_:.4f}"
            )
        else:
            print(
                f"Louvain algorithm (seed: {self.best_random_state_}) detected {num_detected_communities} communities with modularity: {self.modularity_:.4f}"
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
        If mpld3 is available, saves an interactive HTML file. Otherwise, saves a static PNG.

        Args:
            output_file_path (Path): The path to save the visualization.
                                     Extension will be changed to .html if mpld3 is used.
            **kwargs: Additional arguments for layout and drawing functions.
                      Includes 'figsize', 'k', 'iterations', 'node_size', 'alpha_nodes',
                      'alpha_edges', 'edge_width', 'title'.
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

        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (15, 12)))

        community_ids = sorted(list(set(self.partition_.values())))
        palette = sns.color_palette("husl", len(community_ids) if community_ids else 1)
        community_color_map = {
            community_id: palette[i] for i, community_id in enumerate(community_ids)
        }

        # Handle cases where partition might be empty or G_ might be empty leading to no nodes in partition
        node_colors = []
        valid_nodes_for_drawing = [
            node for node in self.G_.nodes() if node in self.partition_
        ]
        if not valid_nodes_for_drawing:
            print(
                f"Warning: No nodes from graph found in partition. Cannot draw nodes for {output_file_path.name}"
            )
        else:
            node_colors = [
                community_color_map[self.partition_[node]]
                for node in valid_nodes_for_drawing
            ]

        pos_k = (
            kwargs.get("k", 0.15 / np.sqrt(len(self.G_))) if len(self.G_) > 0 else 0.15
        )
        iterations = kwargs.get("iterations", 30)
        node_size = kwargs.get("node_size", 50 if len(self.G_) < 500 else 20)
        alpha_nodes = kwargs.get("alpha_nodes", 0.8)
        alpha_edges = kwargs.get("alpha_edges", 0.3)
        edge_width = kwargs.get("edge_width", 0.6)
        layout_seed = self.best_random_state_

        try:
            pos = nx.spring_layout(
                self.G_,
                k=pos_k,
                iterations=iterations,
                weight="weight",
                seed=layout_seed,
            )
        except Exception as e:
            print(
                f"Spring layout failed ({e}), attempting random layout for {output_file_path.name}."
            )
            pos = nx.random_layout(self.G_, seed=layout_seed)

        # Draw nodes, ensuring we only draw nodes that have positions and are in the partition
        # Use nodelist to ensure consistency for labels if mpld3 is used
        drawn_nodes = nx.draw_networkx_nodes(
            self.G_,
            pos,
            ax=ax,
            nodelist=valid_nodes_for_drawing,
            node_color=node_colors,
            node_size=node_size,
            alpha=alpha_nodes,
        )
        nx.draw_networkx_edges(self.G_, pos, ax=ax, alpha=alpha_edges, width=edge_width)

        modularity_str = (
            f"{self.modularity_:.4f}" if self.modularity_ is not None else "N/A"
        )
        title_text = kwargs.get(
            "title",
            f"Network with Louvain Communities ({output_file_path.stem})\nModularity: {modularity_str}",
        )
        ax.set_title(title_text, fontsize=14)
        ax.axis("off")
        ensure_dir(output_file_path.parent)

        if MPLD3_AVAILABLE and drawn_nodes is not None:
            html_output_file_path = output_file_path.with_suffix(".html")
            # Create labels for the tooltip plugin
            # Ensure labels correspond to the order of nodes in valid_nodes_for_drawing
            node_labels = [str(node) for node in valid_nodes_for_drawing]

            tooltip = plugins.PointLabelTooltip(drawn_nodes, labels=node_labels)
            plugins.connect(fig, tooltip)

            try:
                mpld3.save_html(fig, str(html_output_file_path))
                print(
                    f"Louvain community interactive HTML visualization saved to {html_output_file_path}"
                )
            except Exception as e:
                print(f"Failed to save interactive HTML with mpld3: {e}")
                print("Falling back to static PNG.")
                plt.savefig(
                    output_file_path.with_suffix(".png"), dpi=300, bbox_inches="tight"
                )
                print(
                    f"Louvain community static PNG visualization saved to {output_file_path.with_suffix('.png')}"
                )
            plt.close(fig)  # Close the figure after saving with mpld3 or static
        else:
            if not MPLD3_AVAILABLE:
                print(
                    f"mpld3 not available. Saving static PNG to {output_file_path.with_suffix('.png')}"
                )
            elif drawn_nodes is None:
                print(
                    f"No nodes were drawn. Skipping visualization saving for {output_file_path.name}"
                )

            plt.savefig(
                output_file_path.with_suffix(".png"), dpi=300, bbox_inches="tight"
            )
            print(
                f"Louvain community static PNG visualization saved to {output_file_path.with_suffix('.png')}"
            )
            plt.close(fig)


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
        output_louvain_dir = (
            data_dir / "processed" / "louvain_clustering_test_output"
        )  # Changed output dir for clarity
        ensure_dir(output_louvain_dir)

        dummy_network_file = dummy_networks_dir / "dummy_network_2024_louvain.csv"
        ensure_dir(dummy_networks_dir)

        if not dummy_network_file.exists():
            print(
                f"Creating dummy weighted network file for Louvain testing: {dummy_network_file}"
            )
            # More complex graph for better community structure
            # Increased weights within communities, distinct community links
            network_content = """node_1,node_2,weighted,unweighted
subA,subB,10,1
subA,subC,8,1
subB,subC,12,1
subA,subX,0.1,1 # Very weak link to another potential group
subD,subE,10,1
subD,subF,8,1
subE,subF,12,1
subG,subH,10,1
subG,subI,8,1
subH,subI,12,1
subA,subD,1,1 # Weak link between community 1 (ABC) and 2 (DEF)
subC,subE,0.5,1 # Another weak link
subF,subG,1,1 # Weak link between community 2 (DEF) and 3 (GHI)
subB,subH,0.2,1 # Very weak link
subX,subY,10,1 # Potentially isolated community (XYZ)
subY,subZ,12,1
subX,subZ,11,1
subP,subQ,5,1 # Another small community
"""
            with open(dummy_network_file, "w") as f:
                f.write(network_content)

        year_tag = "2024_dummy_louvain_test"
        print(
            f"\n--- Testing LouvainCommunityDetector for year {year_tag} (dummy data) ---"
        )

        # Louvain uses WEIGHTED networks
        print(f"Loading weighted network: {dummy_network_file}")
        graph = load_network(
            dummy_network_file, weighted=True
        )  # Ensure this loads 'weight' attribute
        if graph:
            print(f"Graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges.")
            # Check if weights are loaded:
            # for u, v, data in graph.edges(data=True):
            #     if 'weight' not in data:
            #         print(f"Warning: Edge ({u}, {v}) missing 'weight' attribute.")
            #     break # Just check one

        if not graph or not graph.nodes():  # Check if graph is None or empty
            print(
                "Loaded graph is empty or failed to load. Louvain test cannot proceed."
            )
        else:
            current_year_output_dir = output_louvain_dir / year_tag
            ensure_dir(current_year_output_dir)

            # Test with a single run (default)
            print("\n--- Testing with num_runs = 1 (random_state=42) ---")
            detector_single_run = LouvainCommunityDetector(random_state=42, num_runs=1)
            detector_single_run.detect_communities(graph)
            if detector_single_run.get_modularity() is not None:
                print(
                    f"Single run (seed 42) modularity: {detector_single_run.get_modularity():.4f}"
                )
                detector_single_run.save_results(
                    current_year_output_dir / f"louvain_single_run_{year_tag}.csv"
                )
                viz_path_single = (
                    current_year_output_dir / f"louvain_viz_single_run_{year_tag}.png"
                )  # Base name, ext changes
                detector_single_run.visualize_communities(viz_path_single)

            # Test with multiple runs
            print("\n--- Testing with num_runs = 5 (base_random_state=42) ---")
            detector_multi_run = LouvainCommunityDetector(random_state=42, num_runs=5)
            detector_multi_run.detect_communities(graph)

            partition = detector_multi_run.get_partition()  # Renamed from detector
            modularity = detector_multi_run.get_modularity()  # Renamed from detector

            if partition and modularity is not None:
                print(f"\nBest of 5 runs (base seed 42) modularity: {modularity:.4f}")
                print(
                    f"Seed that produced best modularity: {detector_multi_run.best_random_state_}"
                )

                detector_multi_run.save_results(
                    current_year_output_dir
                    / f"louvain_communities_multi_run_{year_tag}.csv"
                )
                viz_path_multi = (
                    current_year_output_dir / f"louvain_viz_multi_run_{year_tag}.png"
                )  # Base name, ext changes
                detector_multi_run.visualize_communities(
                    viz_path_multi, k=0.3, node_size=70
                )  # Renamed from detector
            else:
                print(
                    "Louvain detection (multi-run) did not yield a valid partition or modularity."
                )

        print("\nLouvainCommunityDetector example test complete.")
