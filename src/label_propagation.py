import networkx as nx
import json
import pandas as pd
import numpy as np
from sklearn.semi_supervised import LabelSpreading
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Corrected import assuming utils.py is in the same directory (src)
# For a package structure, this relative import is appropriate.
from utils import load_network, load_subreddit_metadata, ensure_dir, get_project_root


class LabelPropagator:
    """
    Performs label propagation on a network to predict node labels based on a
    set of seed labels.
    """

    def __init__(
        self,
        alpha: float = 0.8,
        max_iter: int = 1000,
        kernel: str = "knn",
        random_state: int | None = None,
    ):
        """
        Initializes the LabelPropagator.

        Args:
            alpha (float): Clamping factor. A higher alpha means more reliance on initial labels.
            max_iter (int): Maximum number of iterations.
            kernel (str): Kernel to use ('knn' or 'rbf'). 'knn' is generally good for graph-based propagation.
            random_state (int | None): Seed for reproducibility.
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.kernel = kernel
        self.random_state = random_state

        # Initialize model based on parameters that are supported by the scikit-learn version
        model_params = {
            "kernel": self.kernel,
            "alpha": self.alpha,
            "max_iter": self.max_iter,
            "n_jobs": -1,  # Use all available cores
        }

        # Only add random_state if it's not None
        if self.random_state is not None:
            # Check scikit-learn version to determine if random_state is supported
            try:
                # Some scikit-learn versions might not support random_state
                self.model = LabelSpreading(
                    random_state=self.random_state, **model_params
                )
            except TypeError:
                print(
                    "Warning: Your scikit-learn version does not support random_state for LabelSpreading."
                )
                self.model = LabelSpreading(**model_params)
        else:
            self.model = LabelSpreading(**model_params)

        self.nodes_ = None
        self.node_to_idx_ = None
        self.label_to_num_ = None
        self.num_to_label_ = None
        self.propagated_labels_numeric_ = None
        self.G_ = None  # To store the graph used for fitting/visualization

    def _prepare_data(self, G: nx.Graph, seed_labels: dict):
        """
        Prepares data for label propagation: creates adjacency matrix,
        initial label array, and mappings.
        """
        self.G_ = G  # Store graph for later use (e.g. visualization)
        self.nodes_ = list(G.nodes())
        if not self.nodes_:
            # This case should ideally be handled before calling fit,
            # but good to have a safeguard.
            print("Warning: Graph has no nodes. Cannot prepare data.")
            return None, None  # Indicate failure to prepare

        self.node_to_idx_ = {node: idx for idx, node in enumerate(self.nodes_)}

        A = nx.adjacency_matrix(G, nodelist=self.nodes_).toarray()
        y_initial = np.full(len(self.nodes_), -1, dtype=int)  # -1 for unlabeled

        unique_string_labels = sorted(
            list(
                set(
                    str_label
                    for str_label in seed_labels.values()
                    if str_label is not None
                )
            )
        )

        if not unique_string_labels:
            print(
                "Warning: No valid string labels found in seed_labels. All nodes will be treated as unlabeled initially."
            )

        self.label_to_num_ = {label: i for i, label in enumerate(unique_string_labels)}
        self.num_to_label_ = {i: label for label, i in self.label_to_num_.items()}

        seeds_in_graph_count = 0
        for node, str_label in seed_labels.items():
            if (
                node in self.node_to_idx_
                and str_label is not None
                and str_label in self.label_to_num_
            ):
                y_initial[self.node_to_idx_[node]] = self.label_to_num_[str_label]
                seeds_in_graph_count += 1

        if unique_string_labels and seeds_in_graph_count == 0:
            print(
                "Warning: None of the provided seed labels correspond to nodes in the graph."
            )
        elif unique_string_labels:
            print(f"Initialized {seeds_in_graph_count} seed labels in the graph.")

        return A, y_initial

    def fit(self, G: nx.Graph, seed_labels: dict):
        """
        Fits the label propagation model to the graph and seed labels.
        """
        if not G.nodes():
            print("Warning: Input graph is empty. Skipping fitting.")
            self.propagated_labels_numeric_ = np.array([], dtype=int)
            self.nodes_ = []
            self.G_ = G  # Store the empty graph
            return

        A, y_initial = self._prepare_data(G, seed_labels)

        if A is None:  # Data preparation failed (e.g. graph had no nodes initially)
            self.propagated_labels_numeric_ = np.array([], dtype=int)
            return

        if np.all(y_initial == -1):
            print(
                "Warning: No initial labels could be assigned to graph nodes. "
                "LabelSpreading will treat all nodes as unlabeled."
            )
            # sklearn's LabelSpreading can handle y with all -1, often resulting in one dominant cluster
            # or behavior dependent on the unlabelled data structure.

        self.model.fit(A, y_initial)
        self.propagated_labels_numeric_ = self.model.transduction_

    def predict(self) -> pd.DataFrame:
        """
        Returns the propagated labels as a DataFrame.
        """
        if (
            self.propagated_labels_numeric_ is None
            or self.nodes_ is None
            or not self.nodes_
        ):
            # This implies fit wasn't called, graph was empty, or preparation failed
            print(
                "Warning: Model not fitted or graph is empty. Returning empty DataFrame."
            )
            return pd.DataFrame(columns=["subreddit", "predicted_label"])

        # Ensure num_to_label_ is available; it might not be if no valid seed labels were given
        if self.num_to_label_ is None:
            # This state implies _prepare_data was called but found no unique_string_labels
            # All propagated_labels_numeric_ should be -1 or what the model defaults to
            # We will map all to "unknown"
            predicted_str_labels = ["unknown"] * len(self.propagated_labels_numeric_)
        else:
            predicted_str_labels = [
                self.num_to_label_.get(label_num, "unknown")
                for label_num in self.propagated_labels_numeric_
            ]

        results_df = pd.DataFrame(
            {"subreddit": self.nodes_, "predicted_label": predicted_str_labels}
        )
        return results_df

    def save_results(self, output_file_path: Path):
        """
        Saves the propagation results to a CSV file.
        """
        results_df = self.predict()
        if not results_df.empty:
            ensure_dir(output_file_path.parent)
            results_df.to_csv(output_file_path, index=False)
            print(f"Label propagation results saved to {output_file_path}")
        else:
            print(
                f"No results to save for {output_file_path.name} (possibly due to empty graph or failed fit)."
            )

    def visualize_results(self, output_file_path: Path, **kwargs):
        """
        Creates and saves a visualization of the network with propagated labels.
        """
        if (
            self.G_ is None
            or not self.G_.nodes()
            or self.propagated_labels_numeric_ is None
        ):
            print(
                f"Cannot visualize: Graph or labels not available. Skipping visualization for {output_file_path.name}"
            )
            return

        plt.figure(figsize=kwargs.get("figsize", (15, 12)))

        unique_numeric_propagated_labels = sorted(
            np.unique(self.propagated_labels_numeric_)
        )

        # Determine string labels and color mapping
        # Handles cases: 1) no seed labels (all unknown), 2) seed labels exist
        str_labels_for_legend = {}
        if not self.num_to_label_:  # Case 1: No seed labels were processed
            palette = sns.color_palette("pastel", 1)
            str_labels_for_legend["unknown"] = palette[0]
            # Map all numeric labels (likely -1 or a single cluster from LabelSpreading) to "unknown"
            color_map_numeric = {
                num_label: palette[0] for num_label in unique_numeric_propagated_labels
            }
        else:  # Case 2: Seed labels were processed
            # Get all unique string labels that appeared in the output
            present_str_labels = sorted(
                list(
                    set(
                        self.num_to_label_.get(l, "unknown")
                        for l in unique_numeric_propagated_labels
                    )
                )
            )
            palette = sns.color_palette("husl", len(present_str_labels))
            str_labels_for_legend = {
                str_label: palette[i] for i, str_label in enumerate(present_str_labels)
            }

            color_map_numeric = {}
            for num_label in unique_numeric_propagated_labels:
                str_l = self.num_to_label_.get(num_label, "unknown")
                color_map_numeric[num_label] = str_labels_for_legend[str_l]

        node_colors = [
            color_map_numeric.get(label_num, str_labels_for_legend.get("unknown"))
            for label_num in self.propagated_labels_numeric_
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

        try:
            pos = nx.spring_layout(
                self.G_, k=pos_k, iterations=iterations, seed=self.random_state
            )
        except (
            Exception
        ) as e:  # Broad exception for layout issues (e.g. disconnected, too few nodes)
            print(
                f"Spring layout failed ({e}), attempting random layout for {output_file_path.name}."
            )
            pos = nx.random_layout(self.G_, seed=self.random_state)

        nx.draw_networkx_nodes(
            self.G_, pos, node_color=node_colors, node_size=node_size, alpha=alpha_nodes
        )
        nx.draw_networkx_edges(self.G_, pos, alpha=alpha_edges, width=edge_width)

        if str_labels_for_legend:
            legend_elements = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    label=str_label,
                    markersize=8,
                )
                for str_label, color in str_labels_for_legend.items()
            ]
            plt.legend(
                handles=legend_elements,
                title="Predicted Labels",
                loc="best",
                fontsize="small",
            )

        title = kwargs.get(
            "title", f"Network with Propagated Labels ({output_file_path.stem})"
        )
        plt.title(title, fontsize=14)
        plt.axis("off")

        ensure_dir(output_file_path.parent)
        plt.savefig(output_file_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Label propagation visualization saved to {output_file_path}")


if __name__ == "__main__":
    project_root = get_project_root()
    print(f"Running LabelPropagator example from project root: {project_root}")

    # Define paths using project_root, consistent with utils.py structure
    data_dir = project_root / "data"
    dummy_networks_dir = data_dir / "networks"
    dummy_metadata_dir = data_dir / "metadata"
    output_lp_dir = data_dir / "processed" / "label_propagation"

    ensure_dir(output_lp_dir)

    # Dummy files (paths defined as in utils.py for consistency)
    dummy_network_file = (
        dummy_networks_dir / "dummy_network_2024.csv"
    )  # Changed name for clarity
    dummy_metadata_file = dummy_metadata_dir / "dummy_subreddits_metadata.jsonl"

    # Create dummy files if they don't exist (similar to utils.py for standalone test)
    ensure_dir(dummy_networks_dir)
    ensure_dir(dummy_metadata_dir)

    if not dummy_network_file.exists():
        print(f"Creating dummy network file for testing: {dummy_network_file}")
        network_content = """node_1,node_2,weighted,unweighted
subA,subB,5,1
subB,subC,10,1
subA,subC,0,0
subC,subD,3,1
subD,subE,2,1
subE,subF,7,1
subX,subY,2,1
subY,subZ,3,1
subP,subQ,1,1
"""
        with open(dummy_network_file, "w") as f:
            f.write(network_content)

    if not dummy_metadata_file.exists():
        print(f"Creating dummy metadata file for testing: {dummy_metadata_file}")
        metadata_content = """{"subreddit": "subA", "party": "dem"}
{"subreddit": "subB", "party": "dem"}
{"subreddit": "subC", "party": "rep"}
{"subreddit": "subF", "party": "dem"}
{"subreddit": "subX", "party": "neutral"}
{"subreddit": "subP", "party": "rep"}
{"subreddit": "nonExistentNode", "party": "dem"}
"""
        with open(dummy_metadata_file, "w") as f:
            f.write(metadata_content)

    print(f"\n--- Testing LabelPropagator for year 2024 (dummy data) ---")
    year_tag = "2024_dummy_lp_test"

    print(f"Loading unweighted network: {dummy_network_file}")
    # Label Propagation uses unweighted networks as per plan
    graph = load_network(dummy_network_file, weighted=False)
    print(f"Graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges.")

    if not graph.nodes():
        print("Loaded graph is empty. LabelPropagator test cannot proceed.")
    else:
        print(f"Loading metadata: {dummy_metadata_file}")
        seed_labels_map = load_subreddit_metadata(dummy_metadata_file, seed_key="party")
        print(f"Seed labels loaded: {len(seed_labels_map)} entries.")

        current_year_output_dir = output_lp_dir / year_tag
        ensure_dir(current_year_output_dir)

        propagator = LabelPropagator(alpha=0.85, max_iter=500, random_state=42)

        print("\nFitting LabelPropagator...")
        propagator.fit(graph, seed_labels_map)

        print("\nPredicting labels...")
        results_df = propagator.predict()

        if not results_df.empty:
            print("\nPropagated Labels (first 5 and last 5):")
            print(pd.concat([results_df.head(), results_df.tail()]))

            output_csv = current_year_output_dir / f"propagated_labels_{year_tag}.csv"
            propagator.save_results(output_csv)

            output_png = (
                current_year_output_dir / f"network_visualization_{year_tag}.png"
            )
            propagator.visualize_results(output_png, k=0.2, node_size=70)
        else:
            print("No prediction results to display or save.")

    print("\nLabelPropagator example test complete.")
