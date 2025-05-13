import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

from utils import load_network, ensure_dir, get_project_root

try:
    import community as community_louvain  # For modularity calculation
except ImportError:
    community_louvain = None
    print(
        "Warning: `community` package not found. Modularity calculation might be affected if not using NetworkX's internal modularity."
    )


class SpectralClusteringDetector:
    """
    Detects communities using Spectral Clustering.
    This implementation includes eigengap heuristic for selecting k,
    and uses k-Means on the eigenvectors of the graph Laplacian.
    It is intended for use with unweighted networks as per the plan.
    """

    def __init__(
        self,
        max_clusters: int = 10,
        laplacian_type: str = "normalized",
        random_state: int | None = None,
    ):
        """
        Initializes the SpectralClusteringDetector.

        Args:
            max_clusters (int): Maximum number of clusters to consider for eigengap heuristic.
            laplacian_type (str): Type of Laplacian to use: 'unnormalized' or 'normalized'.
                                  Lecture 9 suggests normalized Laplacian for conductance optimization.
            random_state (int | None): Seed for reproducibility in k-Means.
        """
        self.max_clusters = max_clusters
        if laplacian_type not in ["unnormalized", "normalized"]:
            raise ValueError("laplacian_type must be 'unnormalized' or 'normalized'.")
        self.laplacian_type = laplacian_type
        self.random_state = random_state

        self.G_ = None
        self.nodes_ = None
        self.node_to_idx_ = None
        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.optimal_k_ = None
        self.partition_ = None  # {node: community_id}
        self.communities_ = None  # {community_id: {nodes}}
        self.modularity_ = None

    def _calculate_laplacian(self, G: nx.Graph):
        """Calculates the specified type of graph Laplacian and its eigen-decomposition."""
        if not G.nodes():
            self.eigenvalues_ = np.array([])
            self.eigenvectors_ = np.array([])
            return

        # Ensure node ordering for matrix operations
        self.nodes_ = list(G.nodes())
        self.node_to_idx_ = {node: idx for idx, node in enumerate(self.nodes_)}

        # Handle disconnected graphs by working with the largest connected component
        # if G is not nx.is_connected(G):
        #     print("Graph is not connected. Spectral clustering will be performed on the largest connected component.")
        #     largest_cc = max(nx.connected_components(G), key=len)
        #     G_comp = G.subgraph(largest_cc).copy()
        #     self.nodes_ = list(G_comp.nodes()) # Update nodes based on the component
        #     self.node_to_idx_ = {node: idx for idx, node in enumerate(self.nodes_)}
        # else:
        #     G_comp = G
        # Using the full graph as per typical spectral clustering, even if disconnected.
        # NetworkX handles this; eigenvalues related to connected components will be zero.
        G_comp = G

        if self.laplacian_type == "normalized":
            # Symmetric normalized Laplacian: L_sym = D^(-1/2) * L * D^(-1/2) = I - D^(-1/2) * A * D^(-1/2)
            # Eigenvalues are between 0 and 2.
            L = nx.normalized_laplacian_matrix(G_comp, nodelist=self.nodes_)
        else:  # unnormalized
            # L = D - A
            L = nx.laplacian_matrix(G_comp, nodelist=self.nodes_)

        # Eigen-decomposition. eigvals are sorted by default by numpy.linalg.eigvalsh
        # For sparse matrices, scipy.sparse.linalg.eigsh is preferred for large graphs.
        # NetworkX returns a SciPy sparse matrix, so convert to dense for numpy or use sparse solvers.
        try:
            # Calculate a limited number of eigenvalues/vectors for efficiency if graph is large
            # k_eigs should be at least max_clusters + 1 (or more for stability)
            num_nodes = len(self.nodes_)
            k_eigs = min(
                num_nodes - 1 if num_nodes > 1 else 1, self.max_clusters + 5
            )  # Request a few more for stability

            if num_nodes == 0:
                self.eigenvalues_ = np.array([])
                self.eigenvectors_ = np.array([])
                return
            if k_eigs <= 0:
                k_eigs = 1  # must be at least 1

            if hasattr(L, "todense"):
                L_dense = L.todense()
            else:
                L_dense = L  # if it was already dense (e.g. small test graph)

            # For very small graphs, eigvalsh might be more direct
            try:
                if num_nodes < k_eigs * 2:  # Heuristic threshold
                    eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
                else:
                    # This part is tricky with eigsh for smallest. Scikit-learn's spectral_embedding handles it better.
                    # Using dense for now as per lecture simplicity, but aware of scalability.
                    eigenvalues, eigenvectors = np.linalg.eigh(L_dense)

                # Sort eigenvalues and corresponding eigenvectors
                sorted_indices = np.argsort(eigenvalues)
                self.eigenvalues_ = eigenvalues[sorted_indices]
                self.eigenvectors_ = eigenvectors[:, sorted_indices]
            except Exception as e:
                print(
                    f"Eigen-decomposition failed: {e}. This can happen with very small or unusual graphs."
                )
                self.eigenvalues_ = np.array([])
                self.eigenvectors_ = np.array([])
                return
        except Exception as e:
            print(
                f"Eigen-decomposition failed: {e}. This can happen with very small or unusual graphs."
            )
            self.eigenvalues_ = np.array([])
            self.eigenvectors_ = np.array([])
            return

    def _find_optimal_k_eigengap(self, max_k_to_check: int | None = None) -> int:
        """Determines the optimal number of clusters (k) using the eigengap heuristic."""
        if self.eigenvalues_ is None or len(self.eigenvalues_) < 2:
            print(
                "Not enough eigenvalues to determine k using eigengap. Defaulting to k=2 (if possible) or 1."
            )
            return min(2, len(self.eigenvalues_)) if len(self.eigenvalues_) > 0 else 1

        max_k = min(self.max_clusters, len(self.eigenvalues_) - 1)
        if max_k_to_check is not None:
            max_k = min(max_k, max_k_to_check, len(self.eigenvalues_) - 1)

        if max_k < 1:  # Need at least 1 eigenvalue to check for k=1, or 2 for k > 1
            print(
                "Cannot determine k via eigengap with current eigenvalues. Defaulting to k=1."
            )
            return 1

        gaps = np.diff(
            self.eigenvalues_[: max_k + 1]
        )  # Gaps between lambda_i and lambda_{i+1}

        if not gaps.size:  # Only one eigenvalue (or less)
            print("No gaps to analyze. Defaulting to k=1.")
            return 1

        # The optimal k is often where the gap is largest.
        # E.g., if max gap is between lambda_k and lambda_{k+1}, then choose k clusters.
        # So, if gaps[i] is max, then k = i+1 (since gaps are 0-indexed, eigenvalues 0-indexed)
        # We are looking for k components, so index of largest gap + 1.
        optimal_k = np.argmax(gaps[:max_k]) + 1
        # Add 1 because argmax returns 0-indexed position in `gaps` array.
        # If eigenvalues are lambda_0, lambda_1, lambda_2, ..., lambda_m
        # gaps = [lambda_1-lambda_0, lambda_2-lambda_1, ...]
        # If largest gap is gaps[i] = lambda_{i+1} - lambda_i, then k = i+1.

        # A common convention: if all eigenvalues are very close (small gaps), it might mean k=1
        # For simplicity, we directly use the largest gap.
        print(
            f"Eigengap heuristic: Optimal k = {optimal_k} (checked up to {max_k} clusters)."
        )
        self.optimal_k_ = optimal_k
        return optimal_k

    def detect_communities(self, G: nx.Graph, k: int | None = None):
        """
        Detects communities using spectral clustering.

        Args:
            G (nx.Graph): The input graph (unweighted is typical for this setup).
            k (int | None): The number of clusters. If None, uses eigengap heuristic.
        """
        if not G.nodes():
            print("Warning: Input graph is empty. No communities to detect.")
            self.partition_ = {}
            self.communities_ = defaultdict(set)
            self.modularity_ = None
            self.G_ = G
            self.optimal_k_ = 0
            return

        self.G_ = G
        self._calculate_laplacian(G)

        if self.eigenvalues_ is None or len(self.eigenvalues_) == 0:
            print(
                "Eigen-decomposition failed or yielded no eigenvalues. Cannot proceed."
            )
            # Set to empty/default states
            self.partition_ = {node: 0 for node in self.nodes_} if self.nodes_ else {}
            self.communities_ = defaultdict(set)
            if self.nodes_:
                self.communities_[0] = set(self.nodes_)
            self.modularity_ = None
            self.optimal_k_ = 0 if not self.nodes_ else 1
            return

        if k is None:
            self.optimal_k_ = self._find_optimal_k_eigengap()
            if (
                self.optimal_k_ == 0 and len(self.nodes_) > 0
            ):  # Should not happen if nodes exist
                print(
                    "Warning: Eigengap resulted in k=0, defaulting to k=1 for non-empty graph."
                )
                self.optimal_k_ = 1
            elif self.optimal_k_ == 0 and len(self.nodes_) == 0:
                # This is already handled by graph empty check, but as safeguard
                return
        else:
            self.optimal_k_ = k

        if (
            self.optimal_k_ <= 0
        ):  # If k was specified as 0 or less, or eigengap failed badly
            print(
                f"Warning: Number of clusters k is {self.optimal_k_}. Cannot perform clustering. Defaulting all nodes to one community."
            )
            self.partition_ = {node: 0 for node in self.nodes_}
            self.communities_ = defaultdict(set)
            self.communities_[0] = set(self.nodes_)
            self.modularity_ = None  # Or 0, depends on definition for single cluster
            return

        # Ensure k is not more than the number of available eigenvectors/nodes
        if self.optimal_k_ > len(self.eigenvectors_[0]):
            print(
                f"Warning: Optimal k ({self.optimal_k_}) is greater than number of available eigenvectors/nodes ({len(self.eigenvectors_[0])}). Clamping k."
            )
            self.optimal_k_ = len(self.eigenvectors_[0])
            if self.optimal_k_ == 0 and len(self.nodes_) > 0:
                self.optimal_k_ = 1  # Ensure at least one cluster for non-empty graph
            elif self.optimal_k_ == 0 and len(self.nodes_) == 0:
                return  # Already handled

        # Select the first k eigenvectors (corresponding to the k smallest eigenvalues)
        # For normalized Laplacian, these are the ones to use.
        # For unnormalized, also the smallest k (Fiedler vector is 2nd smallest, corresponding to k=2)
        embedding_matrix = self.eigenvectors_[:, : self.optimal_k_]

        # Normalize rows of the embedding matrix before k-Means (often improves k-Means performance)
        # Some spectral clustering versions skip this, others include it.
        # Scikit-learn's SpectralClustering does this internally if using 'kmeans' assign_labels.
        embedding_matrix_normalized = normalize(embedding_matrix, norm="l2", axis=1)

        # Perform k-Means on the embedded data
        # Check if optimal_k_ makes sense (e.g. not more than samples)
        if self.optimal_k_ > embedding_matrix_normalized.shape[0]:
            print(
                f"Warning: k ({self.optimal_k_}) > number of samples ({embedding_matrix_normalized.shape[0]}). Adjusting k."
            )
            self.optimal_k_ = embedding_matrix_normalized.shape[0]
            if self.optimal_k_ == 0 and len(self.nodes_) > 0:
                self.optimal_k_ = 1
            elif self.optimal_k_ == 0 and len(self.nodes_) == 0:
                return

        if self.optimal_k_ == 0:  # Final check if k became 0
            print("Final k is 0, assigning all to one community.")
            self.partition_ = {node: 0 for node in self.nodes_}
            self.communities_ = defaultdict(set)
            self.communities_[0] = set(self.nodes_)
            self.modularity_ = None
            return

        kmeans = KMeans(
            n_clusters=self.optimal_k_, random_state=self.random_state, n_init="auto"
        )
        try:
            cluster_labels = kmeans.fit_predict(embedding_matrix_normalized)
        except ValueError as e:
            # This can happen if k_means receives an empty array or k is too large for the data points
            print(
                f"KMeans clustering failed: {e}. Assigning all nodes to one community."
            )
            self.partition_ = {node: 0 for node in self.nodes_}
            self.communities_ = defaultdict(set)
            self.communities_[0] = set(self.nodes_)
            self.modularity_ = None
            return

        self.partition_ = {
            node: label for node, label in zip(self.nodes_, cluster_labels)
        }

        self.communities_ = defaultdict(set)
        for node, community_id in self.partition_.items():
            self.communities_[community_id].add(node)

        # Calculate modularity
        if community_louvain and G.edges():  # Modularity requires edges
            self.modularity_ = community_louvain.modularity(
                self.partition_, G, weight="weight"
            )  # Use weight if present for fair comparison
            print(
                f"Spectral clustering (k={self.optimal_k_}) detected {len(self.communities_)} communities with modularity: {self.modularity_:.4f}"
            )
        elif G.edges():  # Use NetworkX modularity if python-louvain not available
            # nx.community.modularity requires communities as list of frozensets/sets
            community_list_of_sets = [
                frozenset(self.communities_[cid])
                for cid in sorted(self.communities_.keys())
            ]
            if community_list_of_sets:  # Ensure it's not empty
                self.modularity_ = nx.community.modularity(
                    G, community_list_of_sets, weight="weight"
                )
                print(
                    f"Spectral clustering (k={self.optimal_k_}) detected {len(self.communities_)} communities with modularity (NX): {self.modularity_:.4f}"
                )
            else:
                self.modularity_ = None
                print(
                    f"Spectral clustering (k={self.optimal_k_}) detected {len(self.communities_)} communities. Modularity not computed (no communities/edges)."
                )
        else:
            self.modularity_ = None  # No edges, modularity is ill-defined or 0
            print(
                f"Spectral clustering (k={self.optimal_k_}) detected {len(self.communities_)} communities. Modularity not computed (no edges)."
            )

    def get_partition(self) -> dict | None:
        return self.partition_

    def get_communities(self) -> list[set]:
        if not self.communities_:
            return []
        return list(self.communities_.values())

    def get_community_map(self) -> dict[int, set]:
        return self.communities_

    def get_modularity(self) -> float | None:
        return self.modularity_

    def get_optimal_k(self) -> int | None:
        return self.optimal_k_

    def get_eigenvalues(self) -> np.ndarray | None:
        return self.eigenvalues_

    def save_results(self, output_file_path: Path):
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
        print(f"Spectral clustering assignments saved to {output_file_path}")

    def plot_eigengap(self, output_file_path: Path):
        if self.eigenvalues_ is None or len(self.eigenvalues_) < 2:
            print(
                f"Not enough eigenvalues to plot eigengap for {output_file_path.name}."
            )
            return

        plt.figure(figsize=(10, 6))
        # Plot the first few eigenvalues (e.g., up to max_clusters + some buffer)
        max_eig_to_plot = min(len(self.eigenvalues_), self.max_clusters + 5)
        eigenvalues_to_plot = self.eigenvalues_[:max_eig_to_plot]

        plt.plot(
            range(1, len(eigenvalues_to_plot) + 1),
            eigenvalues_to_plot,
            marker="o",
            linestyle="-",
        )
        plt.xlabel("Eigenvalue Index (Sorted)")
        plt.ylabel("Eigenvalue")
        plt.title(
            f"Eigengap Heuristic ({output_file_path.stem})\nLaplacian: {self.laplacian_type}, Optimal k found: {self.optimal_k_}"
        )
        plt.xticks(range(1, len(eigenvalues_to_plot) + 1))
        if (
            self.optimal_k_ is not None
            and self.optimal_k_ > 0
            and self.optimal_k_ < len(eigenvalues_to_plot)
        ):
            # Highlight the gap after the (optimal_k-1)-th eigenvalue (which leads to optimal_k clusters)
            # The gap is (lambda_k - lambda_{k-1}). We use optimal_k clusters.
            # The largest gap is typically after the (k-1)th smallest non-trivial eigenvalue when choosing k clusters.
            # The lecture says lambda_k+1 - lambda_k. If we choose k based on this, it means we select k eigenvectors.
            # The k in plot should be the number of clusters.
            plt.axvline(
                x=self.optimal_k_,
                color="r",
                linestyle="--",
                label=f"Optimal k = {self.optimal_k_}",
            )
            # The gap occurs between eigenvalue k and k+1 (1-indexed). So we draw line at k.
            # If eigenvalues are $\lambda_1, \lambda_2, ..., \lambda_N$. optimal_k is index of gap +1.
            # plt.axvline(x=self.optimal_k_ + 0.5, color='r', linestyle='--', label=f'Eigengap after $\lambda_{{{self.optimal_k_}}}$')
        plt.legend()
        plt.grid(True)
        ensure_dir(output_file_path.parent)
        plt.savefig(output_file_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Eigengap plot saved to {output_file_path}")

    def visualize_communities(self, output_file_path: Path, **kwargs):
        if (
            self.G_ is None
            or not self.G_.nodes()
            or self.partition_ is None
            or not self.partition_
        ):
            print(
                f"Cannot visualize: Graph or partition not available. Skipping {output_file_path.name}"
            )
            return

        plt.figure(figsize=kwargs.get("figsize", (15, 12)))
        community_ids = sorted(list(set(self.partition_.values())))
        palette = sns.color_palette("husl", len(community_ids))
        community_color_map = {cid: palette[i] for i, cid in enumerate(community_ids)}
        node_colors = [
            community_color_map.get(self.partition_[node], "grey")
            for node in self.G_.nodes()
        ]

        pos_k = (
            kwargs.get("k", 0.15 / np.sqrt(len(self.G_))) if len(self.G_) > 0 else 0.15
        )
        iterations = kwargs.get("iterations", 30)
        node_size = kwargs.get("node_size", 50 if len(self.G_) < 500 else 20)

        try:
            # Unweighted graphs for spectral clustering, so don't pass weight to layout typically
            pos = nx.spring_layout(
                self.G_, k=pos_k, iterations=iterations, seed=self.random_state
            )
        except Exception as e:
            print(
                f"Spring layout failed ({e}), attempting random layout for {output_file_path.name}."
            )
            pos = nx.random_layout(self.G_, seed=self.random_state)

        nx.draw_networkx_nodes(
            self.G_, pos, node_color=node_colors, node_size=node_size, alpha=0.8
        )
        nx.draw_networkx_edges(self.G_, pos, alpha=0.3, width=0.6)

        mod_text = (
            f"Modularity: {self.modularity_:.4f}"
            if self.modularity_ is not None
            else "Modularity: N/A"
        )
        title = kwargs.get(
            "title",
            f"Spectral Clustering Communities ({output_file_path.stem})\n$k={self.optimal_k_}$, {mod_text}",
        )
        plt.title(title, fontsize=14)
        plt.axis("off")
        ensure_dir(output_file_path.parent)
        plt.savefig(output_file_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Spectral clustering visualization saved to {output_file_path}")


if __name__ == "__main__":
    project_root = get_project_root()
    print(
        f"Running SpectralClusteringDetector example from project root: {project_root}"
    )

    data_dir = project_root / "data"
    dummy_networks_dir = data_dir / "networks"
    output_spectral_dir = data_dir / "processed" / "spectral_clustering"
    ensure_dir(output_spectral_dir)

    dummy_network_file = dummy_networks_dir / "dummy_network_2024_spectral.csv"
    ensure_dir(dummy_networks_dir)

    if not dummy_network_file.exists():
        print(
            f"Creating dummy unweighted network file for Spectral testing: {dummy_network_file}"
        )
        # A graph that should ideally have 2-3 clear clusters for spectral methods
        network_content = """node_1,node_2,weighted,unweighted
subA,subB,1,1
subA,subC,1,1
subB,subC,1,1
subD,subE,1,1
subD,subF,1,1
subE,subF,1,1
subG,subH,1,1
subG,subI,1,1
subH,subI,1,1
subA,subD,0.1,1 # Bridge node, present in unweighted
subD,subG,0.1,1 # Bridge node, present in unweighted
subX,subY,1,1
subY,subZ,1,1
subX,subZ,1,1
subC,subX,0.05,1 # very weak connection to 4th cluster
"""
        with open(dummy_network_file, "w") as f:
            f.write(network_content)

    year_tag = "2024_dummy_spectral_test"
    print(
        f"\n--- Testing SpectralClusteringDetector for year {year_tag} (dummy data) ---"
    )

    # Spectral Clustering uses UNWEIGHTED networks as per plan
    print(f"Loading unweighted network: {dummy_network_file}")
    graph = load_network(dummy_network_file, weighted=False)
    print(f"Graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges.")

    if not graph.nodes():
        print("Loaded graph is empty. Spectral Clustering test cannot proceed.")
    else:
        current_year_output_dir = output_spectral_dir / year_tag
        ensure_dir(current_year_output_dir)

        detector = SpectralClusteringDetector(
            max_clusters=5, laplacian_type="normalized", random_state=42
        )
        # detector = SpectralClusteringDetector(max_clusters=5, laplacian_type='unnormalized', random_state=42)

        print(
            "\nDetecting communities with Spectral Clustering (auto k via eigengap)..."
        )
        detector.detect_communities(graph)
        # detector.detect_communities(graph, k=3) # To test with fixed k

        partition = detector.get_partition()
        modularity = detector.get_modularity()
        optimal_k = detector.get_optimal_k()

        if partition and optimal_k is not None:
            print(f"\nOptimal k chosen: {optimal_k}")
            print(
                f"Detected partition (first 10 nodes): {dict(list(partition.items())[:10])}"
            )
            if modularity is not None:
                print(f"Modularity: {modularity:.4f}")

            output_csv = (
                current_year_output_dir
                / f"spectral_communities_k{optimal_k}_{year_tag}.csv"
            )
            detector.save_results(output_csv)

            output_eigengap_png = (
                current_year_output_dir
                / f"spectral_eigengap_k{optimal_k}_{year_tag}.png"
            )
            detector.plot_eigengap(output_eigengap_png)

            output_viz_png = (
                current_year_output_dir
                / f"spectral_visualization_k{optimal_k}_{year_tag}.png"
            )
            detector.visualize_communities(output_viz_png)
        else:
            print("Spectral clustering did not yield a valid partition or k.")

    print("\nSpectralClusteringDetector example test complete.")
