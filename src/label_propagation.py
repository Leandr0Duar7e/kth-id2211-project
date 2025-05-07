import networkx as nx
import json
import pandas as pd
import numpy as np
from sklearn.semi_supervised import LabelSpreading
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob

def load_network(network_file):
    """Load the network from a CSV file with source, target, weighted, and unweighted columns."""
    df = pd.read_csv(network_file)
    
    G = nx.DiGraph()
    
    for _, row in df.iterrows():
        G.add_edge(row.iloc[0], row.iloc[1], 
                  weight=row.iloc[2], 
                  unweighted=row.iloc[3])  
    
    return G

def load_subreddit_metadata(metadata_file):
    """Load subreddit metadata and extract party information from JSONL format."""
    party_labels = {}
    
    with open(metadata_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'party' in data and data['party']:
                    party_labels[data['subreddit']] = data['party']
            except json.JSONDecodeError:
                continue
    
    return party_labels

def prepare_label_propagation(G, seed_labels):
    """Prepare data for label propagation."""
    # Create a mapping of node names to indices
    nodes = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    
    # Create the adjacency matrix
    A = nx.adjacency_matrix(G).toarray()
    
    # Create label array (-1 for unlabeled)
    y = np.full(len(nodes), -1)
    
    # Create a mapping of party strings to numeric values
    unique_parties = sorted(set(seed_labels.values()))
    party_to_num = {party: idx for idx, party in enumerate(unique_parties)}
    
    # Set seed labels with numeric values
    for node, label in seed_labels.items():
        if node in node_to_idx:
            y[node_to_idx[node]] = party_to_num[label]
    
    return A, y, nodes, party_to_num

def perform_label_propagation(A, y, alpha=0.8, max_iter=1000):
    """Perform label propagation using scikit-learn's LabelSpreading."""
    label_prop_model = LabelSpreading(kernel='knn', alpha=alpha, max_iter=max_iter)
    label_prop_model.fit(A, y)
    return label_prop_model.transduction_

def save_results(nodes, labels, party_to_num, output_file):
    """Save the propagation results to a file."""
    # Create reverse mapping for numeric labels to party names
    num_to_party = {v: k for k, v in party_to_num.items()}
    
    # Convert numeric labels back to party names
    party_labels = [num_to_party[label] if label != -1 else "unknown" for label in labels]
    
    results = pd.DataFrame({
        'subreddit': nodes,
        'predicted_party': party_labels
    })
    results.to_csv(output_file, index=False)
    return results

def visualize_results(G, labels, party_to_num, output_file):
    """Create a visualization of the network with party labels."""
    plt.figure(figsize=(12, 8))
    
    # Create reverse mapping for numeric labels to party names
    num_to_party = {v: k for k, v in party_to_num.items()}
    
    # Create a color map for parties
    unique_labels = np.unique(labels)
    colors = sns.color_palette("husl", len(unique_labels))
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    # Create node colors
    node_colors = [color_map[label] for label in labels]
    
    # Draw the network
    pos = nx.spring_layout(G, k=1, iterations=50)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=50, alpha=0.6)
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color, label=num_to_party[label], 
                                 markersize=10) 
                      for label, color in color_map.items()]
    plt.legend(handles=legend_elements, title="Political Parties")
    
    plt.title("Subreddit Network with Propagated Party Labels")
    plt.axis('off')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def process_network(network_file, metadata_file, output_dir, year):
    """Process a single network file and create visualizations."""
    print(f"\nProcessing network for year {year}...")
    
    # Load data
    print("Loading network...")
    G = load_network(network_file)
    
    print("Loading metadata...")
    seed_labels = load_subreddit_metadata(metadata_file)
    
    # Prepare and perform label propagation
    print("Preparing label propagation...")
    A, y, nodes, party_to_num = prepare_label_propagation(G, seed_labels)
    
    print("Performing label propagation...")
    propagated_labels = perform_label_propagation(A, y)
    
    # Save results
    print("Saving results...")
    results = save_results(nodes, propagated_labels, party_to_num, 
                         output_dir / f"propagated_labels_{year}.csv")
    
    # Create visualization
    print("Creating visualization...")
    visualize_results(G, propagated_labels, party_to_num, 
                     output_dir / f"network_visualization_{year}.png")
    
    return results

def main():
    # Define paths
    base_path = Path("data")
    networks_dir = base_path / "networks"
    metadata_file = base_path / "metadata" / "subreddits_metadata.json"
    output_dir = base_path / "processed"
    output_dir.mkdir(exist_ok=True)
    
    # Find all network files
    network_files = sorted(glob.glob(str(networks_dir / "networks_*.csv")))
    
    if not network_files:
        print("No network files found!")
        return
    
    # Process each network file
    all_results = {}
    for network_file in network_files:
        # Extract year from filename
        year = Path(network_file).stem.split('_')[1]
        print(f"\nProcessing year {year}")
        
        # Process the network
        results = process_network(network_file, metadata_file, output_dir, year)
        all_results[year] = results
    
    # Create summary statistics
    print("\nCreating summary statistics...")
    summary = pd.DataFrame()
    
    for year, results in all_results.items():
        # Count subreddits per party
        party_counts = results['predicted_party'].value_counts()
        party_counts.name = year
        summary = pd.concat([summary, party_counts], axis=1)
    
    # Save summary statistics
    summary.to_csv(output_dir / "party_distribution_summary.csv")
    
    # Create summary visualization
    plt.figure(figsize=(12, 6))
    summary.plot(kind='bar', stacked=True)
    plt.title("Distribution of Political Parties Over Years")
    plt.xlabel("Political Party")
    plt.ylabel("Number of Subreddits")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "party_distribution_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nDone! Results saved in:", output_dir)

if __name__ == "__main__":
    main() 