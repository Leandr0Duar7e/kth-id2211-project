import pandas as pd
import numpy as np
from node2vec import Node2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import networkx as nx
import matplotlib.pyplot as plt
import os
import argparse

def load_graph(file_path):
    """Load graph from CSV file"""
    df = pd.read_csv(file_path)
    # Create a NetworkX graph
    G = nx.from_pandas_edgelist(df, source='subreddit_a', target='subreddit_b', edge_attr='weight')
    return G, df

def create_node2vec_model(G, dimensions=128, walk_length=5, num_walks=100, p=1.0, q=1.0):
    """Create and train Node2Vec model"""
    # Create Node2Vec model
    node2vec = Node2Vec(
        G,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=4,
        seed=42
    )
    
    # Train the model
    model = node2vec.fit(window=3, min_count=1, batch_words=4)
    
    return model

def generate_edge_embeddings(model, edges):
    """Generate embeddings for edges"""
    edge_embeddings = []
    for edge in edges:
        try:
            source_embedding = model.wv[str(edge[0])]
            target_embedding = model.wv[str(edge[1])]
            # Concatenate source and target embeddings
            edge_embedding = np.concatenate([source_embedding, target_embedding])
            edge_embeddings.append(edge_embedding)
        except KeyError:
            # Skip edges where nodes are not in the vocabulary
            continue
    return np.array(edge_embeddings)

def prepare_training_data(G, df):
    """Prepare positive and negative samples for training"""
    # Get all existing edges
    positive_edges = list(G.edges())
    
    # Generate negative samples (non-existent edges)
    all_nodes = list(G.nodes())
    negative_edges = []
    while len(negative_edges) < len(positive_edges):
        source = np.random.choice(all_nodes)
        target = np.random.choice(all_nodes)
        if source != target and not G.has_edge(source, target):
            negative_edges.append((source, target))
    
    # Combine positive and negative samples
    X = positive_edges + negative_edges
    y = [1] * len(positive_edges) + [0] * len(negative_edges)
    
    return X, y

def process_single_graph(file_path):
    """Process a single graph and return its AUC score"""
    print(f"Processing {os.path.basename(file_path)}...")
    
    # Load the graph
    G, df = load_graph(file_path)
    
    # Create and train Node2Vec model
    model = create_node2vec_model(G)
    
    # Prepare training data
    X, y = prepare_training_data(G, df)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Generate embeddings for edges
    X_train_embeddings = generate_edge_embeddings(model, X_train)
    X_test_embeddings = generate_edge_embeddings(model, X_test)
    
    # Scale embeddings
    scaler = StandardScaler()
    X_train_embeddings = scaler.fit_transform(X_train_embeddings)
    X_test_embeddings = scaler.transform(X_test_embeddings)

    # Train a simple classifier with increased max_iter
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
    clf.fit(X_train_embeddings, y_train)
    
    # Make predictions
    y_pred = clf.predict_proba(X_test_embeddings)[:, 1]
    
    # Calculate AUC score
    auc_score = roc_auc_score(y_test, y_pred)
    
    return auc_score

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process graphs for edge prediction and calculate AUC scores.')
    parser.add_argument('--graphs', nargs='+', required=True,
                      help='List of graph file paths to process')
    parser.add_argument('--output', default='edge_prediction_results.csv',
                      help='Output CSV file path (default: edge_prediction_results.csv)')
    parser.add_argument('--plot', default='edge_prediction_scores.png',
                      help='Output plot file path (default: edge_prediction_scores.png)')
    
    args = parser.parse_args()
    
    # Process each graph and collect results
    results = []
    for file_path in args.graphs:
        try:
            auc_score = process_single_graph(file_path)
            graph_name = os.path.basename(file_path)
            results.append({
                'graph_file': graph_name,
                'auc_score': auc_score
            })
            print(f"Completed {graph_name} with AUC score: {auc_score:.4f}")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    if not results:
        print("No results were generated!")
        return
    
    # Create results DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    print(f"\nResults have been saved to {args.output}")
    
    # Create a visualization of the results
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['graph_file'], results_df['auc_score'], marker='o')
    plt.xticks(rotation=45)
    plt.title('Edge Prediction AUC Scores Across Different Graphs')
    plt.xlabel('Graph File')
    plt.ylabel('AUC Score')
    plt.tight_layout()
    plt.savefig(args.plot)
    plt.close()
    print(f"Plot has been saved to {args.plot}")

if __name__ == "__main__":
    main() 


# python edge_prediction.py --graphs \
# data/processed/graphs/graph_2016-08_t10.csv \
# data/processed/graphs/graph_2016-09_t10.csv \
# data/processed/graphs/graph_2016-10_t10.csv \
# data/processed/graphs/graph_2016-11_t10.csv \
# data/processed/graphs/graph_2016-12_t10.csv \
# data/processed/graphs/graph_2017-01_t10.csv \
# data/processed/graphs/graph_2017-02_t10.csv \
# data/processed/graphs/leandro_graph_2016-08.csv \
# data/processed/graphs/leandro_graph_2016-09.csv \
# data/processed/graphs/leandro_graph_2016-10.csv \
# data/processed/graphs/leandro_graph_2016-11.csv \
# data/processed/graphs/leandro_graph_2016-12.csv \
# data/processed/graphs/leandro_graph_2017-01.csv \
# data/processed/graphs/leandro_graph_2017-02.csv \

# python edge_prediction.py --graphs \
# data/processed/graphs/graph_2016-11_t10.csv \
# data/processed/graphs/graph_2016-12_t10.csv \
# data/processed/graphs/graph_2017-01_t10.csv \
# data/processed/graphs/graph_2017-02_t10.csv \
# data/processed/graphs/leandro_graph_2016-11.csv \
# data/processed/graphs/leandro_graph_2016-12.csv \
# data/processed/graphs/leandro_graph_2017-01.csv \
# data/processed/graphs/leandro_graph_2017-02.csv \