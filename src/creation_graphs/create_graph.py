import argparse
import json
import re
import os

import pandas as pd
import numpy as np
from tqdm import tqdm


def process_comments_file(comments_file, bot_users):
    """Process a single comments file and return author-subreddit counts"""
    author_subreddit_counts = pd.DataFrame()
    
    # Count total number of lines for progress bar
    total_lines = sum(1 for _ in open(comments_file, 'rb'))
    
    # Process comments in chunks
    for chunk in tqdm(pd.read_json(comments_file, compression='bz2', lines=True, dtype=False, chunksize=100000),
                     total=total_lines//100000 + 1,
                     desc=f"Processing {os.path.basename(comments_file)}"):
        # Filter out deleted authors
        chunk = chunk[chunk['author'] != '[deleted]']
        # Filter out bot users
        chunk = chunk[~chunk['author'].isin(bot_users)]

        # Count author-subreddit interactions using groupby
        counts = chunk.groupby(['author', 'subreddit']).size().reset_index(name='count')
        author_subreddit_counts = pd.concat([author_subreddit_counts, counts])
    
    return author_subreddit_counts


def create_graph(author_subreddit_counts, threshold, target_dir, date):
    """Create and save a graph from author-subreddit counts"""
    # Filter by threshold
    author_subreddit_counts = author_subreddit_counts[author_subreddit_counts['count'] >= threshold]
    
    # Create subreddit mapping
    unique_subreddits = author_subreddit_counts['subreddit'].unique()
    subreddits_to_id = {sub: idx for idx, sub in enumerate(unique_subreddits)}
    id_to_subreddits = list(unique_subreddits)
    num_subreddits = len(subreddits_to_id)

    print(f"\nFound {num_subreddits} unique subreddits")
    print("Creating adjacency matrix...")

    # Create a binary matrix where each row is an author and each column is a subreddit
    author_subreddit_matrix = pd.crosstab(
        author_subreddit_counts['author'],
        author_subreddit_counts['subreddit']
    )
    
    # Convert to numpy array for faster computation
    author_subreddit_matrix = author_subreddit_matrix.values
    
    # Create adjacency matrix by multiplying the matrix with its transpose
    adjacency_mat = np.dot(author_subreddit_matrix.T, author_subreddit_matrix)
    
    # Zero out the diagonal (self-connections)
    np.fill_diagonal(adjacency_mat, 0)

    print("Writing results to file...")

    # Store graph
    target_file = '{}/graph_{}_t{}.csv'.format(
        target_dir, date, threshold
    )

    with open(target_file, 'w') as f:
        f.write('node_1,node_2,weight\n')
        for i in tqdm(range(num_subreddits), desc="Writing edges"):
            for j in range(i + 1, num_subreddits):
                if adjacency_mat[i][j] > 0:
                    f.write(f'{id_to_subreddits[i]},{id_to_subreddits[j]},{adjacency_mat[i][j]}\n')

    print(f"\nDone! Results written to {target_file}")


def main():
    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--comments_files', nargs='+', type=str, required=True, 
                       help='List of comment files to process')
    parser.add_argument('--target_dir', default=None, type=str, required=True, 
                       help='Directory to store comments')
    parser.add_argument('--threshold', default=1, type=int, required=False, 
                       help='Threshold for edge weight')
    args = parser.parse_args()

    # Create target directory if it doesn't exist
    os.makedirs(args.target_dir, exist_ok=True)

    print("Loading bot users...")

    # Load bot users
    bot_users = set()
    with open(os.path.join('data', 'metadata', 'users_metadata.json'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            user_metadata = json.loads(line)
            if user_metadata['bot'] == 1:
                bot_users.add(user_metadata['author'])

    print("Processing comment files...")

    # Process each comment file separately
    for comments_file in args.comments_files:
        print(f"\nProcessing file: {os.path.basename(comments_file)}")
        # Extract date from filename
        date_match = re.findall(r'\d{4}-\d{2}', comments_file)[0]
        
        # Process the file
        author_subreddit_counts = process_comments_file(comments_file, bot_users)
        
        # Create and save graph for this file
        create_graph(author_subreddit_counts, args.threshold, args.target_dir, date_match)


if __name__ == '__main__':
    main()
