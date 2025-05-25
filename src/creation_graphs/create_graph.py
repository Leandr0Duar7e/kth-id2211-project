import argparse
import re
import os

import pandas as pd
import numpy as np


def main():
    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--comments_file', default=None, type=str, required=True, help='Comments file')
    parser.add_argument('--subreddits_file', default=None, type=str, required=False, help='Subreddits file')
    parser.add_argument('--target_dir', default=None, type=str, required=True, help='Directory to store comments')
    parser.add_argument('--threshold', default=1, type=int, required=False, help='Threshold for edge weight')
    args = parser.parse_args()

    # Create target directory if it doesn't exist
    os.makedirs(args.target_dir, exist_ok=True)

    # Read subreddits
    subreddits = None
    if args.subreddits_file:
        with open(args.subreddits_file, 'r') as f:
            subreddits = set(f.read().strip().split('\n'))

    # Initialize data structures
    authors_counts = {}
    subreddits_to_id = {}
    id_to_subreddits = []

    # Process comments in chunks
    for c in pd.read_json(args.comments_file, compression='bz2', lines=True, dtype=False, chunksize=10000):
        if subreddits:
            c = c[c.subreddit.isin(subreddits)]
        
        # Filter out deleted authors
        c = c[c['author'] != '[deleted]']
        
        # Process each comment
        for _, comm in c.iterrows():
            if comm['author'] not in authors_counts:
                authors_counts[comm['author']] = {comm['subreddit']: 1}
            else:
                authors_counts[comm['author']][comm['subreddit']] = authors_counts[comm['author']].get(comm['subreddit'], 0) + 1
            
            if comm['subreddit'] not in subreddits_to_id:
                subreddits_to_id[comm['subreddit']] = len(subreddits_to_id)
                id_to_subreddits.append(comm['subreddit'])

    num_subreddits = len(subreddits_to_id)

    # Create adjacency matrix
    adjacency_mat = np.zeros((num_subreddits, num_subreddits), dtype=np.int32)
    for author in authors_counts:
        for i in range(num_subreddits):
            for j in range(i + 1, num_subreddits):
                subreddit_i = id_to_subreddits[i]
                subreddit_j = id_to_subreddits[j]
                if (subreddit_i in authors_counts[author] and 
                    subreddit_j in authors_counts[author] and
                    authors_counts[author][subreddit_i] >= args.threshold and 
                    authors_counts[author][subreddit_j] >= args.threshold):
                    adjacency_mat[i][j] += 1
                    adjacency_mat[j][i] = adjacency_mat[i][j]  # Make matrix symmetric

    # Store graph
    target_file = '{}/graph_{}.txt'.format(
        args.target_dir, re.findall(r'\d{4}-\d{2}', args.comments_file)[0]
    )

    with open(target_file, 'w') as f:
        f.write('node_1,node_2,weight\n')
        for i in range(num_subreddits):
            for j in range(i + 1, num_subreddits):
                if adjacency_mat[i][j] > 0:
                    f.write(f'{id_to_subreddits[i]},{id_to_subreddits[j]},{adjacency_mat[i][j]}\n')


if __name__ == '__main__':
    main()
