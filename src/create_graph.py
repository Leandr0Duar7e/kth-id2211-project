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
    args = parser.parse_args()

    # Read subreddits
    if args.subreddits_file:
        with open(args.subreddits_file, 'r') as f:
            subreddits = set(f.read().strip().split('\n'))

    # Load comments in chunks
    comments = list()
    for c in pd.read_json(args.comments_file, compression='bz2', lines=True, dtype=False, chunksize=10000):
        if args.subreddits_file:
            c = c[c.subreddit.isin(subreddits)]
        comments.append(c)
    comments = pd.concat(comments, sort=True)

    # Do something with comments here

    authors_counts = {}
    subreddits_to_id = {}
    id_to_subreddits = []

    comments = comments[comments['author'] != '[deleted]']
    for i, comm in comments.iterrows():
        if comm['author'] not in authors_counts:
            authors_counts[comm['author']] = {f'{comm["subreddit"]}': 1}
        else:
            if comm['subreddit'] not in authors_counts[comm['author']]:
                authors_counts[comm['author']][comm['subreddit']] = 1
            else:
                authors_counts[comm['author']][comm['subreddit']] += 1 
        
        if comm['subreddit'] not in subreddits_to_id:
            subreddits_to_id[comm['subreddit']] = i
            id_to_subreddits.append(comm['subreddit'])

    num_subreddits = len(subreddits_to_id)

    threshold = 1
    adjacency_mat = np.zeros((num_subreddits, num_subreddits), dtype=np.int32)
    for author in authors_counts:
        for i in range(num_subreddits):
            for j in range(i + 1, num_subreddits):
                if not id_to_subreddits[i] in authors_counts[author] or not id_to_subreddits[j] in authors_counts[author]:
                    continue
                if authors_counts[author][id_to_subreddits[i]] >= threshold and authors_counts[author][id_to_subreddits[j]] >= threshold:
                    adjacency_mat[i][j] += 1

    # Store extracted comments
    if args.target_dir:
        target_file = '{}/graph_{}.txt'.format(
            args.target_dir, re.findall(r'\d{4}-\d{2}', args.comments_file)[0]
        )
        with open(target_file, 'w') as f:
            f.write('node_1,node_2,weighted')
            for i in range(num_subreddits):
                for j in range(i + 1, num_subreddits):
                    if adjacency_mat[i][j] > 0:
                        f.write(f'{id_to_subreddits[i]},{id_to_subreddits[j]},{adjacency_mat[i][j]}\n')


if __name__ == '__main__':
    main()
