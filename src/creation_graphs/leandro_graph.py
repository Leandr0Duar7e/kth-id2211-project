import glob
import json
import logging
import os
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Optional

global BOT_USERS

def process_comments_file(
    comments_file: Path,
    chunksize: int = 10_000

) -> pd.DataFrame:
    """
    Load and aggregate sentiment statistics from a bz2-encoded JSON-lines file in chunks.
    Returns a DataFrame with columns:
      ['author', 'subreddit', 'num_pos', 'num_neg', 'sum_pos_scores', 'sum_neg_scores', 'total_comments']
    """
    # Use efficient dtypes for memory
    dtypes = {"author": "category", "subreddit": "category"}
    reader = pd.read_json(
        comments_file,
        compression='bz2',
        lines=True,
        chunksize=chunksize,
        dtype=dtypes
    )

    agg_chunks = []
    for chunk in reader:
        # Filter out deleted
        chunk = chunk.loc[
            (chunk['author'] != '[deleted]') & (~chunk['author'].isin(BOT_USERS))
            ]
        # adjust scores (remove default upvote)
        scores = chunk['score'].sub(1, fill_value=0)
        pos_mask = scores > 0
        neg_mask = scores < 0

        summary = (
            chunk.assign(
                _score=scores,
                is_pos=pos_mask,
                is_neg=neg_mask,
                pos_score=lambda df: df['_score'].clip(lower=0),
                neg_score=lambda df: df['_score'].clip(upper=0).abs()
            )
            .groupby(['author', 'subreddit'], observed=True)
            .agg(
                num_pos=('is_pos', 'sum'),
                num_neg=('is_neg', 'sum'),
                sum_pos_scores=('pos_score', 'sum'),
                sum_neg_scores=('neg_score', 'sum'),
                total_comments=('_score', 'count')
            )
        ).reset_index()
        agg_chunks.append(summary)

    if not agg_chunks:
        return pd.DataFrame(
            columns=['author', 'subreddit', 'num_pos', 'num_neg', 'sum_pos_scores', 'sum_neg_scores', 'total_comments']
        )
    # Merge all chunks efficiently
    df = pd.concat(agg_chunks, ignore_index=True)
    df = (
        df.groupby(['author', 'subreddit'], observed=True, as_index=False)
          .sum()
    )
    return df


def classify_sentiment(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'sentiment' column with values 'Positive', 'Negative', or 'Neutral',
    based on the weighted Leandro score:
      (sum_pos_scores * num_pos - sum_neg_scores * num_neg) / (num_pos + num_neg)
    """
    df = summary.copy()

    # compute denominator, avoid div-by-zero
    tot = df['num_pos'] + df['num_neg']
    raw = (df['sum_pos_scores'] * df['num_pos'] -
           df['sum_neg_scores'] * df['num_neg']) / tot.replace(0, np.nan)

    # classify
    sentiment = np.where(raw >  0, 'Positive',
                 np.where(raw <  0, 'Negative',
                          'Neutral'))

    df['sentiment'] = pd.Categorical(sentiment,
                                     categories=['Negative','Neutral','Positive'],
                                     ordered=True)
    return df


def build_subreddit_interactions(
    summary: pd.DataFrame
) -> pd.DataFrame:
    """
    Build weighted edges between subreddits where weight = number of distinct authors
    with the same sentiment in both subreddits.
    """
    # Keep only relevant columns and drop duplicates to avoid overcounting
    summary = summary[['author', 'subreddit', 'sentiment']].drop_duplicates()

    # Do not look at neutral
    summary = summary[summary['sentiment'] != 'Neutral']

    # Merge on author and sentiment to ensure both entries have the same sentiment
    merged = summary.merge(summary, on=['author', 'sentiment'])

    # Keep only pairs with different subreddits and order them to avoid duplicates
    merged = merged[merged['subreddit_x'] < merged['subreddit_y']]

    # Group by subreddit pair and count the number of distinct authors with same sentiment
    edges = (
        merged.groupby(['subreddit_x', 'subreddit_y'], observed=True)
              .size()
              .reset_index(name='weight')
              .rename(columns={'subreddit_x': 'subreddit_a', 'subreddit_y': 'subreddit_b'})
    )

    return edges


def save_edges(edges: pd.DataFrame, out_csv: Path) -> None:
    edges.to_csv(out_csv, index=False)


def create_interaction_graph(edges: pd.DataFrame) -> nx.Graph:
    """Return a NetworkX graph from the edge list."""
    return nx.from_pandas_edgelist(
        edges,
        source='subreddit_a',
        target='subreddit_b',
        edge_attr='weight',
        create_using=nx.Graph
    )


def main(
    comments_file: Path,
    summary_out: Path,
    out_edge_csv: Path = Path("subreddit_edges.csv"),
    chunksize: int = 10_000
) -> Optional[pd.DataFrame]:
    """
    Full ETL: load, aggregate, classify, build interactions, and save edges.
    Returns edges DataFrame.
    """
    summary = process_comments_file(comments_file, chunksize)
    if summary.empty:
        return None
    classified = classify_sentiment(summary)
    try:
        classified.to_csv(summary_out, index=False)
    except:
        print("Error while writing summary")
    edges = build_subreddit_interactions(classified)
    save_edges(edges, out_edge_csv)
    return edges


if __name__ == "__main__":


    # Define path relative to project root
    project_root = Path(__file__).resolve().parents[2]
    metadata_path = project_root / 'data' / 'metadata' / 'users_metadata.json'

    # Load only bot users
    try:
        df = pd.read_json(metadata_path, lines=True)
        BOT_USERS = set(df.loc[df['bot'] == 1, 'author'].dropna())
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading metadata: {e}")
        BOT_USERS = set()

    input_folder = Path("/home/two_play/datamining_data/2016")
    output_folder = input_folder / 'graphs'
    output_folder.mkdir(parents=True, exist_ok=True)
    graphs: dict[str, nx.Graph] = {}


    for file_path in sorted(input_folder.glob("*.bz2")):
        month_name = file_path.stem
        base = month_name.rsplit('_', 1)[1]#
        out_csv = output_folder / f"graph_{base}.csv"
        out_summary_csv = output_folder / f"summary_{base}.csv"

        print(f"Processing: {file_path.name} → {out_csv.name}")
        try:
            main(comments_file=file_path, out_edge_csv=out_csv, summary_out=out_summary_csv)
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
    # import argparse
    #
    # parser = argparse.ArgumentParser(description="Build subreddit interaction network.")
    # parser.add_argument("comments_file", type=Path, help="path to comments.bz2 jsonlines")
    # parser.add_argument("--out", type=Path, default="subreddit_edges.csv", help="output CSV path")
    # parser.add_argument("--chunksize", type=int, default=10000)
    # args = parser.parse_args()
    #
    # main(args.comments_file, args.out, args.chunksize)
