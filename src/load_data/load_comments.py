import argparse
import re
import logging
import pandas as pd
from typing import Optional
from pathlib import Path
import pandas as pd
import logging
from typing import Optional, Set



def process_comments_file(
    comments_file: Path,
    aggregate_chunk,
    merge_summaries,
    chunksize: int = 10_000,


) -> pd.DataFrame:
    """
    Load comments from a bz2-encoded JSON-lines file in chunks,
    aggregate sentiment statistics, and classify sentiment.
    Returns a DataFrame summary for this file.
    """
    try:
        reader = pd.read_json(
            comments_file,
            compression='bz2',
            lines=True,
            chunksize=chunksize,
            dtype=False
        )
    except ValueError as e:
        logging.error("Error reading JSON from %s: %s", comments_file, e)
        raise
    except FileNotFoundError:
        msg = f"Comments file not found: {comments_file}"
        logging.exception(msg)
        raise

    cumulative: Optional[pd.DataFrame] = None
    for idx, chunk in enumerate(reader):
        logging.debug("Chunk %d: %d rows", idx, len(chunk))
        chunk_summary = aggregate_chunk(chunk)
        cumulative = merge_summaries(cumulative, chunk_summary)

    if cumulative is None or cumulative.empty:
        logging.warning("No data loaded from %s", comments_file)
        return pd.DataFrame()

    return cumulative


def main():

    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--comments_file', default=None, type=str, required=True, help='Comments file')
    parser.add_argument('--subreddits_file', default=None, type=str, required=False, help='Subreddits file')
    parser.add_argument('--target_dir', default=None, type=str, required=False, help='Directory to store comments')
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

    # Store extracted comments
    if args.target_dir:
        target_file = '{}/comments_extracted_{}.json'.format(
            args.target_dir, re.findall(r'\d{4}-\d{2}', args.comments_file)[0]
        )
        comments.to_json(
            target_file,
            orient='records',
            lines=True
        )


if __name__ == '__main__':
    main()
