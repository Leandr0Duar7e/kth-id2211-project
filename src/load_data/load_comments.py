import argparse
import re
import logging
import pandas as pd
from typing import Optional

import pandas as pd
import logging
from typing import Optional, Set

def load_comments(
    comments_file: str,
    subreddits: Optional[Set[str]] = None,
    users: Optional[Set[str]] = None,
    chunksize: int = 10_000
) -> pd.DataFrame:
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

    frames: list[pd.DataFrame] = []

    for idx, chunk in enumerate(reader):
        logging.debug("Processing chunk %d with %d rows", idx, len(chunk))

        if subreddits is not None:
            found_subs = chunk['subreddit'].dropna().unique()
            subreddits.update(found_subs)
            logging.debug("Updated subreddits set with %d entries from chunk %d", len(found_subs), idx)

        if users is not None:
            found_users = chunk['author'].dropna().unique()
            users.update(found_users)
            logging.debug("Updated users set with %d entries from chunk %d", len(found_users), idx)

        frames.append(chunk)

    if not frames:
        logging.warning("No data loaded from %s", comments_file)
        return pd.DataFrame()

    result = pd.concat(frames, sort=False)
    logging.info("Loaded total %d comments", len(result))
    return result




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
