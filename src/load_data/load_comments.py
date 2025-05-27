import argparse
import re
<<<<<<< HEAD
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
=======
import pandas as pd
from pathlib import Path
from typing import Optional, Set


class CommentLoader:
    """Loads and preprocesses comments from monthly Reddit comment files."""

    def __init__(
        self, comments_file_path: Path, bot_user_ids: Optional[Set[str]] = None
    ):
        """
        Initializes the CommentLoader.

        Args:
            comments_file_path (Path): Path to the bz2 compressed JSONL comments file.
            bot_user_ids (Optional[Set[str]], optional): A set of bot user IDs to filter out.
                                                        Defaults to None (no bot filtering).
        """
        self.comments_file_path = comments_file_path
        self.bot_user_ids = bot_user_ids if bot_user_ids else set()

    def load_and_filter_comments(
        self,
        subreddits: Optional[Set[str]] = None,
        columns_to_keep: Optional[list[str]] = None,
        chunksize: int = 10000,
    ) -> pd.DataFrame:
        """
        Loads comments in chunks, filters by subreddits and bots, and selects specified columns.

        Args:
            subreddits (Optional[Set[str]], optional): A set of subreddit names to filter by.
                                                       Defaults to None (no subreddit filtering).
            columns_to_keep (Optional[list[str]], optional): List of columns to retain.
                                                            Defaults to ["id", "author", "subreddit_id", "body_cleaned"].
            chunksize (int, optional): Number of lines to read per chunk. Defaults to 10000.

        Returns:
            pd.DataFrame: A DataFrame with the filtered and processed comments.
        """
        if columns_to_keep is None:
            columns_to_keep = ["id", "author", "subreddit_id", "body_cleaned"]
        else:
            # Ensure essential columns for bot filtering are present if bot_user_ids is used
            if self.bot_user_ids and "author" not in columns_to_keep:
                columns_to_keep.append("author")

        processed_chunks = []
        try:
            for chunk_df in pd.read_json(
                self.comments_file_path,
                compression="bz2",
                lines=True,
                dtype=False,  # Let pandas infer dtypes initially
                chunksize=chunksize,
            ):
                # Filter by subreddits if provided
                if subreddits:
                    chunk_df = chunk_df[chunk_df["subreddit"].isin(subreddits)]

                # Filter out bots if bot_user_ids are provided
                if self.bot_user_ids:
                    chunk_df = chunk_df[
                        ~chunk_df["author"].astype(str).isin(self.bot_user_ids)
                    ]

                # Select specified columns, handling potential missing columns gracefully
                existing_cols = [
                    col for col in columns_to_keep if col in chunk_df.columns
                ]
                chunk_df = chunk_df[existing_cols]

                processed_chunks.append(chunk_df)
        except FileNotFoundError:
            print(f"Error: Comments file not found at {self.comments_file_path}")
            return pd.DataFrame(columns=columns_to_keep)
        except ValueError as e:
            if "Expected object or value" in str(e):
                print(
                    f"Error: File {self.comments_file_path} might be empty or not valid JSONL. Details: {e}"
                )
            else:
                print(
                    f"An unexpected ValueError occurred while reading {self.comments_file_path}: {e}"
                )
            return pd.DataFrame(columns=columns_to_keep)
        except Exception as e:
            print(
                f"An unexpected error occurred while reading {self.comments_file_path}: {e}"
            )
            return pd.DataFrame(columns=columns_to_keep)

        if not processed_chunks:
            print(
                f"No comments loaded from {self.comments_file_path}. This might be due to filters or an empty file."
            )
            final_cols_for_empty_df = [
                col
                for col in columns_to_keep
                if col != "author" or not self.bot_user_ids
            ]
            return pd.DataFrame(columns=final_cols_for_empty_df)

        comments_df = pd.concat(processed_chunks, ignore_index=True)

        # Final column selection after concatenation (in case 'author' was temporarily added for bot filtering)
        final_columns = [
            col for col in columns_to_keep if col != "author" or not self.bot_user_ids
        ]
        # Ensure all desired final columns exist, adding them with NaN if not (e.g. body_cleaned might be missing in rare empty chunks)
        for col in final_columns:
            if col not in comments_df.columns:
                comments_df[col] = pd.NA
        return comments_df[final_columns]
>>>>>>> main


def main():
    parser = argparse.ArgumentParser(description="Load and filter Reddit comments.")
    parser.add_argument(
        "--comments_file",
        type=Path,
        required=True,
        help="Path to the bz2 compressed JSONL comments file (e.g., data/comments/comments_2016-11.bz2).",
    )
    parser.add_argument(
        "--subreddits_file",
        type=Path,
        default=None,
        help="(Optional) Path to a file containing a list of subreddits to filter by (one per line).",
    )
    parser.add_argument(
        "--users_metadata_file",
        type=Path,
        default=None,  # Make this optional, bot filtering will be skipped if not provided
        help="(Optional) Path to the users_metadata.jsonl file for identifying bots.",
    )
    parser.add_argument(
        "--target_dir",
        type=Path,
        default=None,
        help="(Optional) Directory to store the processed comments as a JSONL file.",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default=None,
        help="(Optional) Custom name for the output file. If not provided, a name is generated.",
    )

    args = parser.parse_args()

    bot_ids = set()
    if args.users_metadata_file:
        # Assuming utils.py is in the same directory or accessible via PYTHONPATH
        from utils import load_bot_user_ids

        bot_ids = load_bot_user_ids(args.users_metadata_file)

    subreddits_to_filter = None
    if args.subreddits_file:
        if args.subreddits_file.exists():
            with open(args.subreddits_file, "r") as f:
                subreddits_to_filter = set(line.strip() for line in f if line.strip())
            print(
                f"Loaded {len(subreddits_to_filter)} subreddits to filter from {args.subreddits_file.name}."
            )
        else:
            print(
                f"Warning: Subreddits file {args.subreddits_file} not found. No subreddit filtering will be applied."
            )

    loader = CommentLoader(args.comments_file, bot_user_ids=bot_ids)

    # Specify columns to keep, author is needed for bot filtering and removed later if not in final desired list
    # For topic modeling and sentiment, we will primarily need id, subreddit_id, body_cleaned.
    # Let's assume final desired columns are these (after bot filtering).
    final_desired_columns = ["id", "subreddit_id", "body_cleaned"]
    columns_for_loader = list(
        set(final_desired_columns + ["author"])
    )  # Ensure author is present for filtering

    comments_df = loader.load_and_filter_comments(
        subreddits=subreddits_to_filter, columns_to_keep=columns_for_loader
    )

    # Now, ensure only final_desired_columns are present
    if not comments_df.empty:
        current_cols = list(comments_df.columns)
        cols_to_select_final = [
            col for col in final_desired_columns if col in current_cols
        ]
        comments_df = comments_df[cols_to_select_final]

    if not comments_df.empty:
        print(f"Successfully loaded and filtered {len(comments_df)} comments.")
        print("Sample of loaded comments:")
        print(comments_df.head())

        if args.target_dir:
            args.target_dir.mkdir(parents=True, exist_ok=True)
            if args.output_filename:
                target_file = args.target_dir / args.output_filename
            else:
                # Generate a filename based on the input comments file
                input_file_stem = args.comments_file.stem.replace(
                    ".json", ""
                )  # remove .json if present from .bz2
                target_file_name = f"{input_file_stem}_filtered.jsonl"
                target_file = args.target_dir / target_file_name

            comments_df.to_json(
                target_file,
                orient="records",
                lines=True,
                force_ascii=False,  # Better for various text characters
            )
            print(f"Filtered comments saved to {target_file}")
    else:
        print("No comments were processed or an error occurred.")


if __name__ == "__main__":
    main()
