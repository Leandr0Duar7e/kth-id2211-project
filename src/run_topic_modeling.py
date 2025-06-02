import argparse
import pandas as pd
from pathlib import Path
import time
import json
import os
import gc  # For garbage collection
from glob import glob
import re

# Project specific imports
from utils import get_project_root, ensure_dir
from topic_modeling import SimpleLdaTopicModeler


def process_comments_file_for_topics(
    comments_file_path: Path,
    output_dir: Path,
    output_filename: str = None,
    num_lda_topics: int = 20,
    lda_passes: int = 5,
    lda_iterations: int = 30,
    chunk_size: int = 50000,  # Process in chunks of 50k comments
):
    """
    Processes a comments file for topic modeling only.

    Args:
        comments_file_path: Path to the comments file (.bz2 or .json)
        output_dir: Directory to save results
        output_filename: Optional custom filename for output
        num_lda_topics: Number of topics for LDA
        lda_passes: Number of passes for LDA training
        lda_iterations: Number of iterations for LDA training
        chunk_size: Number of comments to process in each chunk
    """
    print(f"\n--- Processing {comments_file_path.name} for topics ---")
    start_time = time.time()

    # Create output directory
    ensure_dir(output_dir)

    # Determine output filename
    if output_filename:
        save_path = output_dir / output_filename
    else:
        base_name = comments_file_path.stem.replace(".json", "")
        save_path = output_dir / f"{base_name}_topics.jsonl"

    # Check if file already exists and remove if it does
    if save_path.exists():
        os.remove(save_path)
        print(f"Removed existing output file: {save_path}")

    # Initialize chunking parameters
    total_comments = 0
    chunk_counter = 0
    total_chunks_processed = 0

    # Initialize topic modeler
    topic_modeler = SimpleLdaTopicModeler(
        num_topics=num_lda_topics,
        passes=lda_passes,
        iterations=lda_iterations,
        lemmatize=False,  # No lemmatization since text is already processed
    )

    # Process in chunks
    chunk_reader = None

    try:
        # Set up chunk reader based on file type
        if comments_file_path.suffix == ".bz2":
            chunk_reader = pd.read_json(
                comments_file_path,
                lines=True,
                compression="bz2",
                chunksize=chunk_size,
            )
        else:
            chunk_reader = pd.read_json(
                comments_file_path, lines=True, chunksize=chunk_size
            )

        # First pass: Get all unique link_ids and aggregate their content
        print("First pass: Collecting and aggregating threads (link_ids)...")
        aggregated_threads = {}

        for chunk_df in chunk_reader:
            chunk_counter += 1
            total_comments += len(chunk_df)
            print(
                f"Reading chunk {chunk_counter} with {len(chunk_df)} comments... (Total so far: {total_comments})"
            )

            # Ensure required columns exist
            if "link_id" not in chunk_df.columns:
                print("Error: Required column 'link_id' not found in comments file.")
                return

            # Use body_cleaned if it exists, otherwise use body
            if "body_cleaned" not in chunk_df.columns:
                if "body" in chunk_df.columns:
                    chunk_df["body_cleaned"] = chunk_df["body"].astype(str).fillna("")
                else:
                    print(
                        "Error: Neither 'body_cleaned' nor 'body' found in comments file."
                    )
                    return

            # Aggregate comments by link_id within this chunk
            for link_id, group in chunk_df.groupby("link_id"):
                texts = [
                    str(text)
                    for text in group["body_cleaned"]
                    if pd.notna(text) and str(text).strip() != ""
                ]
                if texts:  # Only add if there are non-empty texts
                    if link_id in aggregated_threads:
                        aggregated_threads[link_id].extend(texts)
                    else:
                        aggregated_threads[link_id] = texts

        # Reset chunk reader for second pass
        if comments_file_path.suffix == ".bz2":
            chunk_reader = pd.read_json(
                comments_file_path,
                lines=True,
                compression="bz2",
                chunksize=chunk_size,
            )
        else:
            chunk_reader = pd.read_json(
                comments_file_path, lines=True, chunksize=chunk_size
            )

        # Create DataFrame for topic modeling
        print("\nStep 2: Performing LDA topic modeling on aggregated texts...")
        aggregated_texts_df = pd.DataFrame(
            {
                "link_id": list(aggregated_threads.keys()),
                "merged_body_cleaned": [
                    " ".join(texts) for texts in aggregated_threads.values()
                ],
            }
        )

        # Free up memory
        del aggregated_threads
        gc.collect()

        # Check if we have threads to model
        if aggregated_texts_df.empty:
            print("No non-empty threads to model topics for. Assigning default topic.")
            thread_topics_df = pd.DataFrame(columns=["link_id", "topic_id"])
        else:
            # Apply topic modeling
            aggregated_texts_with_topics_df = topic_modeler.fit_transform(
                aggregated_texts_df, text_column="merged_body_cleaned"
            )

            # Get thread topics mapping
            thread_topics_df = aggregated_texts_with_topics_df[["link_id", "topic_id"]]

            # Print topics summary
            print("\n--- LDA Topic Summary ---")
            topic_words = topic_modeler.get_topic_words()
            for topic_id, words in topic_words.items():
                print(f"Topic {topic_id}: {words}")
            print("--- End LDA Topic Summary ---\n")

        # Free up memory
        del aggregated_texts_df
        gc.collect()

        # Second pass: Process each chunk and merge with topic assignments
        print("\nProcessing chunks and assigning topics...")
        chunk_counter = 0

        for chunk_df in chunk_reader:
            chunk_counter += 1
            total_chunks_processed += 1
            print(f"Processing chunk {chunk_counter} with {len(chunk_df)} comments...")

            # Ensure body_cleaned exists
            if "body_cleaned" not in chunk_df.columns:
                if "body" in chunk_df.columns:
                    chunk_df["body_cleaned"] = chunk_df["body"].astype(str).fillna("")
                else:
                    continue  # Skip this chunk if no body content

            # Merge thread topics into comments
            if not thread_topics_df.empty:
                chunk_df = pd.merge(
                    chunk_df, thread_topics_df, on="link_id", how="left"
                )
                chunk_df["topic_id"] = chunk_df["topic_id"].fillna(-1).astype(int)
            else:
                chunk_df["topic_id"] = -1

            # Explicitly assign -1 topic to empty comments
            chunk_df.loc[
                chunk_df["body_cleaned"].astype(str).str.strip() == "", "topic_id"
            ] = -1

            # Select columns to save
            final_columns = [
                "id",
                "link_id",
                "author",
                "topic_id",
                "body_cleaned",
            ]

            for col in final_columns:
                if col not in chunk_df.columns:
                    chunk_df[col] = None

            final_df = chunk_df[final_columns]

            # Append to output file
            if chunk_counter == 1:  # First chunk, create the file
                final_df.to_json(
                    save_path, orient="records", lines=True, force_ascii=False
                )
            else:  # Append to existing file
                with open(save_path, "a", encoding="utf-8") as f:
                    final_df.to_json(f, orient="records", lines=True, force_ascii=False)

            # Free up memory
            del chunk_df, final_df
            gc.collect()

        total_time = time.time() - start_time
        print(f"Successfully processed and saved to {save_path}")
        print(f"Total comments processed: {total_comments}")
        print(f"Total chunks processed: {total_chunks_processed}")
        print(f"Total time: {total_time:.2f} seconds.")

    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback

        traceback.print_exc()
        return


def process_monthly_comments(
    comments_dir: Path,
    output_dir: Path,
    start_month: str = "2016-08",
    end_month: str = "2017-02",
    **kwargs,
):
    """
    Process all monthly comment files in the given date range.

    Args:
        comments_dir: Directory containing comments files
        output_dir: Directory to save topic modeling results
        start_month: Start month in format YYYY-MM
        end_month: End month in format YYYY-MM
        **kwargs: Additional arguments for process_comments_file_for_topics
    """
    # Create regex patterns for finding the right files
    pattern = re.compile(r"comments_(\d{4}-\d{2})\.bz2")

    # Get all comment files
    comment_files = sorted(glob(str(comments_dir / "comments_*.bz2")))

    # Filter for files in the date range
    files_to_process = []
    for file_path in comment_files:
        match = pattern.search(file_path)
        if match:
            month = match.group(1)
            if start_month <= month <= end_month:
                files_to_process.append(Path(file_path))

    if not files_to_process:
        print(f"No comment files found in range {start_month} to {end_month}")
        return

    print(
        f"Found {len(files_to_process)} files to process in range {start_month} to {end_month}"
    )

    # Process each file
    for file_path in files_to_process:
        process_comments_file_for_topics(
            comments_file_path=file_path, output_dir=output_dir, **kwargs
        )


def main():
    parser = argparse.ArgumentParser(
        description="Process Reddit comments for topic modeling (LDA) only."
    )
    parser.add_argument(
        "--comments_dir",
        type=Path,
        default=None,
        help="Directory containing comments files (.bz2 or .json).",
    )
    parser.add_argument(
        "--comments_file_path",
        type=Path,
        default=None,
        help="Path to a single comments file (.bz2 or .json).",
    )
    parser.add_argument(
        "--output_dir_name",
        type=str,
        default="comments_topics",
        help="Name of the subdirectory within data/processed/ to save results.",
    )
    parser.add_argument(
        "--start_month",
        type=str,
        default="2016-08",
        help="Start month in format YYYY-MM (when using --comments_dir).",
    )
    parser.add_argument(
        "--end_month",
        type=str,
        default="2017-02",
        help="End month in format YYYY-MM (when using --comments_dir).",
    )
    parser.add_argument(
        "--num_lda_topics",
        type=int,
        default=20,
        help="Number of topics for LDA model. Defaults to 20.",
    )
    parser.add_argument(
        "--lda_passes",
        type=int,
        default=5,
        help="Number of passes for LDA training. Defaults to 5.",
    )
    parser.add_argument(
        "--lda_iterations",
        type=int,
        default=30,
        help="Number of iterations for LDA training. Defaults to 30.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=50000,
        help="Number of comments to process in each chunk. Defaults to 50000.",
    )

    args = parser.parse_args()

    project_root = get_project_root()
    output_base_dir = project_root / "data" / "processed"
    final_output_dir = output_base_dir / args.output_dir_name

    # Ensure output directory exists
    ensure_dir(final_output_dir)

    # Process comments
    if args.comments_file_path:
        # Process a single file
        process_comments_file_for_topics(
            comments_file_path=args.comments_file_path,
            output_dir=final_output_dir,
            num_lda_topics=args.num_lda_topics,
            lda_passes=args.lda_passes,
            lda_iterations=args.lda_iterations,
            chunk_size=args.chunk_size,
        )
    elif args.comments_dir:
        # Process all files in the date range
        comments_dir = args.comments_dir
        process_monthly_comments(
            comments_dir=comments_dir,
            output_dir=final_output_dir,
            start_month=args.start_month,
            end_month=args.end_month,
            num_lda_topics=args.num_lda_topics,
            lda_passes=args.lda_passes,
            lda_iterations=args.lda_iterations,
            chunk_size=args.chunk_size,
        )
    else:
        # Default: use data/comments directory in project
        comments_dir = project_root / "data" / "comments"
        if not comments_dir.exists():
            print(
                f"Comments directory not found at {comments_dir}. Please specify --comments_dir or --comments_file_path."
            )
            return

        process_monthly_comments(
            comments_dir=comments_dir,
            output_dir=final_output_dir,
            start_month=args.start_month,
            end_month=args.end_month,
            num_lda_topics=args.num_lda_topics,
            lda_passes=args.lda_passes,
            lda_iterations=args.lda_iterations,
            chunk_size=args.chunk_size,
        )


if __name__ == "__main__":
    main()
