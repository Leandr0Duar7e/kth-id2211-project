import argparse
import pandas as pd
from pathlib import Path
import time
import json
import os
import gc  # For garbage collection
from glob import glob
import re
import torch

# Project specific imports
from utils import get_project_root, ensure_dir
from sentiment_analysis import SentimentAnalyzer


def process_comments_file_directly(
    comments_file_path: Path,
    output_dir: Path,
    output_filename: str = None,
    batch_size: int = 64,
    device: str = None,
    chunk_size: int = 50000,  # Process in chunks
):
    """
    Process a comments file directly for sentiment analysis, without requiring topics.

    Args:
        comments_file_path: Path to the comments file (.bz2 or .json)
        output_dir: Directory to save results
        output_filename: Optional custom filename for output
        batch_size: Batch size for sentiment analysis
        device: Device to run sentiment analysis on (cpu/cuda)
        chunk_size: Number of comments to process in each chunk
    """
    print(f"\n--- Processing {comments_file_path.name} directly for sentiment ---")
    start_time = time.time()

    # Create output directory
    ensure_dir(output_dir)

    # Determine output filename
    if output_filename:
        save_path = output_dir / output_filename
    else:
        base_name = comments_file_path.stem.replace(".json", "")
        save_path = output_dir / f"{base_name}_sentiment.jsonl"

    # Check if file already exists and remove if it does
    if save_path.exists():
        os.remove(save_path)
        print(f"Removed existing output file: {save_path}")

    # Initialize sentiment analyzer
    sentiment_analyzer = SentimentAnalyzer(device=device)

    # Initialize chunking parameters
    total_comments = 0
    chunk_counter = 0

    # Process in chunks
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

        for chunk_df in chunk_reader:
            chunk_counter += 1
            total_comments += len(chunk_df)
            print(f"Processing chunk {chunk_counter} with {len(chunk_df)} comments...")

            # Use body_cleaned if it exists, otherwise use body
            if "body_cleaned" not in chunk_df.columns:
                if "body" in chunk_df.columns:
                    chunk_df["body_cleaned"] = chunk_df["body"].astype(str).fillna("")
                else:
                    print(
                        "Error: Neither 'body_cleaned' nor 'body' found in comments file."
                    )
                    continue

            # Skip empty comments for sentiment analysis
            chunk_df["is_empty"] = (
                chunk_df["body_cleaned"].astype(str).str.strip().eq("")
            )
            non_empty_mask = ~chunk_df["is_empty"]

            # Initialize sentiment column with None
            chunk_df["sentiment"] = None

            # Only run sentiment analysis if we have non-empty comments
            if non_empty_mask.any():
                texts_for_sentiment = chunk_df.loc[
                    non_empty_mask, "body_cleaned"
                ].tolist()
                try:
                    sentiments = sentiment_analyzer.predict_sentiment_batch(
                        texts_for_sentiment, batch_size=batch_size
                    )
                    chunk_df.loc[non_empty_mask, "sentiment"] = sentiments
                except Exception as e:
                    print(f"Error during sentiment analysis: {e}")
                    # Continue with processing, but mark sentiments as None

            # Clean up
            chunk_df = chunk_df.drop(columns=["is_empty"])

            # Select columns to save
            final_columns = [
                "id",
                "link_id",
                "author",
                "sentiment",
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
        print(f"Total time: {total_time:.2f} seconds.")

    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback

        traceback.print_exc()
        return


def process_monthly_comments_directly(
    comments_dir: Path,
    output_dir: Path,
    start_month: str = "2016-08",
    end_month: str = "2017-02",
    **kwargs,
):
    """
    Process all monthly comment files in the given date range directly for sentiment.

    Args:
        comments_dir: Directory containing comments files
        output_dir: Directory to save sentiment analysis results
        start_month: Start month in format YYYY-MM
        end_month: End month in format YYYY-MM
        **kwargs: Additional arguments for process_comments_file_directly
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
        process_comments_file_directly(
            comments_file_path=file_path, output_dir=output_dir, **kwargs
        )


def process_file_for_sentiment(
    input_file_path: Path,
    output_dir: Path,
    output_filename: str = None,
    batch_size: int = 64,
    device: str = None,
    chunk_size: int = 50000,  # Process in chunks
):
    """
    Adds sentiment analysis to a file that already has topic assignments.

    Args:
        input_file_path: Path to the input file with topic assignments
        output_dir: Directory to save results
        output_filename: Optional custom filename for output
        batch_size: Batch size for sentiment analysis
        device: Device to run sentiment analysis on (cpu/cuda)
        chunk_size: Number of comments to process in each chunk
    """
    print(f"\n--- Processing {input_file_path.name} for sentiment ---")
    start_time = time.time()

    # Create output directory
    ensure_dir(output_dir)

    # Determine output filename
    if output_filename:
        save_path = output_dir / output_filename
    else:
        base_name = input_file_path.stem.replace("_topics", "")
        save_path = output_dir / f"{base_name}_topics_sentiment.jsonl"

    # Check if file already exists and remove if it does
    if save_path.exists():
        os.remove(save_path)
        print(f"Removed existing output file: {save_path}")

    # Initialize sentiment analyzer
    sentiment_analyzer = SentimentAnalyzer(device=device)

    # Initialize chunking parameters
    total_comments = 0
    chunk_counter = 0

    # Process in chunks
    try:
        # Set up chunk reader
        chunk_reader = pd.read_json(input_file_path, lines=True, chunksize=chunk_size)

        for chunk_df in chunk_reader:
            chunk_counter += 1
            total_comments += len(chunk_df)
            print(f"Processing chunk {chunk_counter} with {len(chunk_df)} comments...")

            # Skip empty comments for sentiment analysis
            chunk_df["is_empty"] = (
                chunk_df["body_cleaned"].astype(str).str.strip().eq("")
            )
            non_empty_mask = ~chunk_df["is_empty"]

            # Initialize sentiment column with None
            chunk_df["sentiment"] = None

            # Only run sentiment analysis if we have non-empty comments
            if non_empty_mask.any():
                texts_for_sentiment = chunk_df.loc[
                    non_empty_mask, "body_cleaned"
                ].tolist()
                try:
                    sentiments = sentiment_analyzer.predict_sentiment_batch(
                        texts_for_sentiment, batch_size=batch_size
                    )
                    chunk_df.loc[non_empty_mask, "sentiment"] = sentiments
                except Exception as e:
                    print(f"Error during sentiment analysis: {e}")
                    # Continue with processing, but mark sentiments as None

            # Clean up
            chunk_df = chunk_df.drop(columns=["is_empty"])

            # Append to output file
            if chunk_counter == 1:  # First chunk, create the file
                chunk_df.to_json(
                    save_path, orient="records", lines=True, force_ascii=False
                )
            else:  # Append to existing file
                with open(save_path, "a", encoding="utf-8") as f:
                    chunk_df.to_json(f, orient="records", lines=True, force_ascii=False)

            # Free up memory
            del chunk_df
            gc.collect()

        total_time = time.time() - start_time
        print(f"Successfully processed and saved to {save_path}")
        print(f"Total comments processed: {total_comments}")
        print(f"Total time: {total_time:.2f} seconds.")

    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback

        traceback.print_exc()
        return


def process_all_topic_files(
    topics_dir: Path, output_dir: Path, file_pattern: str = "*_topics.jsonl", **kwargs
):
    """
    Process all topic files in the given directory.

    Args:
        topics_dir: Directory containing topic files
        output_dir: Directory to save sentiment analysis results
        file_pattern: Pattern to match topic files
        **kwargs: Additional arguments for process_file_for_sentiment
    """
    # Get all topic files
    topic_files = sorted(glob(str(topics_dir / file_pattern)))

    if not topic_files:
        print(f"No topic files found matching pattern {file_pattern} in {topics_dir}")
        return

    print(f"Found {len(topic_files)} topic files to process")

    # Process each file
    for file_path in topic_files:
        process_file_for_sentiment(
            input_file_path=Path(file_path), output_dir=output_dir, **kwargs
        )


def main():
    parser = argparse.ArgumentParser(
        description="Process Reddit comments for sentiment analysis, with or without topic modeling."
    )
    parser.add_argument(
        "--direct_mode",
        action="store_true",
        help="Process comments files directly for sentiment without requiring topic files.",
    )
    parser.add_argument(
        "--comments_dir",
        type=Path,
        default=None,
        help="Directory containing raw comments files (.bz2 or .json) for direct mode.",
    )
    parser.add_argument(
        "--comments_file_path",
        type=Path,
        default=None,
        help="Path to a single comments file (.bz2 or .json) for direct mode.",
    )
    parser.add_argument(
        "--start_month",
        type=str,
        default="2016-08",
        help="Start month in format YYYY-MM (when using --comments_dir in direct mode).",
    )
    parser.add_argument(
        "--end_month",
        type=str,
        default="2017-02",
        help="End month in format YYYY-MM (when using --comments_dir in direct mode).",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=None,
        help="Directory containing topic-modeled comments files.",
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        default=None,
        help="Path to a single topic-modeled comments file.",
    )
    parser.add_argument(
        "--output_dir_name",
        type=str,
        default=None,
        help="Name of the subdirectory within data/processed/ to save results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Device for sentiment model ('cpu' or 'cuda'). Defaults to cuda if available.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for sentiment analysis. Larger values may be faster but use more GPU memory. Defaults to 64.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=50000,
        help="Number of comments to process in each chunk. Defaults to 50000.",
    )
    parser.add_argument(
        "--file_pattern",
        type=str,
        default="*_topics.jsonl",
        help="Pattern for matching topic files in input_dir.",
    )

    args = parser.parse_args()

    # Check CUDA availability and print info
    if torch.cuda.is_available():
        print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available, will use CPU")
        if args.device == "cuda":
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            args.device = "cpu"

    project_root = get_project_root()
    output_base_dir = project_root / "data" / "processed"

    # Set default output directory name based on mode
    if args.output_dir_name is None:
        if args.direct_mode:
            args.output_dir_name = "comments_sentiment"
        else:
            args.output_dir_name = "comments_topics_sentiment"

    final_output_dir = output_base_dir / args.output_dir_name

    # Ensure output directory exists
    ensure_dir(final_output_dir)

    # Handle direct mode (process comments files directly)
    if args.direct_mode:
        if args.comments_file_path:
            # Process a single comments file directly
            process_comments_file_directly(
                comments_file_path=args.comments_file_path,
                output_dir=final_output_dir,
                device=args.device,
                batch_size=args.batch_size,
                chunk_size=args.chunk_size,
            )
        elif args.comments_dir:
            # Process all comments files in the directory
            process_monthly_comments_directly(
                comments_dir=args.comments_dir,
                output_dir=final_output_dir,
                start_month=args.start_month,
                end_month=args.end_month,
                device=args.device,
                batch_size=args.batch_size,
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

            process_monthly_comments_directly(
                comments_dir=comments_dir,
                output_dir=final_output_dir,
                start_month=args.start_month,
                end_month=args.end_month,
                device=args.device,
                batch_size=args.batch_size,
                chunk_size=args.chunk_size,
            )
    # Handle traditional mode (process topic files)
    else:
        if args.input_file:
            # Process a single topic file
            process_file_for_sentiment(
                input_file_path=args.input_file,
                output_dir=final_output_dir,
                device=args.device,
                batch_size=args.batch_size,
                chunk_size=args.chunk_size,
            )
        elif args.input_dir:
            # Process all topic files in the directory
            process_all_topic_files(
                topics_dir=args.input_dir,
                output_dir=final_output_dir,
                file_pattern=args.file_pattern,
                device=args.device,
                batch_size=args.batch_size,
                chunk_size=args.chunk_size,
            )
        else:
            # Default: look in the comments_topics directory
            topics_dir = project_root / "data" / "processed" / "comments_topics"
            if not topics_dir.exists():
                print(
                    f"Topics directory not found at {topics_dir}. Please specify --input_dir or --input_file or use --direct_mode."
                )
                return

            process_all_topic_files(
                topics_dir=topics_dir,
                output_dir=final_output_dir,
                file_pattern=args.file_pattern,
                device=args.device,
                batch_size=args.batch_size,
                chunk_size=args.chunk_size,
            )


if __name__ == "__main__":
    main()
