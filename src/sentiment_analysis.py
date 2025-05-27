import argparse
import pandas as pd
from pathlib import Path
import time
from typing import List, Dict, Any, Optional, Set
import json
import os
import gc  # For garbage collection

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import numpy as np
from scipy.special import softmax
import torch  # For checking GPU availability

# Project specific imports
from utils import get_project_root, ensure_dir
from topic_modeling import SimpleLdaTopicModeler


class SentimentAnalyzer:
    """Performs sentiment analysis using a Hugging Face transformer model."""

    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        device: Optional[str] = None,
    ):
        """
        Initializes the SentimentAnalyzer.

        Args:
            model_name (str, optional): Name of the Hugging Face model.
                                        Defaults to "cardiffnlp/twitter-roberta-base-sentiment-latest".
            device (Optional[str], optional): Device to run the model on ("cpu", "cuda").
                                             Defaults to "cuda" if available, else "cpu".
        """
        self.model_name = model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = (
                device if device == "cuda" and torch.cuda.is_available() else "cpu"
            )

        print(f"Sentiment model will run on: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.config = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        ).config
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        ).to(self.device)
        # For RoBERTa, label 0 is Negative, 1 is Neutral, 2 is Positive.
        # We want to map these to -1 (Negative), 0 (Neutral), 1 (Positive).
        self.label_mapping = {"negative": -1, "neutral": 0, "positive": 1}
        print(
            f"Sentiment model '{self.model_name}' loaded. Label mapping: {self.config.id2label}"
        )

    def predict_sentiment_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> List[int]:
        """
        Predicts sentiment for a batch of texts.

        Args:
            texts (List[str]): A list of text strings.
            batch_size (int): Number of texts to process in one batch.

        Returns:
            List[int]: A list of sentiment scores (-1, 0, or 1).
        """
        sentiments = []
        num_texts = len(texts)
        if num_texts == 0:
            return []

        print(
            f"Predicting sentiment for {num_texts} texts in batches of {batch_size}..."
        )

        # Use a shorter max_length for faster processing
        max_length = 128

        for i in range(0, num_texts, batch_size):
            batch_texts = texts[i : i + batch_size]

            # No preprocessing needed since text is already processed
            encoded_input = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)

            with torch.no_grad():
                output = self.model(**encoded_input)

            batch_scores = output.logits.detach().cpu().numpy()
            batch_scores = softmax(batch_scores, axis=1)

            predicted_labels_indices = np.argmax(batch_scores, axis=1)

            for label_idx in predicted_labels_indices:
                hf_label = self.config.id2label[label_idx].lower()
                sentiment_val = self.label_mapping.get(
                    hf_label, 0
                )  # Default to neutral
                sentiments.append(sentiment_val)

            if (i // batch_size + 1) % 20 == 0:  # Print progress less frequently
                print(
                    f"  Processed {i + len(batch_texts)} / {num_texts} texts for sentiment."
                )

        return sentiments


class CommentProcessor:
    """Orchestrates loading, topic modeling, and sentiment analysis for comments."""

    def __init__(
        self,
        sentiment_model_device: Optional[str] = None,
        num_lda_topics: int = 20,
        lda_passes: int = 5,
        lda_iterations: int = 30,
    ):
        """
        Initializes the CommentProcessor.

        Args:
            sentiment_model_device (Optional[str]): "cpu" or "cuda" for sentiment model.
            num_lda_topics (int): Number of topics for LDA. Defaults to 20.
            lda_passes (int): Number of passes for LDA training.
            lda_iterations (int): Number of iterations for LDA training.
        """
        self.topic_modeler = SimpleLdaTopicModeler(
            num_topics=num_lda_topics,
            passes=lda_passes,
            iterations=lda_iterations,
            lemmatize=False,  # No lemmatization since text is already processed
        )
        self.sentiment_analyzer = SentimentAnalyzer(device=sentiment_model_device)
        self.project_root = get_project_root()

    def process_comments_file(
        self,
        comments_file_path: Path,
        output_dir: Path,
        output_filename: Optional[str] = None,
        chunk_size: int = 50000,  # Process in chunks of 50k comments
    ):
        """
        Processes a comments file: loads, topics, sentiments, and saves.
        Now processes data in chunks to reduce memory usage.

        Args:
            comments_file_path: Path to the comments file (.bz2 or .json)
            output_dir: Directory to save results
            output_filename: Optional custom filename for output
            chunk_size: Number of comments to process in each chunk
        """
        print(f"\n--- Processing {comments_file_path.name} ---")
        start_time = time.time()

        # Create output directory
        ensure_dir(output_dir)

        # Determine output filename
        if output_filename:
            save_path = output_dir / output_filename
        else:
            base_name = comments_file_path.stem.replace(".json", "")
            save_path = output_dir / f"{base_name}_topics_sentiment.jsonl"

        # Check if file already exists and remove if it does
        if save_path.exists():
            os.remove(save_path)
            print(f"Removed existing output file: {save_path}")

        # Initialize chunking parameters
        total_comments = 0
        chunk_counter = 0
        total_chunks_processed = 0

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
                    print(
                        "Error: Required column 'link_id' not found in comments file."
                    )
                    return

                # Use body_cleaned if it exists, otherwise use body
                if "body_cleaned" not in chunk_df.columns:
                    if "body" in chunk_df.columns:
                        chunk_df["body_cleaned"] = (
                            chunk_df["body"].astype(str).fillna("")
                        )
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
                print(
                    "No non-empty threads to model topics for. Assigning default topic."
                )
                thread_topics_df = pd.DataFrame(columns=["link_id", "topic_id"])
            else:
                # Apply topic modeling
                aggregated_texts_with_topics_df = self.topic_modeler.fit_transform(
                    aggregated_texts_df, text_column="merged_body_cleaned"
                )

                # Get thread topics mapping
                thread_topics_df = aggregated_texts_with_topics_df[
                    ["link_id", "topic_id"]
                ]

                # Print topics summary
                print("\n--- LDA Topic Summary ---")
                topic_words = self.topic_modeler.get_topic_words()
                for topic_id, words in topic_words.items():
                    print(f"Topic {topic_id}: {words}")
                print("--- End LDA Topic Summary ---\n")

            # Free up memory
            del aggregated_texts_df
            gc.collect()

            # Second pass: Process each chunk for sentiment and merge with topic assignments
            print("\nProcessing chunks for sentiment analysis and saving results...")
            chunk_counter = 0

            for chunk_df in chunk_reader:
                chunk_counter += 1
                total_chunks_processed += 1
                print(
                    f"Processing chunk {chunk_counter} with {len(chunk_df)} comments..."
                )

                # Ensure body_cleaned exists
                if "body_cleaned" not in chunk_df.columns:
                    if "body" in chunk_df.columns:
                        chunk_df["body_cleaned"] = (
                            chunk_df["body"].astype(str).fillna("")
                        )
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
                        sentiments = self.sentiment_analyzer.predict_sentiment_batch(
                            texts_for_sentiment, batch_size=64
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
                    "topic_id",
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
                        final_df.to_json(
                            f, orient="records", lines=True, force_ascii=False
                        )

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


def main():
    parser = argparse.ArgumentParser(
        description="Process Reddit comments for topic modeling (LDA) and sentiment analysis."
    )
    parser.add_argument(
        "--comments_file_path",
        type=Path,
        required=True,
        help="Path to comments file (.bz2 or .json).",
    )
    parser.add_argument(
        "--output_dir_name",
        type=str,
        default="comments_topic_sentiment",
        help="Name of the subdirectory within data/processed/ to save results.",
    )
    parser.add_argument(
        "--sentiment_device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Device for sentiment model ('cpu' or 'cuda'). Defaults to auto-detect.",
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
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for sentiment analysis. Larger values may be faster but use more memory. Defaults to 64.",
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

    processor = CommentProcessor(
        sentiment_model_device=args.sentiment_device,
        num_lda_topics=args.num_lda_topics,
        lda_passes=args.lda_passes,
        lda_iterations=args.lda_iterations,
    )

    processor.process_comments_file(
        comments_file_path=args.comments_file_path,
        output_dir=final_output_dir,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
