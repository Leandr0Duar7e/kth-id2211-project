import argparse
import pandas as pd
from pathlib import Path
import time
from typing import List, Dict, Any, Optional, Set

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
)
import numpy as np
from scipy.special import softmax
import torch  # For checking GPU availability

# Project specific imports
from utils import get_project_root, ensure_dir, load_bot_user_ids
from load_data.load_comments import CommentLoader
from topic_modeling import GuidedTopicModeler, DEFAULT_SEED_TOPICS


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
            self.device = (
                0 if torch.cuda.is_available() else -1
            )  # 0 for cuda:0, -1 for CPU in pipeline
            self.torch_device_name = "cuda" if self.device == 0 else "cpu"
        elif device == "cuda" and torch.cuda.is_available():
            self.device = 0
            self.torch_device_name = "cuda"
        else:
            self.device = -1
            self.torch_device_name = "cpu"

        print(f"Sentiment model will run on: {self.torch_device_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        ).to(self.torch_device_name)
        # For RoBERTa, label 0 is Negative, 1 is Neutral, 2 is Positive.
        # We want to map these to -1 (Negative), 0 (Neutral), 1 (Positive).
        # IMPORTANT: Use lowercase keys to match the model's output labels
        self.label_mapping = {"negative": -1, "neutral": 0, "positive": 1}
        print(
            f"Sentiment model '{self.model_name}' loaded. Label mapping: {self.config.id2label}"
        )
        print(f"Internal mapping to scores: {self.label_mapping}")

        # Validate that our label mapping will work with the model's output labels
        for idx, label in self.config.id2label.items():
            if label not in self.label_mapping:
                print(
                    f"WARNING: Model label '{label}' not found in label_mapping dictionary!"
                )
                print(
                    f"Available keys in label_mapping: {list(self.label_mapping.keys())}"
                )
                print(f"This may cause all sentiments to default to 0 (neutral).")

    def _preprocess_text_for_sentiment(self, text: str) -> str:
        """Basic preprocessing for Twitter-trained models (placeholder for @user, http)."""
        new_text = []
        for t in str(text).split(" "):
            t = "@user" if t.startswith("@") and len(t) > 1 else t
            t = "http" if t.startswith("http") else t
            new_text.append(t)
        return " ".join(new_text)

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

        # Use a fixed max_length for tokenization to avoid integer overflow errors
        # RoBERTa base models typically use 512, but 128 is often sufficient for sentiment analysis
        max_length = 128

        # Track distribution of sentiment predictions for debugging
        sentiment_distribution = {
            "negative": 0,
            "neutral": 0,
            "positive": 0,
            "unknown": 0,
        }

        for i in range(0, num_texts, batch_size):
            batch_texts = texts[i : i + batch_size]
            preprocessed_batch = [
                self._preprocess_text_for_sentiment(text) for text in batch_texts
            ]

            encoded_input = self.tokenizer(
                preprocessed_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,  # Use fixed max_length
            ).to(self.torch_device_name)

            with torch.no_grad():
                output = self.model(**encoded_input)

            batch_scores = output.logits.detach().cpu().numpy()
            batch_scores = softmax(batch_scores, axis=1)

            predicted_labels_indices = np.argmax(batch_scores, axis=1)

            for label_idx in predicted_labels_indices:
                hf_label = self.config.id2label[
                    label_idx
                ]  # e.g., 'negative', 'neutral', 'positive'

                # Ensure case matching (model output is lowercase)
                sentiment_val = self.label_mapping.get(hf_label.lower(), None)

                if sentiment_val is not None:
                    sentiments.append(sentiment_val)
                    sentiment_distribution[hf_label.lower()] += 1
                else:
                    # If mapping fails, default to neutral but log the issue
                    print(
                        f"WARNING: Unknown sentiment label '{hf_label}'. Using neutral (0) as default."
                    )
                    sentiments.append(0)  # Default to Neutral
                    sentiment_distribution["unknown"] += 1

            if (i // batch_size + 1) % 10 == 0:  # Print progress every 10 batches
                print(
                    f"  Processed {i + len(batch_texts)} / {num_texts} texts for sentiment."
                )

        # Print distribution of detected sentiments
        print("Sentiment prediction complete.")
        print(f"Sentiment distribution from model: {sentiment_distribution}")
        sentiment_counts = {-1: 0, 0: 0, 1: 0}
        for s in sentiments:
            sentiment_counts[s] = sentiment_counts.get(s, 0) + 1
        print(f"Final sentiment value counts: {sentiment_counts}")

        return sentiments


class CommentProcessor:
    """Orchestrates loading, topic modeling, and sentiment analysis for comments."""

    def __init__(
        self,
        users_metadata_file: Optional[Path] = None,
        seed_topic_list: List[List[str]] = DEFAULT_SEED_TOPICS,
        sentiment_model_device: Optional[str] = None,
    ):
        """
        Initializes the CommentProcessor.

        Args:
            users_metadata_file (Optional[Path]): Path to the user metadata for bot filtering.
            seed_topic_list (List[List[str]]): List of seed keywords for guided topic modeling.
            sentiment_model_device (Optional[str]): "cpu" or "cuda" for sentiment model.
        """
        self.bot_user_ids = (
            load_bot_user_ids(users_metadata_file) if users_metadata_file else set()
        )
        selftopic_modeler = GuidedTopicModeler(
            seed_topic_list=seed_topic_list,
            min_topic_size=15,  # Higher minimum size
            nr_topics=25,  # cap on total topics (seed topics + some emergent ones)
            disable_reduction_for_small_datasets=False,
        )
        self.sentiment_analyzer = SentimentAnalyzer(device=sentiment_model_device)
        self.project_root = get_project_root()

    def process_single_month_file(
        self,
        comments_file_path: Path,
        output_dir: Path,
        subreddits_to_filter: Optional[Set[str]] = None,
        output_filename: Optional[str] = None,
    ):
        """
        Processes a single monthly comment file: loads, filters, topics, sentiments, and saves.

        Args:
            comments_file_path (Path): Path to the monthly comments bz2 file.
            output_dir (Path): Directory to save the processed file.
            subreddits_to_filter (Optional[Set[str]]): Set of subreddits to focus on. Defaults to None.
            output_filename (Optional[str]): Custom name for the output file. Defaults to None.
        """
        print(f"\n--- Processing {comments_file_path.name} ---")
        start_time = time.time()

        # 1. Load and filter comments
        print("Step 1: Loading and filtering comments...")
        loader = CommentLoader(comments_file_path, bot_user_ids=self.bot_user_ids)
        # Columns needed: id, subreddit_id, body_cleaned (for topic/sentiment), author (for bot filtering)
        cols_for_loading = ["id", "author", "subreddit_id", "body_cleaned"]
        comments_df = loader.load_and_filter_comments(
            subreddits=subreddits_to_filter, columns_to_keep=cols_for_loading
        )

        if (
            comments_df.empty
            or "body_cleaned" not in comments_df.columns
            or comments_df["body_cleaned"].isnull().all()
        ):
            print(
                f"No processable comments found in {comments_file_path.name} after loading/filtering. Skipping further processing."
            )
            return

        # Keep only necessary columns for further processing to save memory
        comments_df = comments_df[["id", "subreddit_id", "body_cleaned"]]
        comments_df["body_cleaned"] = (
            comments_df["body_cleaned"].astype(str).fillna("")
        )  # Ensure string type and handle NaNs

        print(f"Loaded {len(comments_df)} comments after filtering.")
        if comments_df.empty:
            print(f"No comments to process for {comments_file_path.name}.")
            return

        # 2. Topic Modeling
        print("\nStep 2: Performing guided topic modeling...")
        comments_df, topic_map = self.topic_modeler.fit_transform(
            comments_df, text_column="body_cleaned"
        )
        # topic_map can be saved or logged if needed: print(f"Topic ID to Label Map: {topic_map}")

        # 3. Sentiment Analysis
        print("\nStep 3: Performing sentiment analysis...")
        # Use 'body_cleaned' for sentiment as it's the cleaned text available
        texts_for_sentiment = comments_df["body_cleaned"].tolist()
        sentiments = self.sentiment_analyzer.predict_sentiment_batch(
            texts_for_sentiment
        )
        comments_df["sentiment"] = sentiments

        # 4. Finalize DataFrame and Save
        print("\nStep 4: Finalizing and saving results...")
        # Columns to save: id, subreddit_id, topic_label (more readable), sentiment
        final_df = comments_df[["id", "subreddit_id", "topic_label", "sentiment"]]

        ensure_dir(output_dir)
        if output_filename:
            save_path = output_dir / output_filename
        else:
            base_name = comments_file_path.stem.replace(
                ".json", ""
            )  # e.g., comments_2016-11
            save_path = output_dir / f"{base_name}_topics_sentiment.jsonl"

        final_df.to_json(save_path, orient="records", lines=True, force_ascii=False)
        total_time = time.time() - start_time
        print(f"Successfully processed and saved to {save_path}")
        print(f"Total time for {comments_file_path.name}: {total_time:.2f} seconds.")
        print(f"Sample of final data:\n{final_df.head()}")


def main():
    parser = argparse.ArgumentParser(
        description="Process monthly Reddit comments for topic modeling and sentiment analysis."
    )
    parser.add_argument(
        "--comments_file_path",
        type=Path,
        required=True,
        help="Path to a single monthly comments bz2 file (e.g., data/comments/comments_2016-11.bz2).",
    )
    parser.add_argument(
        "--users_metadata_file",
        type=Path,
        default=None,  # Optional
        help="Path to the users_metadata.jsonl file for bot identification.",
    )
    parser.add_argument(
        "--output_dir_name",
        type=str,
        default="comments_topic_sentiment",
        help="Name of the subdirectory within data/processed/ to save results. Defaults to 'comments_topic_sentiment'.",
    )
    parser.add_argument(
        "--sentiment_device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,  # Auto-detect
        help="Device for sentiment model ('cpu' or 'cuda'). Defaults to auto-detect (cuda if available).",
    )
    # Potentially add --subreddits_file_path if needed for specific runs

    args = parser.parse_args()

    project_root = get_project_root()
    output_base_dir = project_root / "data" / "processed"
    final_output_dir = output_base_dir / args.output_dir_name

    users_metadata_full_path = None
    if args.users_metadata_file:
        # If a relative path is given, assume it's relative to data/metadata/
        if not args.users_metadata_file.is_absolute():
            users_metadata_full_path = (
                project_root / "data" / "metadata" / args.users_metadata_file
            )
        else:
            users_metadata_full_path = args.users_metadata_file
        if not users_metadata_full_path.exists():
            print(
                f"Warning: Provided users metadata file {users_metadata_full_path} does not exist. Bot filtering will be skipped."
            )
            users_metadata_full_path = None  # Reset to skip
    else:  # Default path if not provided
        default_users_metadata_path = (
            project_root / "data" / "metadata" / "users_metadata.jsonl"
        )  # or .json if that's the actual name
        if default_users_metadata_path.exists():
            users_metadata_full_path = default_users_metadata_path
            print(f"Using default users metadata file: {users_metadata_full_path}")
        else:
            print(
                f"Warning: Default users metadata file {default_users_metadata_path} not found. Bot filtering will be skipped."
            )

    processor = CommentProcessor(
        users_metadata_file=users_metadata_full_path,
        seed_topic_list=DEFAULT_SEED_TOPICS,  # Using the default from topic_modeling.py
        sentiment_model_device=args.sentiment_device,
    )

    processor.process_single_month_file(
        comments_file_path=args.comments_file_path, output_dir=final_output_dir
    )


if __name__ == "__main__":
    main()
