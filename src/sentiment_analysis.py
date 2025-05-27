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
from topic_modeling import LdaTopicModeler, DEFAULT_SEED_TOPICS


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
        num_lda_topics: int = 15,
        lda_passes: int = 10,
        lda_iterations: int = 50,
    ):
        """
        Initializes the CommentProcessor.

        Args:
            users_metadata_file (Optional[Path]): Path to the user metadata for bot filtering.
            seed_topic_list (List[List[str]]): List of seed keywords for guided topic modeling.
            sentiment_model_device (Optional[str]): "cpu" or "cuda" for sentiment model.
            num_lda_topics (int): Number of topics for LDA. Defaults to 15.
            lda_passes (int): Number of passes for LDA training.
            lda_iterations (int): Number of iterations for LDA training.
        """
        self.bot_user_ids = (
            load_bot_user_ids(users_metadata_file) if users_metadata_file else set()
        )
        self.topic_modeler = LdaTopicModeler(
            seed_topic_list=seed_topic_list,
            num_lda_topics=num_lda_topics,
            passes=lda_passes,
            iterations=lda_iterations,
            lemmatize=True,
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
        Also prints raw LDA topic details for manual review.
        """
        print(f"\n--- Processing {comments_file_path.name} ---")
        start_time = time.time()

        # 1. Load and filter comments
        print("Step 1: Loading and filtering comments...")
        loader = CommentLoader(comments_file_path, bot_user_ids=self.bot_user_ids)
        cols_for_loading = ["id", "author", "subreddit_id", "body_cleaned", "link_id"]
        comments_df = loader.load_and_filter_comments(
            subreddits=subreddits_to_filter, columns_to_keep=cols_for_loading
        )

        if (
            comments_df.empty
            or "body_cleaned" not in comments_df.columns
            or comments_df["body_cleaned"].isnull().all()
            or "link_id" not in comments_df.columns
        ):
            print(
                f"No processable comments (or missing 'body_cleaned'/'link_id') found in {comments_file_path.name} after loading/filtering. Skipping further processing."
            )
            return

        comments_df["body_cleaned"] = comments_df["body_cleaned"].astype(str).fillna("")
        comments_df["link_id"] = comments_df["link_id"].astype(str).fillna("NO_LINK_ID")

        print(f"Loaded {len(comments_df)} comments after filtering.")
        if comments_df.empty:
            print(f"No comments to process for {comments_file_path.name}.")
            return

        print("\nStep 2: Performing LDA topic modeling on aggregated texts...")

        def aggregate_texts(series):
            return " ".join(
                text for text in series if pd.notna(text) and str(text).strip() != ""
            )

        aggregated_texts_df = (
            comments_df.groupby("link_id")["body_cleaned"]
            .apply(aggregate_texts)
            .reset_index()
        )
        aggregated_texts_df.rename(
            columns={"body_cleaned": "merged_body_cleaned"}, inplace=True
        )
        aggregated_texts_df = aggregated_texts_df[
            aggregated_texts_df["merged_body_cleaned"].str.strip() != ""
        ]

        if aggregated_texts_df.empty:
            print("No non-empty threads to model topics for. Assigning default topic.")
            comments_df["topic_label"] = "NO_TOPIC_NO_THREADS"
        else:
            aggregated_texts_with_topics_df = self.topic_modeler.fit_transform(
                aggregated_texts_df, text_column="merged_body_cleaned"
            )
            thread_topics_df = aggregated_texts_with_topics_df[
                ["link_id", "topic_label"]
            ]
            comments_df = pd.merge(
                comments_df, thread_topics_df, on="link_id", how="left"
            )
            comments_df["topic_label"].fillna("NO_TOPIC_THREAD_UNMODELED", inplace=True)

            # --- Print LDA Topic Details for Manual Review ---
            print("\n--- LDA Topic Details for Manual Review ---")
            raw_lda_topics = self.topic_modeler.get_topic_info()
            if raw_lda_topics:
                print("Raw LDA Topics (Top Words):")
                for topic_id, word_probs in raw_lda_topics:
                    words = ", ".join([wp[0] for wp in word_probs])
                    print(f"  LDA Topic {topic_id}: {words}")
            else:
                print("Could not retrieve raw LDA topic information.")

            print("\nAttempted Mapping (LDA ID -> Predefined/Generated Label):")
            mapping = self.topic_modeler.get_lda_to_predefined_mapping()
            if mapping:
                for lda_id, predefined_label in mapping.items():
                    print(f"  LDA Topic {lda_id} -> {predefined_label}")
            else:
                print("No mapping information available.")
            print("--- End LDA Topic Details ---\n")
            # --- End Print LDA Topic Details ---

        # 3. Sentiment Analysis (on individual comments)
        print("\nStep 3: Performing sentiment analysis on individual comments...")
        # Mark empty comments to skip sentiment analysis (original individual comments)
        comments_df["is_empty_comment_body"] = (
            comments_df["body_cleaned"].astype(str).str.strip().eq("")
        )

        # Only perform sentiment analysis on non-empty comments
        non_empty_mask = ~comments_df["is_empty_comment_body"]

        # Initialize sentiment column with None
        comments_df["sentiment"] = None

        if non_empty_mask.any():
            texts_for_sentiment = comments_df.loc[
                non_empty_mask, "body_cleaned"
            ].tolist()
            sentiments = self.sentiment_analyzer.predict_sentiment_batch(
                texts_for_sentiment
            )
            # Assign sentiments only to non-empty comments
            comments_df.loc[non_empty_mask, "sentiment"] = sentiments
            # Convert sentiment to integer if all are non-null after prediction, else keep as float
            if comments_df["sentiment"].notna().all() and non_empty_mask.any():
                comments_df["sentiment"] = comments_df["sentiment"].astype("Int64")

        # Drop the temporary column
        comments_df = comments_df.drop(columns=["is_empty_comment_body"])

        # 4. Finalize DataFrame and Save
        print("\nStep 4: Finalizing and saving results...")
        # Columns to save: id, subreddit_id, link_id, topic_label, sentiment, body_cleaned
        final_columns = [
            "id",
            "subreddit_id",
            "link_id",
            "topic_label",
            "sentiment",
            "body_cleaned",
        ]
        # Ensure all final columns exist, add with NA if missing from processing steps
        for col in final_columns:
            if col not in comments_df.columns:
                comments_df[col] = pd.NA

        final_df = comments_df[final_columns]

        ensure_dir(output_dir)
        if output_filename:
            save_path = output_dir / output_filename
        else:
            base_name = comments_file_path.stem.replace(
                ".json", ""
            )  # e.g., comments_2016-11
            save_path = output_dir / f"{base_name}_lda_topics_sentiment.jsonl"

        final_df.to_json(save_path, orient="records", lines=True, force_ascii=False)
        total_time = time.time() - start_time
        print(f"Successfully processed and saved to {save_path}")
        print(f"Total time for {comments_file_path.name}: {total_time:.2f} seconds.")
        print(f"Sample of final data:\n{final_df.head()}")


def main():
    parser = argparse.ArgumentParser(
        description="Process monthly Reddit comments for topic modeling (LDA) and sentiment analysis."
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
        default="comments_lda_topic_sentiment",
        help="Name of the subdirectory within data/processed/ to save results. Defaults to 'comments_lda_topic_sentiment'.",
    )
    parser.add_argument(
        "--sentiment_device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,  # Auto-detect
        help="Device for sentiment model ('cpu' or 'cuda'). Defaults to auto-detect (cuda if available).",
    )
    parser.add_argument(
        "--num_lda_topics",
        type=int,
        default=15,
        help="Number of topics for LDA model. Defaults to 15.",
    )
    parser.add_argument(
        "--lda_passes",
        type=int,
        default=10,
        help="Number of passes for LDA training. Defaults to 10.",
    )
    parser.add_argument(
        "--lda_iterations",
        type=int,
        default=50,
        help="Number of iterations for LDA training. Defaults to 50.",
    )
    # Potentially add --subreddits_file_path if needed for specific runs

    args = parser.parse_args()

    project_root = get_project_root()
    output_base_dir = project_root / "data" / "processed"
    final_output_dir = output_base_dir / args.output_dir_name

    users_metadata_full_path = None
    if args.users_metadata_file:
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
            users_metadata_full_path = None
    else:
        default_users_metadata_path = (
            project_root / "data" / "metadata" / "users_metadata.jsonl"
        )
        if default_users_metadata_path.exists():
            users_metadata_full_path = default_users_metadata_path
            print(f"Using default users metadata file: {users_metadata_full_path}")
        else:
            print(
                f"Warning: Default users metadata file {default_users_metadata_path} not found. Bot filtering will be skipped."
            )

    processor = CommentProcessor(
        users_metadata_file=users_metadata_full_path,
        seed_topic_list=DEFAULT_SEED_TOPICS,
        sentiment_model_device=args.sentiment_device,
        num_lda_topics=args.num_lda_topics,
        lda_passes=args.lda_passes,
        lda_iterations=args.lda_iterations,
    )

    processor.process_single_month_file(
        comments_file_path=args.comments_file_path, output_dir=final_output_dir
    )


if __name__ == "__main__":
    main()
