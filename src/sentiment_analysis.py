import argparse
import pandas as pd
from pathlib import Path
import time
from typing import List, Dict, Any, Optional, Set
import json

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
    ):
        """
        Processes a comments file: loads, topics, sentiments, and saves.
        """
        print(f"\n--- Processing {comments_file_path.name} ---")
        start_time = time.time()

        # 1. Load comments
        print("Step 1: Loading comments...")
        if comments_file_path.suffix == ".bz2":
            comments_df = pd.read_json(
                comments_file_path, lines=True, compression="bz2"
            )
        else:
            comments_df = pd.read_json(comments_file_path, lines=True)

        # Ensure required columns exist
        required_cols = ["id", "link_id"]
        for col in required_cols:
            if col not in comments_df.columns:
                print(f"Error: Required column '{col}' not found in comments file.")
                return

        # Use body_cleaned if it exists, otherwise use body
        if "body_cleaned" not in comments_df.columns:
            if "body" in comments_df.columns:
                comments_df["body_cleaned"] = comments_df["body"].astype(str).fillna("")
            else:
                print(
                    "Error: Neither 'body_cleaned' nor 'body' found in comments file."
                )
                return

        print(f"Loaded {len(comments_df)} comments.")

        # 2. Perform topic modeling on aggregated texts
        print("\nStep 2: Performing LDA topic modeling on aggregated texts...")

        # Aggregate comments by link_id
        aggregated_texts_df = (
            comments_df.groupby("link_id")["body_cleaned"]
            .apply(
                lambda x: " ".join(
                    str(text)
                    for text in x
                    if pd.notna(text) and str(text).strip() != ""
                )
            )
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
            comments_df["topic_id"] = -1
        else:
            # Apply topic modeling
            aggregated_texts_with_topics_df = self.topic_modeler.fit_transform(
                aggregated_texts_df, text_column="merged_body_cleaned"
            )

            # Merge topics back to comments
            thread_topics_df = aggregated_texts_with_topics_df[["link_id", "topic_id"]]
            comments_df = pd.merge(
                comments_df, thread_topics_df, on="link_id", how="left"
            )
            comments_df["topic_id"] = comments_df["topic_id"].fillna(-1).astype(int)

            # Print topics summary
            print("\n--- LDA Topic Summary ---")
            topic_words = self.topic_modeler.get_topic_words()
            for topic_id, words in topic_words.items():
                print(f"Topic {topic_id}: {words}")
            print("--- End LDA Topic Summary ---\n")

        # 3. Sentiment Analysis (on individual comments)
        print("\nStep 3: Performing sentiment analysis on individual comments...")

        # Skip empty comments
        comments_df["is_empty"] = (
            comments_df["body_cleaned"].astype(str).str.strip().eq("")
        )
        non_empty_mask = ~comments_df["is_empty"]

        # Initialize sentiment column with None
        comments_df["sentiment"] = None

        if non_empty_mask.any():
            texts_for_sentiment = comments_df.loc[
                non_empty_mask, "body_cleaned"
            ].tolist()
            sentiments = self.sentiment_analyzer.predict_sentiment_batch(
                texts_for_sentiment, batch_size=64
            )
            comments_df.loc[non_empty_mask, "sentiment"] = sentiments

        # Clean up
        comments_df = comments_df.drop(columns=["is_empty"])

        # 4. Save results
        print("\nStep 4: Finalizing and saving results...")

        # Select columns to save
        final_columns = ["id", "link_id", "topic_id", "sentiment", "body_cleaned"]
        for col in final_columns:
            if col not in comments_df.columns:
                comments_df[col] = None

        final_df = comments_df[final_columns]

        # Create output directory and save
        ensure_dir(output_dir)
        if output_filename:
            save_path = output_dir / output_filename
        else:
            base_name = comments_file_path.stem.replace(".json", "")
            save_path = output_dir / f"{base_name}_topics_sentiment.jsonl"

        final_df.to_json(save_path, orient="records", lines=True, force_ascii=False)

        total_time = time.time() - start_time
        print(f"Successfully processed and saved to {save_path}")
        print(f"Total time: {total_time:.2f} seconds.")
        print(f"Sample of final data:\n{final_df.head()}")


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
        comments_file_path=args.comments_file_path, output_dir=final_output_dir
    )


if __name__ == "__main__":
    main()
