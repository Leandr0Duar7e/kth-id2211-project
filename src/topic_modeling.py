import pandas as pd
from bertopic import BERTopic
from typing import List, Dict, Tuple, Optional


class GuidedTopicModeler:
    """Performs guided topic modeling using BERTopic."""

    def __init__(
        self,
        seed_topic_list: List[List[str]],
        language: str = "english",
        top_n_words: int = 10,
        min_topic_size: int = 10,
        disable_reduction_for_small_datasets: bool = True,
        **kwargs,
    ):
        """
        Initializes the GuidedTopicModeler.

        Args:
            seed_topic_list (List[List[str]]): A list of lists, where each inner list contains seed keywords for a topic.
            language (str, optional): Language for BERTopic. Defaults to "english".
            top_n_words (int, optional): Number of words per topic. Defaults to 10.
            min_topic_size (int, optional): Minimum size of a topic. Defaults to 10.
            disable_reduction_for_small_datasets (bool, optional): If True, disables UMAP and other
                                                   dimensionality reduction for small datasets. Defaults to True.
            **kwargs: Additional arguments to pass to BERTopic.
        """
        self.seed_topic_list = seed_topic_list

        # For small datasets, disable UMAP and other dimensionality reduction components
        # These can cause issues with smaller samples
        if disable_reduction_for_small_datasets:
            print(
                "Small dataset mode: Disabling UMAP and certain components for better topic detection"
            )
            # Set reduced clustering parameters for small datasets
            self.model = BERTopic(
                seed_topic_list=self.seed_topic_list,
                language=language,
                top_n_words=top_n_words,
                min_topic_size=min_topic_size,
                umap_model=None,  # Skip dimensionality reduction
                calculate_probabilities=False,  # Speed up the process
                verbose=True,
                **kwargs,
            )
        else:
            # Standard configuration for larger datasets
            self.model = BERTopic(
                seed_topic_list=self.seed_topic_list,
                language=language,
                top_n_words=top_n_words,
                min_topic_size=min_topic_size,
                verbose=True,
                **kwargs,
            )

        self.topic_mapping: Optional[Dict[int, str]] = None

    def fit_transform(
        self, comments_df: pd.DataFrame, text_column: str = "body_cleaned"
    ) -> Tuple[pd.DataFrame, Optional[Dict[int, str]]]:
        """
        Fits the BERTopic model to the comments and assigns topics.

        Args:
            comments_df (pd.DataFrame): DataFrame containing comments, with at least `text_column` and `id`.
            text_column (str, optional): The name of the column containing the text to model.
                                       Defaults to "body_cleaned".

        Returns:
            Tuple[pd.DataFrame, Optional[Dict[int, str]]]:
                - DataFrame with an added 'topic_id' column and potentially 'topic_label'.
                - A dictionary mapping topic IDs to their descriptive labels (based on seed keywords).
        """
        if (
            comments_df.empty
            or text_column not in comments_df.columns
            or comments_df[text_column].isnull().all()
        ):
            print(
                "Warning: Comments DataFrame is empty, lacks the text column, or text column has all null values. Skipping topic modeling."
            )
            comments_df["topic_id"] = -2  # Special ID for skipped/empty
            comments_df["topic_label"] = "NO_TOPIC_EMPTY_INPUT"
            return comments_df, None

        # Mark empty comments before processing
        comments_df["is_empty_comment"] = (
            comments_df[text_column].astype(str).str.strip().eq("")
        )

        # Create a copy of the dataframe with only non-empty comments for modeling
        non_empty_df = comments_df[~comments_df["is_empty_comment"]].copy()

        if non_empty_df.empty:
            print("Warning: All comments are empty. Skipping topic modeling.")
            comments_df["topic_id"] = -2  # Special ID for skipped/empty
            comments_df["topic_label"] = "EMPTY_COMMENT"
            return comments_df, None

        # BERTopic expects a list of documents
        documents = non_empty_df[text_column].astype(str).fillna("").tolist()

        try:
            topics, _ = self.model.fit_transform(documents)

            # Initialize topic columns for the full dataframe with empty values
            comments_df["topic_id"] = None
            comments_df["topic_label"] = None

            # Assign topics only to non-empty comments
            non_empty_df["topic_id"] = topics

            # Create a mapping from topic ID to a more descriptive label from seed list if possible
            # BERTopic might generate topics not perfectly aligned with seeds, or merge them.
            # The custom_labels_ attribute can be set if topics directly map to seeds.
            # For guided, the topic IDs *should* align with the order of seed_topic_list if distinct topics are formed for each.
            # Topic -1 is for outliers.
            self.topic_mapping = {-1: "OUTLIERS", -2: "EMPTY_COMMENT"}
            generated_topic_info = self.model.get_topic_info()

            # Try to map generated topics back to the initial seed lists for clearer labels
            # This is a simple approach; more sophisticated mapping might be needed if BERTopic merges/alters guided topics significantly.
            for i, seed_words in enumerate(self.seed_topic_list):
                # Check if a topic corresponding to this seed index was actually generated
                if i in generated_topic_info["Topic"].values:
                    # Use the first few seed words as a label, or a predefined name if you have one
                    self.topic_mapping[i] = f"TOPIC_{i}_-" + "_".join(seed_words[:3])
                else:
                    # If BERTopic didn't create a topic for this seed set (e.g., due to min_topic_size or merged topics)
                    # we note it, but it won't appear in comment_df["topic_label"] unless assigned later.
                    print(
                        f"Note: Seed topic {i} ({'_'.join(seed_words[:3])}) might not have formed a distinct topic or was merged."
                    )

            # Add any other topics BERTopic might have generated beyond the guided ones (should be rare in pure guided mode)
            for topic_id in generated_topic_info["Topic"].values:
                if topic_id not in self.topic_mapping:
                    self.topic_mapping[topic_id] = generated_topic_info[
                        generated_topic_info["Topic"] == topic_id
                    ]["Name"].iloc[0]

            # Apply topic labels to non-empty comments
            non_empty_df["topic_label"] = (
                non_empty_df["topic_id"].map(self.topic_mapping).fillna("UNKNOWN_TOPIC")
            )

            # Update the original dataframe with topic information for non-empty comments
            comments_df.loc[~comments_df["is_empty_comment"], "topic_id"] = (
                non_empty_df["topic_id"].values
            )
            comments_df.loc[~comments_df["is_empty_comment"], "topic_label"] = (
                non_empty_df["topic_label"].values
            )

            # Assign empty comment topic to empty comments
            comments_df.loc[comments_df["is_empty_comment"], "topic_id"] = -2
            comments_df.loc[comments_df["is_empty_comment"], "topic_label"] = (
                "EMPTY_COMMENT"
            )

            # Drop the temporary column
            comments_df = comments_df.drop(columns=["is_empty_comment"])

            print("Topic modeling complete.")
            print(f"Topic counts:\n{comments_df['topic_label'].value_counts()}")

        except Exception as e:
            print(f"Error during BERTopic fit_transform: {e}")
            comments_df["topic_id"] = -3  # Special ID for error
            comments_df["topic_label"] = "NO_TOPIC_MODELING_ERROR"
            self.topic_mapping = None

            # Drop the temporary column if it exists
            if "is_empty_comment" in comments_df.columns:
                comments_df = comments_df.drop(columns=["is_empty_comment"])

        return comments_df, self.topic_mapping

    def get_topic_info(self) -> Optional[pd.DataFrame]:
        """Returns information about the generated topics."""
        if hasattr(self.model, "topics_") and self.model.topics_ is not None:
            return self.model.get_topic_info()
        return None

    def get_topic_mapping(self) -> Optional[Dict[int, str]]:
        """Returns the mapping of topic IDs to descriptive labels."""
        return self.topic_mapping


# Comprehensive seed topics for 2016 US election analysis
# These topics cover the main candidates, policy issues, controversies, and social movements
# that were prominent during the 2016 US presidential election cycle
DEFAULT_SEED_TOPICS = [
    # Major candidates and their campaigns
    [
        "Trump",
        "Donald Trump",
        "MAGA",
        "Make America Great Again",
        "Republicans",
        "conservative movement",
    ],
    [
        "Clinton",
        "Hillary",
        "Democrats",
        "I'm With Her",
        "emails",
        "Benghazi",
        "liberal",
    ],
    [
        "Sanders",
        "Bernie",
        "FeelTheBern",
        "progressive",
        "political revolution",
        "inequality",
    ],
    # Election process and key issues
    ["election", "vote", "polls", "rigged", "voter fraud", "debates", "media bias"],
    [
        "gun control",
        "2nd Amendment",
        "NRA",
        "firearms",
        "shootings",
        "gun rights",
        "background checks",
    ],
    ["economy", "jobs", "taxes", "trade", "unemployment", "debt", "Wall Street"],
    ["healthcare", "Obamacare", "ACA", "insurance", "repeal and replace"],
    ["immigration", "border", "wall", "refugees", "deportation", "sanctuary cities"],
    # Foreign policy and national security
    ["foreign policy", "ISIS", "Russia", "terrorism", "national security", "Syria"],
    # Social movements and controversies
    [
        "BlackLivesMatter",
        "BLM",
        "protests",
        "social justice",
        "equality",
        "police brutality",
    ],
    ["scandal", "emails", "Access Hollywood", "WikiLeaks", "Comey", "FBI"],
]

if __name__ == "__main__":
    # This is a placeholder for direct testing if needed.
    # Actual execution will be orchestrated by sentiment_analysis.py
    print("GuidedTopicModeler class defined. Ready for use in a pipeline.")

    # Minimal example:
    # Ensure you have a sample DataFrame to test with
    data = {
        "id": ["c1", "c2", "c3", "c4", "c5", "c6"],
        "subreddit_id": ["t5_1", "t5_1", "t5_2", "t5_2", "t5_1", "t5_3"],
        "body_cleaned": [
            "discussion about trump and his maga campaign",
            "hillary clinton emails are a big deal for democrats",
            "the election polls are looking rigged for sure",
            "another terrible shooting, we need gun control now",
            "donald trump talks about economy and jobs",
            "outlier comment that does not fit any topic well and is short",
        ],
    }
    sample_comments_df = pd.DataFrame(data)

    print("\n--- Testing GuidedTopicModeler with sample data ---")
    topic_modeler = GuidedTopicModeler(
        seed_topic_list=DEFAULT_SEED_TOPICS,
        min_topic_size=1,  # min_topic_size=1 for small test sample
        disable_reduction_for_small_datasets=True,  # Disable UMAP for small test datasets
    )

    # Ensure 'body_cleaned' exists and is not all null
    if (
        "body_cleaned" in sample_comments_df.columns
        and not sample_comments_df["body_cleaned"].isnull().all()
    ):
        comments_with_topics_df, topic_map = topic_modeler.fit_transform(
            sample_comments_df, text_column="body_cleaned"
        )
        print("\nComments with Topics:")
        print(comments_with_topics_df)
        print("\nTopic Info from BERTopic model:")
        topic_info = topic_modeler.get_topic_info()
        if topic_info is not None:
            print(topic_info)
        else:
            print(
                "No topic info available (model likely not fitted or error occurred)."
            )
        print("\nGenerated Topic ID to Label Mapping:")
        print(topic_map)
    else:
        print(
            "Skipping GuidedTopicModeler test due to missing or all-null 'body_cleaned' column in sample data."
        )
