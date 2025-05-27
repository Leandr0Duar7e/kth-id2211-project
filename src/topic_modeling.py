import pandas as pd
from bertopic import BERTopic
from typing import List, Dict, Tuple, Optional

# Added imports for LDA
import gensim
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS as GENSIM_STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import word_tokenize
import nltk
from collections import Counter
import string

# Ensure NLTK resources are available (run this once if needed)
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


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


class LdaTopicModeler:
    """Performs topic modeling using LDA with Gensim and maps topics to predefined seeds."""

    def __init__(
        self,
        seed_topic_list: List[List[str]],
        num_lda_topics: Optional[int] = None,  # Number of topics for LDA to find
        passes: int = 10,
        iterations: int = 50,
        min_word_length: int = 3,
        lemmatize: bool = True,
    ):
        self.seed_topic_list = seed_topic_list
        self.num_predefined_topics = len(seed_topic_list)
        # If num_lda_topics is not specified, set it slightly higher than predefined
        # to allow some flexibility, or equal if precise matching is desired.
        # For now, let's aim to match it closely.
        self.num_lda_topics = (
            num_lda_topics if num_lda_topics is not None else self.num_predefined_topics
        )

        self.passes = passes
        self.iterations = iterations
        self.min_word_length = min_word_length
        self.lemmatize_flag = lemmatize
        self.dictionary: Optional[corpora.Dictionary] = None
        self.lda_model: Optional[models.LdaModel] = None
        self.lda_to_predefined_mapping: Dict[int, str] = (
            {}
        )  # Maps LDA topic ID to predefined label
        self.predefined_topic_labels: List[str] = [
            f"TOPIC_{i}_-" + "_".join(seeds[:3])
            for i, seeds in enumerate(self.seed_topic_list)
        ]

        if self.lemmatize_flag:
            self.lemmatizer = WordNetLemmatizer()

    def _lemmatize(self, text: str) -> str:
        if not self.lemmatize_flag:
            return text
        return " ".join(
            [self.lemmatizer.lemmatize(word) for word in word_tokenize(text)]
        )

    def _preprocess_text(self, document: str) -> List[str]:
        """Tokenizes, removes stopwords, and optionally lemmatizes."""
        processed_tokens = []
        # Lemmatize first if flag is set
        text_to_process = (
            self._lemmatize(document.lower())
            if self.lemmatize_flag
            else document.lower()
        )

        for token in simple_preprocess(
            text_to_process, deacc=True
        ):  # simple_preprocess handles tokenization and lowercasing
            if token not in GENSIM_STOPWORDS and len(token) >= self.min_word_length:
                processed_tokens.append(token)
        return processed_tokens

    def _map_lda_topics_to_predefined(self):
        """Maps trained LDA topics to the closest predefined seed topics."""
        if not self.lda_model or not self.dictionary:
            print("Error: LDA model not trained yet.")
            return

        self.lda_to_predefined_mapping = {}
        unmapped_lda_topics = set(range(self.num_lda_topics))

        # Prepare predefined topic seed sets for faster lookup
        predefined_seed_sets = [
            set(self._preprocess_text(" ".join(seeds)))
            for seeds in self.seed_topic_list
        ]

        lda_topic_words = []
        for i in range(self.num_lda_topics):
            # Get top N words for each LDA topic
            top_words = [word for word, prob in self.lda_model.show_topic(i, topn=15)]
            lda_topic_words.append(set(top_words))

        # Iterate through each predefined topic and find the best matching LDA topic
        assigned_lda_topics = set()
        for predefined_idx, seed_set in enumerate(predefined_seed_sets):
            if not seed_set:
                continue  # Skip if seed set is empty after preprocessing

            best_lda_topic_idx = -1
            max_similarity = -1.0

            for lda_idx in unmapped_lda_topics:
                if lda_idx in assigned_lda_topics:
                    continue  # Already assigned to a different predefined topic

                lda_words_set = lda_topic_words[lda_idx]
                if not lda_words_set:
                    continue

                # Jaccard similarity
                intersection_len = len(lda_words_set.intersection(seed_set))
                union_len = len(lda_words_set.union(seed_set))
                similarity = intersection_len / union_len if union_len > 0 else 0.0

                if similarity > max_similarity:
                    max_similarity = similarity
                    best_lda_topic_idx = lda_idx

            # Assign if a reasonable match is found (e.g., similarity > 0.05 or at least one word overlap)
            # This threshold might need tuning.
            # And ensure an LDA topic is not assigned to multiple predefined topics.
            if (
                best_lda_topic_idx != -1
                and max_similarity > 0.01
                and best_lda_topic_idx not in assigned_lda_topics
            ):
                self.lda_to_predefined_mapping[best_lda_topic_idx] = (
                    self.predefined_topic_labels[predefined_idx]
                )
                assigned_lda_topics.add(best_lda_topic_idx)
                print(
                    f"Mapped Predefined Topic '{self.predefined_topic_labels[predefined_idx]}' to LDA Topic {best_lda_topic_idx} (Similarity: {max_similarity:.2f})"
                )

        # Handle any LDA topics that couldn't be mapped to a predefined one
        # (e.g., give them a generic label or mark as 'UNMAPPED_LDA_TOPIC_X')
        for lda_idx in range(self.num_lda_topics):
            if lda_idx not in self.lda_to_predefined_mapping:
                unmapped_label = f"UNMAPPED_LDA_TOPIC_{lda_idx}"
                # Try to get top words for unmapped LDA topic for a more descriptive label
                try:
                    top_unmapped_words = "_".join(
                        [word for word, _ in self.lda_model.show_topic(lda_idx, 3)]
                    )
                    unmapped_label = f"LDA_TOPIC_{lda_idx}_{top_unmapped_words}"
                except Exception:
                    pass  # Stick to generic if words can't be fetched
                self.lda_to_predefined_mapping[lda_idx] = unmapped_label
                print(
                    f"LDA Topic {lda_idx} could not be mapped to a predefined topic. Labeled as '{unmapped_label}'."
                )

        print(f"Final LDA to Predefined Mapping: {self.lda_to_predefined_mapping}")

    def fit_transform(
        self,
        aggregated_texts_df: pd.DataFrame,
        text_column: str = "merged_body_cleaned",
    ) -> pd.DataFrame:
        """
        Fits LDA model to aggregated texts and assigns predefined topic labels.

        Args:
            aggregated_texts_df (pd.DataFrame): DataFrame with 'link_id' and a text column
                                                (e.g., 'merged_body_cleaned').
            text_column (str): Name of the column containing the aggregated text.

        Returns:
            pd.DataFrame: The input DataFrame with an added 'topic_label' column.
        """
        if aggregated_texts_df.empty or text_column not in aggregated_texts_df.columns:
            print(
                "Warning: Aggregated texts DataFrame is empty or lacks the text column. Skipping LDA."
            )
            aggregated_texts_df["topic_label"] = "NO_TOPIC_EMPTY_INPUT"
            return aggregated_texts_df

        documents = aggregated_texts_df[text_column].astype(str).fillna("").tolist()
        preprocessed_docs = [self._preprocess_text(doc) for doc in documents]
        preprocessed_docs = [
            doc for doc in preprocessed_docs if doc
        ]  # Remove empty docs after preprocessing

        if not preprocessed_docs:
            print(
                "Warning: No processable documents after preprocessing. Skipping LDA."
            )
            aggregated_texts_df["topic_label"] = "NO_TOPIC_PREPROCESSING_EMPTY"
            return aggregated_texts_df

        self.dictionary = corpora.Dictionary(preprocessed_docs)
        # Filter extremes: remove tokens that appear in less than 5 documents or more than 50% of the documents
        self.dictionary.filter_extremes(
            no_below=5, no_above=0.5
        )  # These values may need tuning

        corpus = [self.dictionary.doc2bow(doc) for doc in preprocessed_docs]
        # Filter out empty items in corpus (documents that had all their words filtered out)
        # And keep track of original indices to map back
        original_indices = [i for i, c in enumerate(corpus) if c]
        filtered_corpus = [c for c in corpus if c]

        if not filtered_corpus:
            print("Warning: Corpus is empty after dictionary filtering. Skipping LDA.")
            aggregated_texts_df["topic_label"] = "NO_TOPIC_EMPTY_CORPUS"
            return aggregated_texts_df

        print(
            f"Training LDA model with {self.num_lda_topics} topics on {len(filtered_corpus)} documents..."
        )
        self.lda_model = models.LdaModel(
            corpus=filtered_corpus,
            id2word=self.dictionary,
            num_topics=self.num_lda_topics,
            passes=self.passes,
            iterations=self.iterations,
            random_state=42,  # For reproducibility
            # alpha='auto', # Let gensim learn alpha
            # eta='auto' # Let gensim learn eta
        )

        self._map_lda_topics_to_predefined()

        # Assign topics to the original documents that made it into the filtered_corpus
        doc_topics = [
            self.lda_model.get_document_topics(filtered_corpus[i])
            for i in range(len(filtered_corpus))
        ]

        assigned_topic_labels = []
        for i in range(len(doc_topics)):
            # Get the LDA topic with the highest probability for the current document
            if doc_topics[i]:  # Check if topics were assigned
                best_lda_topic_id = sorted(
                    doc_topics[i], key=lambda x: x[1], reverse=True
                )[0][0]
                assigned_topic_labels.append(
                    self.lda_to_predefined_mapping.get(
                        best_lda_topic_id, "UNKNOWN_TOPIC"
                    )
                )
            else:
                assigned_topic_labels.append(
                    "NO_TOPIC_ASSIGNED"
                )  # Document couldn't be assigned a topic by LDA

        # Create a temporary series for merging, aligning with original_indices
        topic_labels_series = pd.Series(
            assigned_topic_labels, index=aggregated_texts_df.index[original_indices]
        )

        # Add topic_label column to the input DataFrame
        aggregated_texts_df["topic_label"] = topic_labels_series
        # Fill NaNs for documents that were filtered out or couldn't be processed
        aggregated_texts_df["topic_label"].fillna("NO_TOPIC_FILTERED_OUT", inplace=True)

        print("LDA Topic modeling complete.")
        print(f"Topic counts:\n{aggregated_texts_df['topic_label'].value_counts()}")
        return aggregated_texts_df

    def get_topic_info(self) -> Optional[List[Tuple[int, List[Tuple[str, float]]]]]:
        """Returns information about the generated LDA topics (top words)."""
        if self.lda_model:
            return self.lda_model.show_topics(
                num_topics=self.num_lda_topics, num_words=10, formatted=False
            )
        return None

    def get_lda_to_predefined_mapping(self) -> Dict[int, str]:
        """Returns the mapping of LDA topic IDs to predefined topic labels."""
        return self.lda_to_predefined_mapping


class SimpleLdaTopicModeler:
    """Performs topic modeling using LDA with Gensim without mapping to predefined topics."""

    def __init__(
        self,
        num_topics: int = 20,
        passes: int = 5,
        iterations: int = 30,
        min_word_length: int = 3,
        lemmatize: bool = False,  # Set default to False since text is already processed
    ):
        """
        Initializes the SimpleLdaTopicModeler.

        Args:
            num_topics (int): Number of topics for LDA to find. Defaults to 20.
            passes (int): Number of passes for LDA training. Defaults to 5.
            iterations (int): Number of iterations for LDA training. Defaults to 30.
            min_word_length (int): Minimum word length to keep. Defaults to 3.
            lemmatize (bool): Whether to lemmatize words. Defaults to False since text is already processed.
        """
        self.num_topics = num_topics
        self.passes = passes
        self.iterations = iterations
        self.min_word_length = min_word_length
        self.lemmatize_flag = lemmatize
        self.dictionary = None
        self.lda_model = None
        self.topic_words = {}
        # Extended stopwords list for political content
        self.stopwords = set(GENSIM_STOPWORDS).union(
            {
                # Standard English stopwords (some overlap with GENSIM, but ensures coverage)
                "probably",
                "end",
                "knows",
                "states",
                "dr.",
                "vote",
                "voting",
                "mean",
                "nice",
                "great",
                "reply",
                "points",
                "nearly",
                "looking",
                "hard",
                "passed",
                "making",
                "better",
                "result",
                "frankly",
                "agree",
                "saying",
                "important",
                "certainly",
                "political",
                "called",
                "exactly",
                "dude",
                "wrong",
                "away",
                "comparison",
                "effort",
                "little",
                "regular",
                "lots",
                "based",
                "gets",
                "position",
                "want",
                "ones",
                "took",
                "short",
                "makes",
                "elected",
                "realize",
                "calling",
                "dont",
                "open",
                "catch",
                "late",
                "significatn",
                "candidate",
                "maybe",
                "state",
                "course",
                "form",
                "reason",
                "remember",
                "capslock",
                "mind",
                "talk",
                "rest",
                "job",
                "sense",
                "code",
                "yeah",
                "responses",
                "feel",
                "thought",
                "speak",
                "instead",
                "read",
                "wow",
                "friend",
                "country",
                "completely",
                "soon",
                "crazy",
                "wish",
                "long",
                "bunch",
                "question",
                "shows",
                "members",
                "definitely",
                "moving",
                "yes",
                "assuming",
                "understand",
                "happens",
                "near",
                "caliming",
                "reach",
                "needed",
                "example",
                "support",
                "bro",
                "wait",
                "a",
                "about",
                "above",
                "after",
                "again",
                "against",
                "all",
                "am",
                "an",
                "and",
                "any",
                "are",
                "as",
                "at",
                "be",
                "because",
                "been",
                "before",
                "being",
                "below",
                "between",
                "both",
                "but",
                "by",
                "can",
                "did",
                "do",
                "does",
                "doing",
                "down",
                "during",
                "each",
                "few",
                "for",
                "from",
                "further",
                "had",
                "has",
                "have",
                "having",
                "he",
                "her",
                "here",
                "hers",
                "herself",
                "him",
                "himself",
                "his",
                "how",
                "i",
                "if",
                "in",
                "into",
                "is",
                "it",
                "its",
                "itself",
                "just",
                "me",
                "more",
                "most",
                "my",
                "myself",
                "no",
                "nor",
                "not",
                "now",
                "of",
                "off",
                "on",
                "once",
                "only",
                "or",
                "other",
                "our",
                "ours",
                "ourselves",
                "out",
                "over",
                "own",
                "s",
                "same",
                "she",
                "should",
                "so",
                "some",
                "such",
                "t",
                "than",
                "that",
                "the",
                "their",
                "theirs",
                "them",
                "themselves",
                "then",
                "there",
                "these",
                "they",
                "this",
                "those",
                "through",
                "to",
                "too",
                "under",
                "until",
                "up",
                "very",
                "was",
                "we",
                "were",
                "what",
                "when",
                "where",
                "which",
                "while",
                "who",
                "whom",
                "why",
                "will",
                "with",
                "you",
                "your",
                "yours",
                "yourself",
                "yourselves",
                # Contractions and common informalities
                "n't",
                "'s",
                "'re",
                "'ve",
                "'ll",
                "'d",
                "'m",
                "ca",
                "wo",
                "sha",
                "gon",
                "wan",
                "got",
                "get",
                "lot",
                "let",
                # Common words that add little meaning in general discussion context
                "also",
                "would",
                "could",
                "should",
                "might",
                "may",
                "must",
                "really",
                "good",
                "well",
                "much",
                "many",
                "one",
                "two",
                "see",
                "say",
                "said",
                "says",
                "make",
                "made",
                "take",
                "new",
                "old",
                "big",
                "small",
                "high",
                "low",
                "even",
                "still",
                "since",
                "back",
                "way",
                "time",
                "people",
                "person",
                "thing",
                "things",
                "something",
                "nothing",
                "everything",
                "another",
                "other",
                "others",
                "first",
                "last",
                "next",
                "look",
                "looks",
                "use",
                "used",
                "using",
                "come",
                "comes",
                "came",
                "give",
                "gives",
                "gave",
                "ask",
                "asks",
                "asked",
                "tell",
                "tells",
                "told",
                "try",
                "tries",
                "tried",
                "day",
                "days",
                "week",
                "month",
                "year",
                "years",
                "today",
                "tomorrow",
                "yesterday",
                # Reddit specific / common internet slang
                "url",
                "comment",
                "post",
                "thread",
                "subreddit",
                "reddit",
                "op",
                "lol",
                "wtf",
                "imo",
                "btw",
                "http",
                "https",
                "www",
                "com",
                "org",
                "net",
                "gov",
                "edu",
                "thanks",
                "please",
                "sorry",
                "actually",
                "point",
                "article",
                "link",
                "don",
                "nt",
                "re",
                "ve",
                "ll",
                "d",
                "m",
                "s",
                "think",
                "going",
                "like",
                "man",
                "know",
                "does",
                "has",
                "did",
            }
        )

        if self.lemmatize_flag:
            self.lemmatizer = WordNetLemmatizer()

    def _lemmatize(self, text: str) -> str:
        """Lemmatizes text if lemmatize_flag is set."""
        if not self.lemmatize_flag:
            return text
        return " ".join(
            [self.lemmatizer.lemmatize(word) for word in word_tokenize(text)]
        )

    def _preprocess_text(self, document: str) -> List[str]:
        """Tokenizes, removes stopwords, and filters for minimum length."""
        # For already processed text, split by space and filter for stopwords and minimum length
        # This assumes body_cleaned is already tokenized and cleaned
        return [
            token
            for token in document.lower().split()
            if (
                token not in self.stopwords
                and len(token) >= self.min_word_length
                and not token.startswith("'")  # remove tokens like 's
                and not token.startswith("...")  # remove ellipsis tokens
                and not token.isdigit()  # remove numbers
                and not all(
                    c in string.punctuation for c in token
                )  # remove tokens made of only punctuation
            )
        ]

    def fit_transform(
        self,
        aggregated_texts_df: pd.DataFrame,
        text_column: str = "merged_body_cleaned",
    ) -> pd.DataFrame:
        """
        Fits LDA model to aggregated texts and assigns topic IDs.

        Args:
            aggregated_texts_df (pd.DataFrame): DataFrame with 'link_id' and a text column.
            text_column (str): Name of the column containing the aggregated text.

        Returns:
            pd.DataFrame: The input DataFrame with an added 'topic_id' column.
        """
        # Initialize topic_id column with -1 (default for empty or error cases)
        aggregated_texts_df["topic_id"] = -1

        # Check if input is valid
        if aggregated_texts_df.empty or text_column not in aggregated_texts_df.columns:
            print(
                "Warning: Aggregated texts DataFrame is empty or lacks the text column. Skipping LDA."
            )
            return aggregated_texts_df

        # Explicitly mark empty texts with topic_id = -1
        empty_mask = aggregated_texts_df[text_column].astype(str).str.strip() == ""
        if empty_mask.all():
            print("Warning: All texts are empty. Skipping LDA.")
            return aggregated_texts_df

        # Just tokenize by whitespace since text is already cleaned
        # Focus only on non-empty documents
        non_empty_df = aggregated_texts_df[~empty_mask].copy()
        documents = non_empty_df[text_column].astype(str).fillna("").tolist()
        preprocessed_docs = [self._preprocess_text(doc) for doc in documents]

        # Remove documents that become too short after preprocessing (e.g., < 5 words)
        valid_indices = []
        filtered_preprocessed_docs = []
        for i, doc in enumerate(preprocessed_docs):
            if len(doc) >= 5:
                valid_indices.append(i)
                filtered_preprocessed_docs.append(doc)

        if not filtered_preprocessed_docs:
            print(
                "Warning: No processable documents after preprocessing (all too short). Skipping LDA."
            )
            return aggregated_texts_df

        # Create dictionary and corpus - with filtering optimized for meaningful topics
        self.dictionary = corpora.Dictionary(filtered_preprocessed_docs)

        # Apply more aggressive filtering for political content:
        # - no_below=3: Remove terms that appear in fewer than 3 documents (remove rare terms)
        # - no_above=0.6: Remove terms that appear in more than 60% of documents (more aggressive than 0.7)
        # - keep_n=3000: Keep only the top 3000 most frequent terms after filtering (more aggressive)
        self.dictionary.filter_extremes(no_below=3, no_above=0.6, keep_n=3000)

        corpus = [self.dictionary.doc2bow(doc) for doc in filtered_preprocessed_docs]

        # Filter empty items in corpus (documents with no words left after dictionary filtering)
        valid_corpus_indices = []
        filtered_corpus = []
        for i, bow in enumerate(corpus):
            if bow:  # Not empty
                valid_corpus_indices.append(valid_indices[i])
                filtered_corpus.append(bow)

        if not filtered_corpus:
            print("Warning: Corpus is empty after dictionary filtering. Skipping LDA.")
            return aggregated_texts_df

        # Train LDA model with adjusted alpha and eta parameters for political content
        print(
            f"Training LDA model with {self.num_topics} topics on {len(filtered_corpus)} documents..."
        )

        # Use asymmetric alpha (some topics more prevalent than others)
        # Lower eta (makes each topic more distinct with fewer shared words)
        self.lda_model = models.LdaModel(
            corpus=filtered_corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            passes=self.passes,
            iterations=self.iterations,
            alpha="asymmetric",  # Let some topics be more common than others
            eta=0.01,  # Sparse word-topic distribution (fewer words per topic)
            random_state=42,  # For reproducibility
            chunksize=100,  # Process 100 documents at a time
            update_every=1,  # Update model after each chunk
        )

        # Store topic words for reference
        self.topic_words = {}
        for topic_id in range(self.num_topics):
            # Get more words per topic to better understand it
            top_words = [
                word for word, _ in self.lda_model.show_topic(topic_id, topn=15)
            ]
            self.topic_words[topic_id] = ", ".join(top_words)

        # Assign topics to documents - only for those that made it through the filtering process
        for i, corpus_doc in enumerate(filtered_corpus):
            original_df_idx = non_empty_df.index[valid_corpus_indices[i]]
            topic_dist = self.lda_model.get_document_topics(
                corpus_doc, minimum_probability=0.15
            )
            if topic_dist:  # Check if any topic meets the probability threshold
                dominant_topic = max(topic_dist, key=lambda x: x[1])[0]
                aggregated_texts_df.loc[original_df_idx, "topic_id"] = dominant_topic

        print("LDA topic modeling complete.")
        topic_counts = aggregated_texts_df["topic_id"].value_counts()
        print(f"Topic distribution:\n{topic_counts}")

        return aggregated_texts_df

    def get_topic_words(self) -> Dict[int, str]:
        """Returns the top words for each topic."""
        return self.topic_words


if __name__ == "__main__":
    # This is a placeholder for direct testing if needed.
    # Actual execution will be orchestrated by sentiment_analysis.py
    print(
        "GuidedTopicModeler and LdaTopicModeler classes defined. Ready for use in a pipeline."
    )

    # Minimal example for GuidedTopicModeler (existing):
    # ... (keep existing BERTopic test if desired, or comment out) ...

    print("\n--- Testing LdaTopicModeler with sample data ---")
    # Sample data: Each row is an "aggregated thread"
    lda_sample_data = {
        "link_id": [
            "t3_link1",
            "t3_link2",
            "t3_link3",
            "t3_link4",
            "t3_link5",
            "t3_link6",
            "t3_link7",
        ],
        "merged_body_cleaned": [
            "Several comments discussing Donald Trump, his MAGA campaign, and Republican strategies for the election. Trump Trump Trump.",  # doc 0
            "This thread is all about Hillary Clinton, her emails controversy, and what Democrats think. Clinton emails.",  # doc 1
            "Lots of talk about the election itself, polls, voter concerns, and media bias. election polls vote.",  # doc 2
            "A heated discussion on gun control, the second amendment, NRA, and recent shootings. gun control NRA.",  # doc 3
            "Economic issues are the focus here: jobs, taxes, trade deficits. economy jobs taxes.",  # doc 4
            "A long thread about healthcare, Obamacare, and the Affordable Care Act. healthcare obamacare.",  # doc 5
            "This document is different and talks about kittens and puppies for variety.",  # doc 6 (potential outlier)
        ],
    }
    sample_aggregated_df = pd.DataFrame(lda_sample_data)

    # Initialize LdaTopicModeler
    lda_topic_modeler = LdaTopicModeler(
        seed_topic_list=DEFAULT_SEED_TOPICS,
        num_lda_topics=len(
            DEFAULT_SEED_TOPICS
        ),  # Aim for one LDA topic per seed initially
        passes=1,  # Fewer passes for quick test
        iterations=10,  # Fewer iterations for quick test
        lemmatize=False,  # Lemmatization can be slow for quick tests; NLTK setup might also be needed
    )

    # Perform LDA topic modeling
    aggregated_with_topics_df = lda_topic_modeler.fit_transform(
        sample_aggregated_df, text_column="merged_body_cleaned"
    )

    print("\nAggregated Texts with LDA Topics:")
    print(aggregated_with_topics_df)

    print("\nLDA Topic Info (Top words per LDA topic):")
    lda_topic_info = lda_topic_modeler.get_topic_info()
    if lda_topic_info:
        for i, topic in lda_topic_info:
            words = ", ".join([w[0] for w in topic])
            print(f"LDA Topic {i}: {words}")

    print("\nLDA Topic ID to Predefined Label Mapping:")
    print(lda_topic_modeler.get_lda_to_predefined_mapping())

    print("\n--- BERTopic Test (Original) ---")
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
    sample_comments_df_bert = pd.DataFrame(data)

    print("\n--- Testing GuidedTopicModeler with sample data ---")
    topic_modeler_bert = GuidedTopicModeler(
        seed_topic_list=DEFAULT_SEED_TOPICS,
        min_topic_size=1,  # min_topic_size=1 for small test sample
        disable_reduction_for_small_datasets=True,  # Disable UMAP for small test datasets
    )

    # Ensure 'body_cleaned' exists and is not all null
    if (
        "body_cleaned" in sample_comments_df_bert.columns
        and not sample_comments_df_bert["body_cleaned"].isnull().all()
    ):
        comments_with_topics_df_bert, topic_map_bert = topic_modeler_bert.fit_transform(
            sample_comments_df_bert, text_column="body_cleaned"
        )
        print("\nComments with Topics (BERTopic):")
        print(comments_with_topics_df_bert)
        print("\nTopic Info from BERTopic model:")
        topic_info_bert = topic_modeler_bert.get_topic_info()
        if topic_info_bert is not None:
            print(topic_info_bert)
        else:
            print(
                "No topic info available (model likely not fitted or error occurred)."
            )
        print("\nGenerated Topic ID to Label Mapping (BERTopic):")
        print(topic_map_bert)
    else:
        print(
            "Skipping GuidedTopicModeler test due to missing or all-null 'body_cleaned' column in sample data."
        )
