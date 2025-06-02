# Reddit Comments Processing Guide

This guide explains how to process Reddit comment data in two ways:

1. **Two-step approach**:
   - Step 1: Topic modeling (CPU-based)
   - Step 2: Sentiment analysis (GPU-based)

2. **Direct sentiment analysis**:
   - Run sentiment analysis directly on comment files (GPU-based)

## Overview

The processing pipeline offers multiple scripts and modes:

1. `run_topic_modeling.py` - Performs topic modeling using LDA (runs on CPU)
2. `run_sentiment_analysis.py` - Adds sentiment analysis with two possible modes:
   - Traditional mode: Works with topic-modeled files
   - Direct mode: Works directly with raw comment files

This flexibility allows you to:
- Run the more computationally intensive topic modeling on CPU first
- Run the deep learning-based sentiment analysis on GPU later
- Skip topic modeling if you only need sentiment analysis
- Process data in batches or on different machines if needed

## Step 1: Topic Modeling

Run this step on a CPU machine to assign topics to comments.

### Basic Usage

To process all monthly comment files from August 2016 to February 2017:

```bash
python src/run_topic_modeling.py
```

This will:
1. Look for comments files in `FinalProject/data/comments/`
2. Process each file from `comments_2016-08.bz2` to `comments_2017-02.bz2`
3. Save results to `FinalProject/data/processed/comments_topics/`

### Options

```bash
python src/run_topic_modeling.py --help
```

Key options:
- `--comments_dir PATH`: Custom directory containing comment files
- `--comments_file_path PATH`: Process a single comment file instead of a directory
- `--output_dir_name NAME`: Custom output directory name (default: "comments_topics")
- `--start_month YYYY-MM`: Start month (default: "2016-08")
- `--end_month YYYY-MM`: End month (default: "2017-02")
- `--num_lda_topics N`: Number of LDA topics (default: 20)
- `--lda_passes N`: Number of LDA passes (default: 5)
- `--lda_iterations N`: Number of LDA iterations (default: 30)
- `--chunk_size N`: Process comments in chunks of N (default: 50000)

### Example

Process a specific file with custom parameters:

```bash
python src/run_topic_modeling.py \
  --comments_file_path "data/comments/comments_2016-10.bz2" \
  --output_dir_name "custom_topics_output" \
  --num_lda_topics 25 \
  --lda_passes 10
```

## Step 2: Sentiment Analysis

You can run sentiment analysis in two ways:

1. **Traditional mode**: Process files that already have topic assignments
2. **Direct mode**: Process comment files directly, skipping topic modeling

### Basic Usage for Traditional Mode

To process all topic-modeled files:

```bash
python src/run_sentiment_analysis.py
```

This will:
1. Look for topic files in `FinalProject/data/processed/comments_topics/`
2. Add sentiment analysis to each file
3. Save results to `FinalProject/data/processed/comments_topics_sentiment/`

### Basic Usage for Direct Mode

To process comment files directly for sentiment analysis:

```bash
python src/run_sentiment_analysis.py --direct_mode
```

This will:
1. Look for comments files in `FinalProject/data/comments/`
2. Process each file from `comments_2016-08.bz2` to `comments_2017-02.bz2` for sentiment
3. Save results to `FinalProject/data/processed/comments_sentiment/`

### Options

```bash
python src/run_sentiment_analysis.py --help
```

Key options:
- `--direct_mode`: Process comment files directly (skips topic modeling)
- `--comments_dir PATH`: Directory with comment files for direct mode
- `--comments_file_path PATH`: Process a single comment file in direct mode
- `--start_month YYYY-MM`: Start month for direct mode (default: "2016-08")
- `--end_month YYYY-MM`: End month for direct mode (default: "2017-02")
- `--input_dir PATH`: Directory with topic-modeled files (traditional mode)
- `--input_file PATH`: Process a single topic-modeled file (traditional mode)
- `--output_dir_name NAME`: Custom output directory name
- `--device {cpu,cuda}`: Device to run sentiment analysis on (default: auto-detect)
- `--batch_size N`: Batch size for sentiment analysis (default: 64)
- `--chunk_size N`: Process data in chunks of N (default: 50000)
- `--file_pattern PATTERN`: Pattern for matching topic files (default: "*_topics.jsonl")

### Examples

Process a specific topic file on GPU (traditional mode):

```bash
python src/run_sentiment_analysis.py \
  --input_file "data/processed/comments_topics/comments_2016-10_topics.jsonl" \
  --device cuda \
  --batch_size 128
```

Process a specific comment file directly on GPU:

```bash
python src/run_sentiment_analysis.py \
  --direct_mode \
  --comments_file_path "data/comments/comments_2016-10.bz2" \
  --device cuda \
  --batch_size 128
```

## Output Format

The scripts produce JSONL files with one JSON object per line.

1. Topic modeling output (`*_topics.jsonl`) includes:
   - `id`: Comment ID
   - `link_id`: Thread ID
   - `author`: Author ID
   - `topic_id`: Assigned topic ID
   - `body_cleaned`: Comment text

2. Sentiment analysis output (traditional mode: `*_topics_sentiment.jsonl`) adds:
   - `sentiment`: Sentiment score (-1 for negative, 0 for neutral, 1 for positive)

3. Direct sentiment analysis output (`*_sentiment.jsonl`) includes:
   - `id`: Comment ID
   - `link_id`: Thread ID
   - `author`: Author ID
   - `sentiment`: Sentiment score (-1 for negative, 0 for neutral, 1 for positive)
   - `body_cleaned`: Comment text

## Tips

- For large datasets, adjust the `chunk_size` parameter to control memory usage
- Increase `batch_size` on GPU to process sentiment analysis faster (if you have enough VRAM)
- Use a larger `num_lda_topics` value for more granular topic modeling
- The sentiment model uses RoBERTa and requires about 2GB of GPU memory at minimum
- Direct mode is useful when you only need sentiment analysis and not topic modeling 