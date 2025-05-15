# ID2211 Data Mining Project: Temporal Evolution of Political Communities on Reddit

This project analyzes the temporal evolution of political communities on Reddit from 2008 to 2019. We employ graph-based methods to understand how these communities form, change, and interact, particularly in relation to major real-world political events.

## What We've Done

Our analysis so far has focused on understanding the structure and dynamics of the Reddit politosphere:

1.  **Exploratory Data Analysis (EDA):** We began by exploring the yearly network data derived from user comment overlaps between subreddits. This helped us understand the overall network characteristics and identify consistently influential or central subreddits.
2.  **Temporal Shortest Path Analysis:** For key subreddits identified during EDA, we calculated their average shortest path distances to clusters of Democrat-affiliated, Republican-affiliated, and Gun Control-focused subreddits over the 12-year period. This revealed how their proximity to these ideological/topical groups evolved, often showing distinct patterns around major political events.
3.  **Network Visualization & Characterization:** We generated visualizations of the yearly subreddit interaction networks, color-coding nodes based on available metadata (e.g., political party affiliation, banned status, focus on gun control). These visualizations provided qualitative insights into community clustering, polarization, and the an/Users/ldr0/Documents/KTH/Data Mining/FinalProject/src/step0_analysis.pyual prominence of different groups.

## Key Scripts

*   `src/networks_eda.ipynb`: Jupyter notebook containing the initial exploratory data analysis of the subreddit network structures and dynamics.
*   `src/step0_analysis.py`: Script for performing the temporal shortest path analysis between main subreddits and defined target groups (Democrat, Republican, Gun Control). It generates plots showing the evolution of these distances.
*   `src/visualize_networks_by_metadata.py`: Script to generate and save visualizations of the yearly subreddit networks, with nodes colored according to their metadata attributes, allowing for visual inspection of community structures.

## Full Report

For a comprehensive understanding of our methodology, detailed findings, and in-depth analysis, please refer to our full project report:
[**ID2211_Final_Project_Report___Group_1.pdf**](Report/ID2211_Final_Project_Report___Group_1.pdf)

## Next Steps

Building on our current findings, our future work will involve:

*   **Advanced Community Detection:** Applying and refining community detection algorithms (with a primary focus on the Louvain method due to promising initial results) to identify cohesive subreddit clusters within the yearly networks, particularly for impactful years like 2013 and 2016.
*   **Refining Labeling Strategies:** Improving our label propagation approach to categorize a broader range of subreddits into meaningful topical or ideological groups (e.g., Democrat, Republican, Gun Control, Radical).
*   **Textual Analysis of Key Communities:** For subreddit clusters identified as highly responsive to major events in specific years:
    *   **Topic Modeling:** Analyzing their comment data to confirm that discussions centered on those critical events.
    *   **Sentiment Analysis:** Gauging the collective sentiment and emotional tone within these communities concerning those events.
*   **Hypothesis Testing:** Using the combined network and textual analysis to test specific hypotheses, such as how mass shootings influence discussions on gun control, or how public opinion within online communities shifts towards political figures during election periods.

## Installation

1. Clone the repository:

```bash
git clone <repository-url> # Replace <repository-url> with the actual URL
cd <repository-directory-name> # Replace with your local directory name
```

2. Create and activate a virtual environment (recommended):

```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On Unix/MacOS
python -m venv .venv
source .venv/bin/activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```
