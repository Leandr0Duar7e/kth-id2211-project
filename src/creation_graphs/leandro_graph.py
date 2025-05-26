import pandas as pd
import numpy as np
from src.load_data.load_comments import *

def create_leandro_graph(comments: pd.DataFrame) -> pd.DataFrame:
    # remove default upvote
    comments = comments.copy()
    comments['score'] -= 1

    # flags and raw sums
    comments['is_positive'] = comments['score'] > 0
    comments['is_negative'] = comments['score'] < 0

    # aggregate in one go
    summary = (
        comments
        .groupby(['author', 'subreddit'], as_index=False)
        .agg(
            num_positive_comments=('is_positive', 'sum'),
            num_negative_comments=('is_negative', 'sum'),
            total_positive_score=('score', lambda x: x.clip(lower=0).sum()),
            total_negative_score=('score', lambda x: x.clip(upper=0).sum()),
            total_comments=('score', 'count'),
        )
    )

    # avoid NaN
    summary[['total_positive_score', 'total_negative_score']] = \
        summary[['total_positive_score', 'total_negative_score']].fillna(0)

    # compute numeric sentiment
    # (re-using your Leandro score formula)
    np_pos = summary['num_positive_comments']
    np_neg = summary['num_negative_comments']
    sp    = summary['total_positive_score'].abs()
    sn    = summary['total_negative_score'].abs()
    tot   = np_pos + np_neg
    # protect div-by-zero
    tot = tot.replace({0: np.nan})
    raw_sent = (sp * np_pos / tot) - (sn * np_neg / tot)
    raw_sent = raw_sent.fillna(0)

    # map to categories
    summary['sentiment'] = np.select(
        [raw_sent > 0, raw_sent < 0],
        ['Positive', 'Negative'],
        default='Neutral'
    )

    return summary



