"""
Helper notebook utilities for Week 4 Vector Space Visualizations.
Re-exports plot_vectors and vector operations from utils.
"""

from utils import (
    process_tweet,
    get_dict,
    cosine_similarity,
    plot_vectors,
)

__all__ = ['process_tweet', 'get_dict', 'cosine_similarity', 'plot_vectors']
