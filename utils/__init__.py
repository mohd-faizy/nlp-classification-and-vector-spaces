"""
Natural Language Processing with Classification and Vector Spaces
"""

from .utils import (
    process_tweet,
    build_freqs,
    lookup,
    test_lookup,
    confidence_ellipse,
    get_vectors,
    get_dict,
    cosine_similarity,
    plot_vectors,
)
from .utils_nb import (
    plot_vectors as plot_vectors_nb,
)
from .utils_vecs import (
    load_translations,
    remove_accents,
)

__all__ = [
    'process_tweet',
    'build_freqs',
    'lookup',
    'test_lookup',
    'confidence_ellipse',
    'get_vectors',
    'get_dict',
    'cosine_similarity',
    'plot_vectors',
    'plot_vectors_nb',
    'load_translations',
    'remove_accents',
]
