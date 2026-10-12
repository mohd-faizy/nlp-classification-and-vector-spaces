"""
Natural Language Processing with Classification and Vector Spaces
Utility functions module (Course 1 - DeepLearning.AI)
Consolidated utilities supporting all assignments and lecture notebooks (Weeks 1-4).
"""

import os
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

# ---------------------------------------------------------------------------
# Auto-configure local NLTK data paths so pre-downloaded corpora load seamlessly
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
for _candidate in [
    os.path.join(_BASE_DIR, 'data', 'corpora', 'nltk_stopwords'),
    os.path.join(_BASE_DIR, 'data', 'corpora'),
    os.path.join(_BASE_DIR, 'data', 'tmp2'),
    os.path.join(_BASE_DIR, 'data'),
    os.path.join(_BASE_DIR, 'tmp2'),
]:
    if os.path.exists(_candidate) and _candidate not in nltk.data.path:
        nltk.data.path.append(_candidate)


# ---------------------------------------------------------------------------
# Week 1 & 2: Text Preprocessing and Frequency Extraction
# ---------------------------------------------------------------------------
def process_tweet(tweet):
    """
    Cleans and tokenizes a raw tweet for NLP classification.
    
    Steps:
      1. Removes stock tickers ($GE), old-style retweet text (RT), URLs, and hashtags (#).
      2. Tokenizes into words using TweetTokenizer.
      3. Removes English stopwords and punctuation marks.
      4. Stems words using PorterStemmer.

    Parameters:
        tweet (str): Raw tweet text string.

    Returns:
        list of str: List of cleaned, stemmed tokens.
    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')

    # Remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # Remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # Remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # Remove hash # signs
    tweet = re.sub(r'#', '', tweet)

    # Tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if word not in stopwords_english and word not in string.punctuation:
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)

    return tweets_clean


def build_freqs(tweets, ys):
    """
    Builds frequency mapping from words and sentiment labels.

    Parameters:
        tweets (list of str): List of tweets.
        ys (array-like): m x 1 array or list of sentiment labels (0 or 1).

    Returns:
        dict: Mapping of (word, sentiment) -> count frequency.
    """
    yslist = np.squeeze(ys).tolist()
    if not isinstance(yslist, list):
        yslist = [yslist]
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs


# ---------------------------------------------------------------------------
# Week 2: Naive Bayes Utilities & Visualization
# ---------------------------------------------------------------------------
def lookup(freqs, word, label):
    """
    Looks up frequency of a (word, label) pair in freqs dictionary.

    Parameters:
        freqs (dict): Frequency dictionary mapping (word, label) to count.
        word (str): The token to search.
        label (int): Sentiment label (0 or 1).

    Returns:
        int: Frequency count of the word for the specified label.
    """
    return freqs.get((word, label), 0)


def test_lookup(func):
    """Sanity check test for the lookup function."""
    freqs = {('sad', 0): 4, ('happy', 1): 12, ('oppressed', 0): 7}
    word = 'happy'
    label = 1
    if func(freqs, word, label) == 12:
        return 'SUCCESS!!'
    return 'Failed Sanity Check!'


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`.

    Parameters:
        x, y (array_like, shape (n, )): Input coordinate data.
        ax (matplotlib.axes.Axes): Target axes object.
        n_std (float): Standard deviations to determine radius.
        facecolor (str): Fill color.
        **kwargs: Additional patch properties passed to Ellipse.

    Returns:
        matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor,
                      **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = (transforms.Affine2D()
              .rotate_deg(45)
              .scale(scale_x, scale_y)
              .translate(mean_x, mean_y))

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


# ---------------------------------------------------------------------------
# Week 3 & 4: Vector Spaces, Word Embeddings & Translation
# ---------------------------------------------------------------------------
def get_vectors(embeddings, words):
    """
    Extracts word embedding matrix for a list of words.

    Parameters:
        embeddings (dict): Pre-computed word embedding dictionary.
        words (list of str): List of words to extract embeddings for.

    Returns:
        np.ndarray: Matrix of shape (len(words), 300) containing embeddings.
    """
    X = np.zeros((1, 300))
    for word in words:
        eng_emb = embeddings[word]
        X = np.row_stack((X, eng_emb))
    X = X[1:, :]
    return X


def get_dict(file_name):
    """
    Reads an English-to-French vocabulary mapping from a space-separated file.
    Automatically checks both the given path and the 'data/' directory.

    Parameters:
        file_name (str): Path or filename of the dictionary file.

    Returns:
        dict: Mapping of English word to French translation.
    """
    target = file_name
    if not os.path.exists(target):
        candidates = [
            os.path.join(_BASE_DIR, 'data', 'dictionaries', os.path.basename(file_name)),
            os.path.join(_BASE_DIR, 'data', 'dictionaries', 'english_french_translation_train.txt' if 'train' in file_name else 'english_french_translation_test.txt' if 'test' in file_name else 'english_french_dictionary.txt'),
            os.path.join(_BASE_DIR, 'data', 'dictionaries', 'country_capitals.txt' if 'capitals' in file_name else os.path.basename(file_name)),
            os.path.join(_BASE_DIR, 'data', os.path.basename(file_name)),
            os.path.join('data', 'dictionaries', os.path.basename(file_name)),
            os.path.join('data', file_name),
        ]
        for c in candidates:
            if os.path.exists(c):
                target = c
                break

    my_file = pd.read_csv(target, delimiter=' ')
    etof = {}
    for i in range(len(my_file)):
        en = my_file.iloc[i, 0]
        fr = my_file.iloc[i, 1]
        etof[en] = fr

    return etof


def cosine_similarity(A, B):
    """
    Calculates cosine similarity between two vector arrays.

    Parameters:
        A, B (np.ndarray): Input vectors.

    Returns:
        float: Cosine similarity value between -1.0 and 1.0.
    """
    dot = np.dot(A, B)
    norma = np.linalg.norm(A)
    normb = np.linalg.norm(B)
    return dot / (norma * normb)


def plot_vectors(vectors, colors=None, axes=None, fname='image.svg', ax=None):
    """
    Plots arrows representing 2D vectors using matplotlib.

    Parameters:
        vectors (list of array-like): List of 2D vectors to plot.
        colors (list of str, optional): Colors for each vector arrow.
        axes (tuple of float, optional): Custom (x_lim, y_lim) bounds.
        fname (str): Filename to save output image when standalone figure.
        ax (matplotlib.axes.Axes, optional): Target axes, if any.
    """
    if colors is None:
        colors = ['k', 'b', 'r', 'm', 'c']

    x_dir = [vec[0][0] for vec in vectors]
    y_dir = [vec[0][1] for vec in vectors]

    standalone = ax is None
    if standalone:
        fig, ax2 = plt.subplots()
    else:
        ax2 = ax

    if axes is None:
        x_axis = 2 + np.max(np.abs(x_dir))
        y_axis = 2 + np.max(np.abs(y_dir))
    else:
        x_axis = axes[0]
        y_axis = axes[1]

    ax2.axis([-x_axis, x_axis, -y_axis, y_axis])

    for i, vec in enumerate(vectors):
        color = colors[i % len(colors)]
        ax2.arrow(0, 0, vec[0][0], vec[0][1],
                  head_width=0.05 * x_axis, head_length=0.05 * y_axis,
                  fc=color, ec=color)

    if standalone:
        # Also ensure directory exists if fname contains one
        if os.path.dirname(fname):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
        fig.savefig(fname)
        plt.show()
