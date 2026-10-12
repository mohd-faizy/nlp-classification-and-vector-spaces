"""
Translation helper utilities (Week 3 Vector Spaces).
"""

import os
import string
import unicodedata
import pandas as pd

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_translations(vec_path='data/embeddings/wiki.multi.fr.vec', dict_path='data/dictionaries/english_french_dictionary.txt', capitals_path='data/dictionaries/country_capitals.txt'):
    """
    Loads English to French translation dictionaries with word embeddings.
    """
    from gensim.models import KeyedVectors
    import nltk

    # Locate files in data/ or data/dictionaries/ if available
    for name, p in [('dict', dict_path), ('cap', capitals_path)]:
        if not os.path.exists(p):
            candidates = [
                os.path.join(_BASE_DIR, 'data', 'dictionaries', os.path.basename(p)),
                os.path.join(_BASE_DIR, 'data', os.path.basename(p)),
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    if name == 'dict':
                        dict_path = cand
                    else:
                        capitals_path = cand
                    break

    fr_embeddings = KeyedVectors.load_word2vec_format(vec_path)
    with open(capitals_path, 'r', encoding='utf-8') as f:
        set_words = set(nltk.word_tokenize(f.read()))

    dict_fr = pd.read_csv(dict_path, delimiter=' ')
    en_to_fr = {}
    fr_to_vec = {}

    for i in range(len(dict_fr)):
        en = dict_fr.iloc[i, 0]
        fr = dict_fr.iloc[i, 1]
        if type(en) != float:
            en = en.capitalize()
        if en in set_words and en not in set(en_to_fr.keys()):
            en_to_fr[en] = fr
            fr_to_vec[fr] = fr_embeddings[fr]

    manual_removals = ['syrienne', 'iranienne', 'malien', 'arménienne', 'chilien', 'équateur']
    for k in manual_removals:
        fr_to_vec.pop(k, None)

    manual_overrides = {
        'Chile': ('chili', 'chili'),
        'Iran': ('iran', 'iran'),
        'Turkey': ('turquie', 'turquie'),
        'Syria': ('syrie', 'syrie'),
        'Nigeria': ('nigeria', 'nigeria'),
        'Mali': ('mali', 'mali'),
        'Armenia': ('arménie', 'arménie'),
        'Ecuador': ('ecuador', 'ecuador'),
        'Niger': ('niger', 'niger')
    }
    for en_k, (fr_k, emb_k) in manual_overrides.items():
        en_to_fr[en_k] = fr_k
        if emb_k in fr_embeddings:
            fr_to_vec[fr_k] = fr_embeddings[emb_k]

    if 'grèce' in fr_embeddings:
        fr_to_vec['grece'] = fr_embeddings['grèce']

    return en_to_fr, fr_to_vec


def remove_accents(data):
    """Normalizes string and strips accents."""
    return ''.join(x for x in unicodedata.normalize('NFKD', data) if x in string.ascii_letters).lower()
