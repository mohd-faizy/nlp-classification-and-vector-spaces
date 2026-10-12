# 📁 Data Directory

This directory stores all datasets, word vector embeddings, vocabulary translation pairs, and corpora used across the assignments and lecture notebooks. The files are organized into dedicated subfolders based on data type and domain function:

```text
data/
├── features/
│   ├── logistic_regression_features.csv    # Extracted positive/negative sentiment features for tweets (Week 1)
│   └── naive_bayes_features.csv           # Extracted positive/negative sentiment frequencies for tweets (Week 2)
│
├── embeddings/
│   ├── english_word_embeddings_subset.p   # Pickled subset of 300D English word vectors (Week 3)
│   ├── english_word_embeddings.p          # 300D English word embeddings for translation training (Week 4)
│   └── french_word_embeddings.p           # 300D French word embeddings for translation target (Week 4)
│
├── dictionaries/
│   ├── country_capitals.txt               # Country-capital word analogy lookup pairs (Week 3)
│   ├── english_french_dictionary.txt      # Full English-French vocabulary dictionary (Week 3 & 4)
│   ├── english_french_translation_train.txt # Training vocabulary pairs for matrix transformation (Week 4)
│   └── english_french_translation_test.txt  # Evaluation vocabulary pairs for translation accuracy (Week 4)
│
└── corpora/
    └── nltk_stopwords/
        └── corpora/
            ├── stopwords/                 # Pre-downloaded multi-language NLTK stopwords
            └── stopwords.zip              # Compressed archive
```

## Detailed File Registry

| Subfolder | File Name | Size | Function & Assignment Mapping |
|---|---|---|---|
| `features/` | `logistic_regression_features.csv` | ~162 KB | Pre-computed bias, positive, and negative feature counts for tweets used in `03_visualizing_decision_boundary.ipynb`. |
| `features/` | `naive_bayes_features.csv` | ~340 KB | Extracted positive and negative frequencies for tweets used in `05_visualizing_naive_bayes_confidence_ellipses.ipynb`. |
| `embeddings/` | `english_word_embeddings_subset.p` | ~309 KB | 300-dimensional word embeddings used in `08_word_embeddings_and_vector_manipulation.ipynb` and `10_vector_spaces_and_word_analogies.ipynb`. |
| `embeddings/` | `english_word_embeddings.p` | ~8.1 MB | English source embeddings for training matrix $R$ in `13_machine_translation_and_document_search.ipynb`. |
| `embeddings/` | `french_word_embeddings.p` | ~7.3 MB | French target embeddings for evaluating matrix $R$ in `13_machine_translation_and_document_search.ipynb`. |
| `dictionaries/` | `country_capitals.txt` | ~154 KB | Capital city to country mapping table used in `10_vector_spaces_and_word_analogies.ipynb`. |
| `dictionaries/` | `english_french_dictionary.txt` | ~1.85 MB | Full dictionary pairs used for cross-lingual alignment. |
| `dictionaries/` | `english_french_translation_train.txt` | ~178 KB | Training word pairs used to learn the transformation matrix $XR = Y$. |
| `dictionaries/` | `english_french_translation_test.txt` | ~50 KB | Test word pairs used to compute nearest-neighbor translation accuracy. |
| `corpora/` | `nltk_stopwords/` | ~23 KB | Local NLTK stopwords corpora (198 English stop words). Auto-detected by `utils.py`. |
