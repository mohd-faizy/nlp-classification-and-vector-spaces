# 🖼️ Assets Directory

This directory stores all visual media, architectural schematics, lecture diagrams, and notebook plots. All files are standardized with clear, numbered naming conventions corresponding to their course week and mathematical concept.

## 1. Course Banner
- `banner.png`: Main header banner for course documentation.

## 2. Theoretical Lecture Diagrams (Referenced in Documentation)
| File | Topic & Concept |
|---|---|
| `fig_01_supervised_learning_cost_minimization.png` | Supervised learning workflow and parameter optimization |
| `fig_02_sentiment_analysis_pipeline.png` | 3-step sentiment classification pipeline: extract, train, predict |
| `fig_03_sparse_vector_representation.png` | Vocabulary indexing and sparse vector representation |
| `fig_04_sparse_representation_tradeoffs.png` | Trade-offs and memory issues with sparse bag-of-words vectors |
| `fig_05_sentiment_word_frequency_table.png` | Positive and negative word frequency counts dictionary |
| `fig_06_feature_extraction_vector.png` | Feature vector format: `[bias, positive_sum, negative_sum]` |
| `fig_07_text_preprocessing_stemming.png` | Text tokenization, stop word elimination, and Porter stemming |
| `fig_08_sigmoid_activation_function.png` | Sigmoid activation function formula $h(z) = \frac{1}{1 + e^{-z}}$ |
| `fig_09_gradient_descent_optimization.png` | Gradient descent weight update algorithm |
| `fig_10_model_evaluation_threshold.png` | Decision boundary and classification accuracy thresholding |
| `fig_11_logistic_cost_function.png` | Binary cross-entropy log-loss cost function |
| `fig_12_naive_bayes_conditional_probability.png` | Conditional probability in tweet sentiment classification |
| `fig_13_negative_sentiment_probability.png` | Complement probability distribution |
| `fig_14_bayes_rule_formula.png` | Bayes' theorem formulation: $P(X \mid Y) = \frac{P(Y \mid X) P(X)}{P(Y)}$ |
| `fig_15_conditional_probability_table.png` | Word-by-class conditional probability matrix |
| `fig_16_laplace_smoothing_formula.png` | Add-1 Laplacian smoothing formulation |
| `fig_17_probability_ratios_table.png` | Ratio of positive-to-negative conditional probabilities |
| `fig_18_likelihood_prior_product.png` | Product of likelihoods and class prior |
| `fig_19_log_likelihood_log_prior.png` | Log likelihood plus log prior score formulation |
| `fig_20_naive_bayes_accuracy_formula.png` | Naïve Bayes accuracy metric on validation sets |
| `fig_21_nb_app_sentiment_analysis.png` | Application 1: Sentiment analysis on social media |
| `fig_22_nb_app_author_identification.png` | Application 2: Stylometry and author identification |
| `fig_23_nb_app_spam_filtering.png` | Application 3: Email spam detection |
| `fig_24_nb_app_information_retrieval.png` | Application 4: Information retrieval and ranking |
| `fig_25_nb_app_word_disambiguation.png` | Application 5: Word sense disambiguation |
| `fig_26_vector_space_firth_context.png` | Distributional hypothesis: "You shall know a word by the company it keeps" |
| `fig_27_word_by_word_cooccurrence.png` | Word-by-word co-occurrence matrix design |
| `fig_28_word_by_document_cooccurrence.png` | Word-by-document matrix design |
| `fig_29_euclidean_distance_formula.png` | Euclidean metric between vectors $d(v, w) = \sqrt{\sum (v_i - w_i)^2}$ |
| `fig_30_cosine_similarity_formula.png` | Cosine similarity formula: $\cos(\beta) = \frac{v \cdot w}{\|v\| \|w\|}$ |
| `fig_31_vector_space_arithmetic_analogies.png` | Vector geometry and semantic relationships (King - Man + Woman = Queen) |
| `fig_32_pca_projection_algorithm.png` | PCA eigenvector projection onto $k$ dimensions |
| `fig_33_pca_variance_retained.png` | Percentage of retained variance formula |
| `fig_34_word_translation_matrix_xr_y.png` | Cross-lingual embedding alignment: $XR \approx Y$ |
| `fig_35_frobenius_loss_translation.png` | Frobenius norm loss minimization: $\|XR - Y\|_F^2$ |
| `fig_36_hash_table_concept.png` | Hash tables and basic hash function mappings |
| `fig_37_hash_function_modulo_buckets.png` | Vector modulo hash distribution |
| `fig_38_lsh_hyperplanes_separation.png` | Vector space partitioning via hyperplanes |
| `fig_39_hyperplane_normal_dot_product.png` | Hyperplane normal vector dot product sign determines region |
| `fig_40_hash_value_binary_encoding.png` | Binary hash value computation $\sum 2^i h_i$ |
| `fig_41_multiple_random_hyperplanes.png` | Multi-plane partitioning for locality sensitive hashing |
| `fig_42_approximate_nearest_neighbors.png` | Candidate pooling across multiple random hash tables |
| `fig_43_document_vector_representation.png` | Representing documents as sums of word vectors |
| `fig_44_laplace_smoothing_probability_table.png` | Smoothed conditional probability table |

## 3. Notebook Visual Figures (Referenced in Notebooks)
| File | Notebook | Description |
|---|---|---|
| `sigmoid_activation_plot.jpg` | `04_logistic_regression_sentiment_analysis.ipynb` | Sigmoid activation curve mapping real numbers to [0, 1] |
| `confidence_ellipses.jpg` | `05_visualizing_naive_bayes_confidence_ellipses.ipynb` | 2-sigma and 3-sigma covariance confidence ellipses |
| `world_map_capitals.jpg` | `10_vector_spaces_and_word_analogies.ipynb` | World map illustration for country-capital analogies |
| `vector_analogies.jpg` | `10_vector_spaces_and_word_analogies.ipynb` | Vector parallelogram analogy geometry |
| `pca_word_embeddings.jpg` | `10_vector_spaces_and_word_analogies.ipynb` | 2D PCA cluster projection of word embeddings |
| `gaussian_scatter_pca.png` | `09_pca_dimensionality_reduction.ipynb` | Gaussian 2D scatter and PCA principal axes |
| `vector_spaces_detailed.jpg` | Week 3 Extra Reference | Detailed vector space illustration |
| `english_french_translation.jpg` | `13_machine_translation_and_document_search.ipynb` | English-to-French word vector translation overview |
| `transformation_matrix_x_to_y.jpg` | `13_machine_translation_and_document_search.ipynb` | Transformation matrix mapping space X to space Y |
| `lsh_single_plane_partition.png` | `13_machine_translation_and_document_search.ipynb` | Single hyperplane division of vector space into 2 buckets |
| `lsh_multiple_planes_partition.png` | `13_machine_translation_and_document_search.ipynb` | Two hyperplanes dividing vector space into 4 buckets |
| `hash_table_lookup.png` | `13_machine_translation_and_document_search.ipynb` | Hash table lookup architecture for fast vector retrieval |
| `vector_transform_x.svg` | `11_vector_rotations_and_transformations.ipynb` | Standalone vector arrow plot |
| `vector_transform_xy.svg` | `11_vector_rotations_and_transformations.ipynb` | Vector pair comparison plot |
| `vector_transform_rotation.svg` | `11_vector_rotations_and_transformations.ipynb` | 2D vector rotation transformation plot |
