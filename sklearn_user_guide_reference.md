# Scikit-Learn User Guide — Quick Reference

> Source: https://scikit-learn.org/stable/user_guide.html
> Fetched: 2026-03-31

---

## Quick Navigation

| | |
|---|---|
| [Install](https://scikit-learn.org/stable/install.html) | [API Reference](https://scikit-learn.org/stable/api/index.html) |
| [Examples](https://scikit-learn.org/stable/auto_examples/index.html) | [Getting Started](https://scikit-learn.org/stable/getting_started.html) |
| [Glossary](https://scikit-learn.org/stable/glossary.html) | [FAQ](https://scikit-learn.org/stable/faq.html) |
| [Release History](https://scikit-learn.org/stable/whats_new.html) | [Choosing the Right Estimator](https://scikit-learn.org/stable/machine_learning_map.html) |

---

## 1. Supervised Learning
https://scikit-learn.org/stable/supervised_learning.html

### [1.1 Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
- 1.1.1 Ordinary Least Squares
- 1.1.2 Ridge regression and classification
- 1.1.3 Lasso
- 1.1.4 Multi-task Lasso
- 1.1.5 Elastic-Net
- 1.1.6 Multi-task Elastic-Net
- 1.1.7 Least Angle Regression
- 1.1.8 LARS Lasso
- 1.1.9 Orthogonal Matching Pursuit (OMP)
- 1.1.10 Bayesian Regression
- 1.1.11 Logistic regression
- 1.1.12 Generalized Linear Models
- 1.1.13 Stochastic Gradient Descent - SGD
- 1.1.14 Robustness regression: outliers and modeling errors
- 1.1.15 Quantile Regression
- 1.1.16 Polynomial regression: extending linear models with basis functions

### [1.2 Linear and Quadratic Discriminant Analysis](https://scikit-learn.org/stable/modules/lda_qda.html)
- 1.2.1 Dimensionality reduction using Linear Discriminant Analysis
- 1.2.2 Mathematical formulation of the LDA and QDA classifiers
- 1.2.3 Mathematical formulation of LDA dimensionality reduction
- 1.2.4 Shrinkage and Covariance Estimator
- 1.2.5 Estimation algorithms

### [1.3 Kernel Ridge Regression](https://scikit-learn.org/stable/modules/kernel_ridge.html)

### [1.4 Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)
- 1.4.1 Classification
- 1.4.2 Regression
- 1.4.3 Density estimation, novelty detection
- 1.4.4 Complexity
- 1.4.5 Tips on Practical Use
- 1.4.6 Kernel functions
- 1.4.7 Mathematical formulation
- 1.4.8 Implementation details

### [1.5 Stochastic Gradient Descent](https://scikit-learn.org/stable/modules/sgd.html)
- 1.5.1 Classification
- 1.5.2 Regression
- 1.5.3 Online One-Class SVM
- 1.5.4 SGD for sparse data
- 1.5.5 Complexity
- 1.5.6 Stopping criterion
- 1.5.7 Tips on Practical Use
- 1.5.8 Mathematical formulation

### [1.6 Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html)
- 1.6.1 Unsupervised Nearest Neighbors
- 1.6.2 Nearest Neighbors Classification
- 1.6.3 Nearest Neighbors Regression
- 1.6.4 Nearest Neighbor Algorithms
- 1.6.5 Nearest Centroid Classifier
- 1.6.6 Nearest Neighbors Transformer
- 1.6.7 Neighborhood Components Analysis

### [1.7 Gaussian Processes](https://scikit-learn.org/stable/modules/gaussian_process.html)
- 1.7.1 Gaussian Process Regression (GPR)
- 1.7.2 Gaussian Process Classification (GPC)
- 1.7.3 GPC examples
- 1.7.4 Kernels for Gaussian Processes

### [1.8 Cross Decomposition](https://scikit-learn.org/stable/modules/cross_decomposition.html)
- 1.8.1 PLSCanonical
- 1.8.2 PLSSVD
- 1.8.3 PLSRegression
- 1.8.4 Canonical Correlation Analysis

### [1.9 Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
- 1.9.1 Gaussian Naive Bayes
- 1.9.2 Multinomial Naive Bayes
- 1.9.3 Complement Naive Bayes
- 1.9.4 Bernoulli Naive Bayes
- 1.9.5 Categorical Naive Bayes
- 1.9.6 Out-of-core naive Bayes model fitting

### [1.10 Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- 1.10.1 Classification
- 1.10.2 Regression
- 1.10.3 Multi-output problems
- 1.10.4 Complexity
- 1.10.5 Tips on practical use
- 1.10.6 Tree algorithms: ID3, C4.5, C5.0 and CART
- 1.10.7 Mathematical formulation
- 1.10.8 Missing Values Support
- 1.10.9 Minimal Cost-Complexity Pruning

### [1.11 Ensembles: Gradient Boosting, Random Forests, Bagging, Voting, Stacking](https://scikit-learn.org/stable/modules/ensemble.html)
- 1.11.1 Gradient-boosted trees
- 1.11.2 Random forests and other randomized tree ensembles
- 1.11.3 Bagging meta-estimator
- 1.11.4 Voting Classifier
- 1.11.5 Voting Regressor
- 1.11.6 Stacked generalization
- 1.11.7 AdaBoost

### [1.12 Multiclass and Multioutput Algorithms](https://scikit-learn.org/stable/modules/multiclass.html)
- 1.12.1 Multiclass classification
- 1.12.2 Multilabel classification
- 1.12.3 Multiclass-multioutput classification
- 1.12.4 Multioutput regression

### [1.13 Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)
- 1.13.1 Removing features with low variance
- 1.13.2 Univariate feature selection
- 1.13.3 Recursive feature elimination
- 1.13.4 Feature selection using SelectFromModel
- 1.13.5 Sequential Feature Selection
- 1.13.6 Feature selection as part of a pipeline

### [1.14 Semi-Supervised Learning](https://scikit-learn.org/stable/modules/semi_supervised.html)
- 1.14.1 Self Training
- 1.14.2 Label Propagation

### [1.15 Isotonic Regression](https://scikit-learn.org/stable/modules/isotonic.html)

### [1.16 Probability Calibration](https://scikit-learn.org/stable/modules/calibration.html)
- 1.16.1 Calibration curves
- 1.16.2 Calibrating a classifier
- 1.16.3 Usage

### [1.17 Neural Network Models (Supervised)](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)
- 1.17.1 Multi-layer Perceptron
- 1.17.2 Classification
- 1.17.3 Regression
- 1.17.4 Regularization
- 1.17.5 Algorithms
- 1.17.6 Complexity
- 1.17.7 Tips on Practical Use
- 1.17.8 More control with warm_start

---

## 2. Unsupervised Learning
https://scikit-learn.org/stable/unsupervised_learning.html

### [2.1 Gaussian Mixture Models](https://scikit-learn.org/stable/modules/mixture.html)
- 2.1.1 Gaussian Mixture
- 2.1.2 Variational Bayesian Gaussian Mixture

### [2.2 Manifold Learning](https://scikit-learn.org/stable/modules/manifold.html)
- 2.2.1 Introduction
- 2.2.2 Isomap
- 2.2.3 Locally Linear Embedding
- 2.2.4 Modified Locally Linear Embedding
- 2.2.5 Hessian Eigenmapping
- 2.2.6 Spectral Embedding
- 2.2.7 Local Tangent Space Alignment
- 2.2.8 Multi-dimensional Scaling (MDS)
- 2.2.9 t-distributed Stochastic Neighbor Embedding (t-SNE)
- 2.2.10 Tips on practical use

### [2.3 Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- 2.3.1 Overview of clustering methods
- 2.3.2 K-means
- 2.3.3 Affinity Propagation
- 2.3.4 Mean Shift
- 2.3.5 Spectral clustering
- 2.3.6 Hierarchical clustering
- 2.3.7 DBSCAN
- 2.3.8 HDBSCAN
- 2.3.9 OPTICS
- 2.3.10 BIRCH
- 2.3.11 Clustering performance evaluation

### [2.4 Biclustering](https://scikit-learn.org/stable/modules/biclustering.html)
- 2.4.1 Spectral Co-Clustering
- 2.4.2 Spectral Biclustering
- 2.4.3 Biclustering evaluation

### [2.5 Decomposing Signals in Components (Matrix Factorization)](https://scikit-learn.org/stable/modules/decomposition.html)
- 2.5.1 Principal component analysis (PCA)
- 2.5.2 Kernel Principal Component Analysis (kPCA)
- 2.5.3 Truncated SVD and latent semantic analysis
- 2.5.4 Dictionary Learning
- 2.5.5 Factor Analysis
- 2.5.6 Independent component analysis (ICA)
- 2.5.7 Non-negative matrix factorization (NMF)
- 2.5.8 Latent Dirichlet Allocation (LDA)

### [2.6 Covariance Estimation](https://scikit-learn.org/stable/modules/covariance.html)
- 2.6.1 Empirical covariance
- 2.6.2 Shrunk Covariance
- 2.6.3 Sparse inverse covariance
- 2.6.4 Robust Covariance Estimation

### [2.7 Novelty and Outlier Detection](https://scikit-learn.org/stable/modules/outlier_detection.html)
- 2.7.1 Overview of outlier detection methods
- 2.7.2 Novelty Detection
- 2.7.3 Outlier Detection
- 2.7.4 Novelty detection with Local Outlier Factor

### [2.8 Density Estimation](https://scikit-learn.org/stable/modules/density.html)
- 2.8.1 Density Estimation: Histograms
- 2.8.2 Kernel Density Estimation

### [2.9 Neural Network Models (Unsupervised)](https://scikit-learn.org/stable/modules/neural_networks_unsupervised.html)
- 2.9.1 Restricted Boltzmann machines

---

## 3. Model Selection and Evaluation
https://scikit-learn.org/stable/model_selection.html

### [3.1 Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- 3.1.1 Computing cross-validated metrics
- 3.1.2 Cross validation iterators
- 3.1.3 A note on shuffling
- 3.1.4 Cross validation and model selection
- 3.1.5 Permutation test score

### [3.2 Tuning Hyper-Parameters](https://scikit-learn.org/stable/modules/grid_search.html)
- 3.2.1 Exhaustive Grid Search
- 3.2.2 Randomized Parameter Optimization
- 3.2.3 Searching with successive halving
- 3.2.4 Tips for parameter search
- 3.2.5 Alternatives to brute force parameter search

### [3.3 Tuning the Decision Threshold](https://scikit-learn.org/stable/modules/classification_threshold.html)
- 3.3.1 Post-tuning the decision threshold

### [3.4 Metrics and Scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)
- 3.4.1 Which scoring function should I use?
- 3.4.2 Scoring API overview
- 3.4.3 The `scoring` parameter
- 3.4.4 Classification metrics
- 3.4.5 Multilabel ranking metrics
- 3.4.6 Regression metrics
- 3.4.7 Clustering metrics
- 3.4.8 Dummy estimators

### [3.5 Validation Curves](https://scikit-learn.org/stable/modules/learning_curve.html)
- 3.5.1 Validation curve
- 3.5.2 Learning curve

---

## 4. Metadata Routing
https://scikit-learn.org/stable/metadata_routing.html

- 4.1 Usage Examples (weighted scoring/fitting, feature selection)
- 4.2 API Interface
- 4.3 Metadata Routing Support Status

---

## 5. Inspection
https://scikit-learn.org/stable/inspection.html

### [5.1 Partial Dependence and ICE Plots](https://scikit-learn.org/stable/modules/partial_dependence.html)
- 5.1.1 Partial dependence plots
- 5.1.2 Individual conditional expectation (ICE) plot
- 5.1.3 Mathematical Definition
- 5.1.4 Computation methods

### [5.2 Permutation Feature Importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
- 5.2.1 Outline of the permutation importance algorithm
- 5.2.2 Relation to impurity-based importance in trees
- 5.2.3 Misleading values on strongly correlated features

---

## 6. Visualizations
https://scikit-learn.org/stable/visualizations.html

- 6.1 Available Plotting Utilities / Display Objects

---

## 7. Dataset Transformations
https://scikit-learn.org/stable/data_transforms.html

### [7.1 Pipelines and Composite Estimators](https://scikit-learn.org/stable/modules/compose.html)
- 7.1.1 Pipeline: chaining estimators
- 7.1.2 Transforming target in regression
- 7.1.3 FeatureUnion: composite feature spaces
- 7.1.4 ColumnTransformer for heterogeneous data
- 7.1.5 Visualizing Composite Estimators

### [7.2 Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html)
- 7.2.1 Loading features from dicts
- 7.2.2 Feature hashing
- 7.2.3 Text feature extraction
- 7.2.4 Image feature extraction

### [7.3 Preprocessing Data](https://scikit-learn.org/stable/modules/preprocessing.html)
- 7.3.1 Standardization, mean removal and variance scaling
- 7.3.2 Non-linear transformation
- 7.3.3 Normalization
- 7.3.4 Encoding categorical features
- 7.3.5 Discretization
- 7.3.6 Imputation of missing values
- 7.3.7 Generating polynomial features
- 7.3.8 Custom transformers

### [7.4 Imputation of Missing Values](https://scikit-learn.org/stable/modules/impute.html)
- 7.4.1 Univariate vs. Multivariate Imputation
- 7.4.2 Univariate feature imputation
- 7.4.3 Multivariate feature imputation
- 7.4.4 Nearest neighbors imputation
- 7.4.5 Keeping the number of features constant
- 7.4.6 Marking imputed values
- 7.4.7 Estimators that handle NaN values

### [7.5 Unsupervised Dimensionality Reduction](https://scikit-learn.org/stable/modules/unsupervised_reduction.html)
- 7.5.1 PCA
- 7.5.2 Random projections
- 7.5.3 Feature agglomeration

### [7.6 Random Projection](https://scikit-learn.org/stable/modules/random_projection.html)
- 7.6.1 The Johnson-Lindenstrauss lemma
- 7.6.2 Gaussian random projection
- 7.6.3 Sparse random projection
- 7.6.4 Inverse Transform

### [7.7 Kernel Approximation](https://scikit-learn.org/stable/modules/kernel_approximation.html)
- 7.7.1 Nystroem Method
- 7.7.2 Radial Basis Function Kernel
- 7.7.3 Additive Chi Squared Kernel
- 7.7.4 Skewed Chi Squared Kernel
- 7.7.5 Polynomial Kernel Approximation via Tensor Sketch

### [7.8 Pairwise Metrics, Affinities and Kernels](https://scikit-learn.org/stable/modules/metrics.html)
- Cosine similarity, Linear, Polynomial, Sigmoid, RBF, Laplacian, Chi-squared kernels

### [7.9 Transforming the Prediction Target (y)](https://scikit-learn.org/stable/modules/preprocessing_targets.html)
- 7.9.1 Label binarization
- 7.9.2 Label encoding

---

## 8. Dataset Loading Utilities
https://scikit-learn.org/stable/datasets.html

### [8.1 Toy Datasets](https://scikit-learn.org/stable/datasets/toy_dataset.html)
- Iris, Diabetes, Digits, Linnerrud, Wine, Breast cancer

### [8.2 Real World Datasets](https://scikit-learn.org/stable/datasets/real_world.html)
- Olivetti faces, 20 newsgroups, LFW faces, Forest covertypes, RCV1, Kddcup 99, California Housing, Species distribution

### [8.3 Generated Datasets](https://scikit-learn.org/stable/datasets/sample_generators.html)
- Generators for classification, clustering, regression, manifold learning, decomposition

### [8.4 Loading Other Datasets](https://scikit-learn.org/stable/datasets/loading_other_datasets.html)
- Sample images, svmlight/libsvm format, OpenML, external datasets

---

## 9. Computing with Scikit-Learn
https://scikit-learn.org/stable/computing.html

### [9.1 Scaling Strategies](https://scikit-learn.org/stable/computing/scaling_strategies.html)
- Out-of-core learning

### [9.2 Computational Performance](https://scikit-learn.org/stable/computing/computational_performance.html)
- Prediction latency, throughput, tips and tricks

### [9.3 Parallelism and Configuration](https://scikit-learn.org/stable/computing/parallelism.html)
- Parallelism, configuration switches

---

## 10. Model Persistence
https://scikit-learn.org/stable/model_persistence.html

- 10.1 Workflow Overview
- 10.2 ONNX
- 10.3 skops.io
- 10.4 pickle, joblib, and cloudpickle
- 10.5 Security & Maintainability Limitations

---

## 11. Common Pitfalls and Recommended Practices
https://scikit-learn.org/stable/common_pitfalls.html

- 11.1 Inconsistent preprocessing
- 11.2 Data Leakage (how to avoid, leakage during pre-processing)
- 11.3 Controlling Randomness

---

## 12. Dispatching / Array API Support
https://scikit-learn.org/stable/dispatching.html

### [12.1 Array API Support (Experimental)](https://scikit-learn.org/stable/modules/array_api.html)
- Enabling array API support, example usage, input/output handling

---

## 13. Choosing the Right Estimator
https://scikit-learn.org/stable/machine_learning_map.html

> Interactive flowchart / cheat sheet for selecting the right algorithm based on data type, size, and task.

---

## 14. External Resources, Videos and Talks
https://scikit-learn.org/stable/presentations.html

- scikit-learn MOOC
- Videos
- New to Scientific Python?
- External Tutorials
