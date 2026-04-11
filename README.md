# Machine Learning Interview Preparation

[![Machine Learning](https://img.shields.io/badge/ML-fundamentals-blue.svg)](https://github.com/BrendanJamesLynskey/Interview_Machine_Learning)

A comprehensive interview preparation resource covering classical machine learning, deep learning theory, training methodologies, optimisation techniques, and deployment strategies.

## Contents

- [01 Classical ML](#01-classical-ml)
- [02 Deep Learning Foundations](#02-deep-learning-foundations)
- [03 CNNs and RNNs](#03-cnns-and-rnns)
- [04 Training and Optimisation](#04-training-and-optimisation)
- [05 Evaluation and Deployment](#05-evaluation-and-deployment)
- [06 Implementation](#06-implementation)
- [07 Quizzes](#07-quizzes)
- [How to Use](#how-to-use)
- [Contributing](#contributing)
- [Related Repositories](#related-repositories)

## 01 Classical ML

Foundational machine learning algorithms and techniques.

- [Supervised vs Unsupervised Learning](./01_classical_ml/supervised_vs_unsupervised.md)
- [Linear Regression and Regularisation](./01_classical_ml/linear_regression_and_regularisation.md)
- [Logistic Regression](./01_classical_ml/logistic_regression.md)
- [Decision Trees and Ensembles](./01_classical_ml/decision_trees_and_ensembles.md)
- [SVM and Kernels](./01_classical_ml/svm_and_kernels.md)
- [Clustering: K-Means and DBSCAN](./01_classical_ml/clustering_kmeans_dbscan.md)

### Worked Problems

- [Problem 01: Bias-Variance Trade-off](./01_classical_ml/worked_problems/problem_01_bias_variance.md)
- [Problem 02: Regularisation Derivation](./01_classical_ml/worked_problems/problem_02_regularisation_derivation.md)
- [Problem 03: Feature Engineering](./01_classical_ml/worked_problems/problem_03_feature_engineering.md)

## 02 Deep Learning Foundations

Core concepts for deep neural networks.

- [Neural Network Basics](./02_deep_learning_foundations/neural_network_basics.md)
- [Backpropagation Derivation](./02_deep_learning_foundations/backpropagation_derivation.md)
- [Activation Functions](./02_deep_learning_foundations/activation_functions.md)
- [Loss Functions](./02_deep_learning_foundations/loss_functions.md)
- [Weight Initialisation](./02_deep_learning_foundations/weight_initialisation.md)
- [Attention and Transformers](./02_deep_learning_foundations/attention_and_transformers.md)

### Worked Problems

- [Problem 01: Backpropagation by Hand](./02_deep_learning_foundations/worked_problems/problem_01_backprop_by_hand.md)
- [Problem 02: Vanishing Gradients](./02_deep_learning_foundations/worked_problems/problem_02_vanishing_gradients.md)
- [Problem 03: Loss Function Choice](./02_deep_learning_foundations/worked_problems/problem_03_loss_function_choice.md)

## 03 CNNs and RNNs

Specialised architectures for structured data.

- [Convolutional Neural Networks](./03_cnns_and_rnns/convolutional_neural_networks.md)
- [CNN Architectures: ResNet and EfficientNet](./03_cnns_and_rnns/cnn_architectures_resnet_efficientnet.md)
- [Recurrent Networks: LSTM and GRU](./03_cnns_and_rnns/recurrent_networks_lstm_gru.md)
- [Sequence to Sequence](./03_cnns_and_rnns/sequence_to_sequence.md)

### Worked Problems

- [Problem 01: Convolution Output Dimensions](./03_cnns_and_rnns/worked_problems/problem_01_conv_output_dimensions.md)
- [Problem 02: Receptive Field](./03_cnns_and_rnns/worked_problems/problem_02_receptive_field.md)
- [Problem 03: LSTM Gates](./03_cnns_and_rnns/worked_problems/problem_03_lstm_gates.md)

## 04 Training and Optimisation

Techniques for effective model training.

- [SGD, Adam and Variants](./04_training_and_optimisation/sgd_adam_and_variants.md)
- [Learning Rate Scheduling](./04_training_and_optimisation/learning_rate_scheduling.md)
- [Batch Normalisation](./04_training_and_optimisation/batch_normalisation.md)
- [Dropout and Regularisation](./04_training_and_optimisation/dropout_and_regularisation.md)
- [Data Augmentation](./04_training_and_optimisation/data_augmentation.md)

### Worked Problems

- [Problem 01: Optimiser Comparison](./04_training_and_optimisation/worked_problems/problem_01_optimiser_comparison.md)
- [Problem 02: Learning Rate Warmup](./04_training_and_optimisation/worked_problems/problem_02_learning_rate_warmup.md)
- [Problem 03: Overfitting Diagnosis](./04_training_and_optimisation/worked_problems/problem_03_overfitting_diagnosis.md)

## 05 Evaluation and Deployment

Model evaluation metrics and production deployment.

- [Metrics: Precision, Recall and F1](./05_evaluation_and_deployment/metrics_precision_recall_f1.md)
- [Cross Validation](./05_evaluation_and_deployment/cross_validation.md)
- [Model Selection](./05_evaluation_and_deployment/model_selection.md)
- [Quantisation and Pruning](./05_evaluation_and_deployment/quantisation_and_pruning.md)
- [ONNX and Inference Runtimes](./05_evaluation_and_deployment/onnx_and_inference_runtimes.md)

### Worked Problems

- [Problem 01: Confusion Matrix](./05_evaluation_and_deployment/worked_problems/problem_01_confusion_matrix.md)
- [Problem 02: Imbalanced Classes](./05_evaluation_and_deployment/worked_problems/problem_02_imbalanced_classes.md)
- [Problem 03: Deployment Pipeline](./05_evaluation_and_deployment/worked_problems/problem_03_deployment_pipeline.md)

## 06 Implementation

Coding challenges to reinforce concepts.

- [Challenge 01: Linear Regression](./06_implementation/coding_challenges/challenge_01_linear_regression.py)
- [Challenge 02: Neural Network from Scratch](./06_implementation/coding_challenges/challenge_02_neural_network_from_scratch.py)
- [Challenge 03: CNN with PyTorch](./06_implementation/coding_challenges/challenge_03_cnn_pytorch.py)
- [Challenge 04: Training Loop](./06_implementation/coding_challenges/challenge_04_training_loop.py)
- [Challenge 05: Gradient Descent Variants](./06_implementation/coding_challenges/challenge_05_gradient_descent_variants.py)

## 07 Quizzes

Self-assessment quizzes to test knowledge.

- [Quiz: Classical ML](./07_quizzes/quiz_classical_ml.md)
- [Quiz: Deep Learning](./07_quizzes/quiz_deep_learning.md)
- [Quiz: Training](./07_quizzes/quiz_training.md)
- [Quiz: Deployment](./07_quizzes/quiz_deployment.md)

## How to Use

This repository is structured as a comprehensive study guide for machine learning interview preparation:

1. **Start with fundamentals**: Begin with classical ML to understand core concepts like regression, classification, and clustering.

2. **Progress to deep learning**: Move through neural network foundations, backpropagation, and common architectures.

3. **Study advanced topics**: Explore CNNs and RNNs for specialised use cases, then training techniques for optimal performance.

4. **Practice problems**: Each section includes worked problems demonstrating practical applications and derivations.

5. **Implement and test**: Use the coding challenges to implement algorithms from scratch and with frameworks like PyTorch.

6. **Self-assess**: Complete quizzes to verify understanding before interviews.

7. **Reference**: Use this repository as a reference during interview preparation and for quick topic reviews.

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch for your changes
3. Ensure clarity and accuracy of content
4. Submit a pull request with a clear description

Please maintain consistency with the existing structure and documentation style.

## Related Repositories

- [Interview_Transformer_Architecture](https://github.com/BrendanJamesLynskey/Interview_Transformer_Architecture)
- [LLM_Transformer_Decoder_guide](https://github.com/BrendanJamesLynskey/LLM_Transformer_Decoder_guide)
- [LLM_articles](https://github.com/BrendanJamesLynskey/LLM_articles)

Visit [BrendanJamesLynskey](https://github.com/BrendanJamesLynskey) on GitHub for more resources.
