# Machine Learning Basics

## What is Machine Learning?

Machine learning (ML) is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. Instead of writing rules by hand, ML algorithms find patterns in data.

## Types of Machine Learning

### Supervised Learning
The model learns from labeled training data, mapping inputs to known outputs.
- **Classification**: Predicts a category (e.g., spam vs. not spam, cat vs. dog).
- **Regression**: Predicts a continuous value (e.g., house prices, stock returns).
- Common algorithms: Linear Regression, Decision Trees, Random Forests, SVM, Neural Networks.

### Unsupervised Learning
The model finds hidden structure in unlabeled data.
- **Clustering**: Groups similar data points (e.g., customer segmentation). K-Means and DBSCAN are popular.
- **Dimensionality Reduction**: Compresses data while preserving structure (PCA, t-SNE, UMAP).
- **Anomaly Detection**: Identifies outliers that deviate from normal patterns.

### Reinforcement Learning
An agent learns to make decisions by interacting with an environment and receiving rewards or penalties. Used in game-playing AI (AlphaGo, Atari), robotics, and autonomous vehicles.

## Key Concepts

- **Training data**: The dataset used to fit the model.
- **Validation data**: Used to tune hyperparameters and detect overfitting.
- **Test data**: Held-out set for final performance evaluation.
- **Overfitting**: The model memorizes training data but fails to generalize.
- **Underfitting**: The model is too simple to capture the underlying patterns.
- **Gradient Descent**: An optimization algorithm that iteratively adjusts model weights to minimize a loss function.

## Neural Networks

Neural networks are inspired by biological neurons. A deep neural network stacks many layers:
- **Input layer**: Receives raw features.
- **Hidden layers**: Learn intermediate representations.
- **Output layer**: Produces predictions.

Activation functions like ReLU, Sigmoid, and Softmax introduce non-linearity, allowing networks to learn complex relationships.

## Evaluation Metrics

- **Accuracy**: Fraction of correct predictions (good for balanced classes).
- **Precision / Recall / F1**: Better for imbalanced datasets.
- **Mean Squared Error (MSE)**: Common for regression tasks.
- **ROC-AUC**: Measures classifier discrimination ability.
