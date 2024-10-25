# Fake News Detection Using RNN

## Overview
This project implements a deep learning model to detect fake news articles using Recurrent Neural Networks (RNNs). The model is trained on a dataset of labeled news articles, where articles are classified as either fake (1) or real (0). The project involves data preprocessing, model architecture design, training, evaluation, and hyperparameter tuning.

## Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [License](#license)

## Dataset
The project uses two datasets:
- `Fake.csv`: Contains articles labeled as fake news.
- `True.csv`: Contains articles labeled as real news.

Both datasets are combined into a single DataFrame, labeled appropriately.

## Installation
To run this project, you will need to install the following dependencies:

```bash
pip install pandas numpy matplotlib tensorflow scikit-learn nltk

Usage

	1.	Load the Datasets: Load and label the datasets using the provided scripts.
	2.	Preprocess the Data: Clean and preprocess the text data (e.g., remove punctuation, convert to lowercase, etc.).
	3.	Tokenization and Padding: Tokenize the text data and pad sequences for model training.
	4.	Model Training: Train the RNN model using the preprocessed data.
	5.	Evaluate the Model: Assess the model’s performance using accuracy, precision, and recall metrics.

Model Architecture

The model architecture consists of:

	•	Embedding Layer: Converts words into dense vectors.
	•	LSTM Layer: Learns from sequential dependencies in the text.
	•	Dropout Layer: Helps prevent overfitting.
	•	Dense Layer: Outputs a binary classification result.

Code Snippet

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
model.add(LSTM(units=128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

Results

The model achieved the following performance metrics:

	•	Test Accuracy: XX.XX%
	•	Precision: XX.XX%
	•	Recall: XX.XX%

Hyperparameter Tuning

Hyperparameters such as LSTM units, dropout rates, batch sizes, and learning rates were tuned to optimize model performance. The best hyperparameters were determined through cross-validation.



