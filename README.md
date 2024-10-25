# Fake News Detection Using RNN

## Overview
This project implements a deep learning model to detect fake news articles using Recurrent Neural Networks (RNNs). The model is trained on a dataset of labeled news articles, where articles are classified as either fake (1) or real (0).

## Dataset
The project uses two datasets:
- `Fake.csv`: Articles labeled as fake news.
- `True.csv`: Articles labeled as real news.

These datasets are combined into a single DataFrame.

## Installation
To run this project, install the required dependencies:

```bash
pip install pandas numpy matplotlib tensorflow scikit-learn nltk

Usage

	1.	Load the Datasets: Load and label the datasets.
	2.	Preprocess the Data: Clean and preprocess the text data.
	3.	Tokenization and Padding: Tokenize the text data and pad sequences.
	4.	Model Training: Train the RNN model.
	5.	Evaluate the Model: Assess performance using accuracy, precision, and recall.

Model Architecture

The model consists of:

	•	Embedding Layer: Converts words into dense vectors.
	•	LSTM Layer: Learns from sequential dependencies.
	•	Dropout Layer: Prevents overfitting.
	•	Dense Layer: Outputs binary classification.

Results

The model’s performance metrics include:

	•	Test Accuracy: 0.9990
	•	Precision: 0.9961
	•	Recall: 0.9936

Hyperparameter Tuning

Hyperparameters were optimized through cross-validation to enhance model performance.


