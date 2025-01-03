# SMS Spam Detection

This repository contains an implementation of an SMS spam detection system. The system classifies SMS messages into two categories: **Spam** and **Ham** (not spam). The project involves data preprocessing, exploratory data analysis (EDA), text vectorization, machine learning model training, and evaluation.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Workflow](#project-workflow)

## Introduction
The SMS Spam Detection project utilizes machine learning techniques to identify spam messages. It preprocesses raw text messages to clean and tokenize them, transforms the text data into numerical features using TF-IDF vectorization, and applies multiple classification algorithms to determine the best-performing model.

## Features
- Data Cleaning: Removed unnecessary columns, duplicates, and missing values.
- Exploratory Data Analysis (EDA): Visualized data distribution and correlations.
- Text Preprocessing: Lowercased text, removed special characters, stop words, and applied stemming.
- Model Training: Trained multiple classifiers, including Naive Bayes, SVM, Random Forest, and others.
- Evaluation: Compared models based on accuracy and precision.
- Deployment: Saved the best model using `pickle` for deployment.

## Technologies Used
- **Python**: Programming language.
- **Pandas**: Data manipulation.
- **NumPy**: Numerical computations.
- **Matplotlib** & **Seaborn**: Data visualization.
- **NLTK**: Text preprocessing.
- **Scikit-learn**: Machine learning models and evaluation.
- **XGBoost**: Advanced gradient boosting.
- **Pickle**: Model serialization.

## Project Workflow
1. **Data Cleaning**:
   - Removed unnecessary columns.
   - Checked for duplicates and missing values.
   - Renamed columns for clarity.

2. **EDA**:
   - Analyzed the class distribution.
   - Explored relationships between text features and target labels.

3. **Text Preprocessing**:
   - Tokenization, lowercasing, stopword removal, and stemming.
   - Created new features: number of characters, words, and sentences.

4. **Feature Extraction**:
   - Used **TF-IDF Vectorization** to convert text into numerical features.

5. **Model Training**:
   - Trained multiple classifiers (e.g., Naive Bayes, Random Forest, SVM).
   - Fine-tuned hyperparameters.

6. **Evaluation**:
   - Compared models based on accuracy and precision.
   - Visualized performance using bar charts.

7. **Model Deployment**:
   - Saved the best-performing model and vectorizer using `pickle`.
