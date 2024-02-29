# SentimentAnalysis
Project Description:
The project aims to perform sentiment analysis on product reviews using natural language processing (NLP) techniques and machine learning classification models. The dataset contains information about product reviews, including the text of the review, ratings, and other relevant features. The goal is to build a model that can classify reviews into positive, negative, or neutral sentiment categories based on the review text.

Working:

Data Preparation: Load the dataset and preprocess the text data by removing noise, such as special characters and stopwords, and performing tokenization and stemming.
Feature Engineering: Extract relevant features from the text data and encode categorical variables.
Model Building: Train machine learning classification models, such as Naive Bayes or LSTM, on the preprocessed text data to classify reviews into sentiment categories.
Model Evaluation: Evaluate the performance of the trained models using metrics such as precision, recall, F1-score, and accuracy.
Visualization: Visualize the results and insights obtained from the analysis using various plots and charts.
Interpretation: Interpret the findings and provide insights into the sentiment expressed in the product reviews.
Findings:

Analysis of sentiment distribution: Understand the distribution of sentiment categories in the dataset.
Feature importance: Identify the most important features that contribute to the classification of sentiment.
Model performance: Evaluate the performance of different classification models and compare their effectiveness in sentiment analysis.
Text analysis: Analyze the sentiment expressed in the product reviews and identify common themes or topics.
Models:

Naive Bayes Classifier: A simple probabilistic classifier based on Bayes' theorem, suitable for text classification tasks.
Long Short-Term Memory (LSTM) Network: A type of recurrent neural network (RNN) capable of learning long-term dependencies in sequential data, commonly used for NLP tasks.
Tests:

Cross-validation: Assess the generalization performance of the models using k-fold cross-validation.
Hyperparameter tuning: Optimize the hyperparameters of the models to improve their performance.
Model comparison: Compare the performance of different models based on various evaluation metrics.
Sensitivity analysis: Analyze the sensitivity of the models to changes in input data or parameters.

Precision, Recall, and F1-Score: The precision, recall, and F1-score for each sentiment class (1 to 5) provide insights into the model's performance for each sentiment category.

Sentiment class 1 has relatively high precision (0.89) but low recall (0.05), indicating that while the model correctly identifies instances of class 1, it misses many actual instances.
Sentiment classes 2, 3, and 4 have very low precision, recall, and F1-scores, suggesting that the model struggles to accurately classify instances belonging to these classes.
Sentiment class 5 has the highest precision, recall, and F1-score, indicating that the model performs well in identifying instances of this class, which represents positive sentiment.
Accuracy: The overall accuracy of the model on the testing set is 67.34%. While this accuracy may seem relatively high, it is important to note that the model's performance varies across different sentiment classes. The weighted average of precision, recall, and F1-score (55%) reflects the overall performance of the model across all sentiment classes, taking into account class imbalances.

Interpretation:

The model performs well in identifying positive sentiment (class 5), which is the majority class in the dataset, as indicated by high precision, recall, and F1-score.
However, the model struggles to accurately classify instances of other sentiment classes (1 to 4), especially classes with fewer instances. This suggests that the model may be biased towards predicting the majority class and fails to capture the nuances of different sentiment categories.
The relatively low precision and recall for classes 1, 2, 3, and 4 indicate that the model's performance for negative and neutral sentiment is poor. This could be due to various factors such as imbalanced dataset, insufficient feature representation, or model complexity.


About Dataset:

This dataset was collected via Python scraper in March 2023 and contains:

information about all beauty products (over 8,000) from the Sephora online store, including product and brand names, prices, ingredients, ratings, and all features.
user reviews (about 1 million on over 2,000 products) of all products from the Skincare category, including user appearances, and review ratings by other users. (I used 10000 reviews due to technical limitations)
