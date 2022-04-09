# Credit Risk Analysis
Module 17 Challenge prepared by Hannah Wikum - April 2022
___
## Resources
Data Source: LoanStats_2019Q1.csv (provided)

Software: Jupyter Notebook, Python 3.7.11 with Pandas 1.3.5, NumPy 1.20.3, SciPy 1.7.3, Scikit-learn 1.0.2
___
## Overview
### Background
This analysis was performed to create a machine learning model to identify credit risk. A CSV of various loan statistics was provided by LendingClub, a peer-to-peer lending service company. This file included information like the loan amout, annual income, home ownership status, and much more. The goal was to create a machine learning model using various oversampling, undersampling, and ensemble learning methods to determine which model best identified the target of loan status. For this challenge, I used Naive Random Oversampling, SMOTE Oversampling, Cluster Centroid Undersampling, SMOTEENN (combinationg of over and undersampling), Balanced Random Forecast Classifier, and then Easy Ensemble AdaBoost Classifier.

### Methodology
To create the models, I started by reading the data from the CSV into a Pandas DataFrame in my Juptyer Notebook. To prep the data, I used provided code to drop null columns and rows, convert interest rates to a numerical value, and changed the values in the target column of loan status to either low risk (for any value where the loan is current) or high risk (anything late, in default, or in grace period). Next step was to define features and the target. For the features (X), I used pd.get_dummies to convert columns from the DataFrame with string values to numerical so they could be handled by the machine learning model. I dropped the loan status column because that is what we want the model to predict. Next, I created the target value (y) by assigning the loan status column to y and using ravel() to convert to an array. Finally, I imported train_test_split from sklearn.model_selection and split the data into X_train, x_test, y_train, and y_test. For all models, I used a random state of 1 for consistency.

The steps described above set-up the data to be resampled with various machine learning models. To train models using oversampling, undersampling, a combination, or ensemble learning, I resampled the data, confirmed the counter between low risk and high risk changed as expected, trained the model using the resampled data, and then displayed the balanced accuracy score, confusion matrix, and imbalanced classification report to check the efficacy of each model.
___
## Results
After running all six machine learning models, here is a comparison of the balanced accuracy score, precision, and recall scores by model:

 * **Random Oversampling** is a technique that involves randomly sampling from the minority high risk class so you have equal amounts of minority inputs as the majority low risk. In this case, we started with 51,366 low risk training data points, so random oversampling increased the amount of high risk to be equal. The balanced accuracy score was XXXXXXXX. The precision was XXXXXXXX and the recall score was XXXXXXXX.

 * **SMOTE Oversampling** is similar to Random Oversampling, but instead involves creating synthetic data points for the minority class in order to have enough data points in the minority class to equal the majority. Since we started with the same 51,366 low risk training data points, SMOTE oversampling also yielded 51,366 high risk training points so the two formerly imbalanced classes were equal. The balanced accuracy score was XXXXXXXX. The precision was XXXXXXXX and the recall score was XXXXXXXX.

 * **Cluster Centroids** is an undersampling technique where we scale down the number of data points in the majority class by identifying clusters of low risk values to scale down the majority data to the size of the minority. Since high risk only had 246 entries in the training data, low risk entries were reduced down to 246, as seen in the counter. The balanced accuracy score on this model was XXXXXXX, which is XXXXXXXX compared to the oversampling methods. The precision was XXXXXXXX and the recall score was XXXXXXXX.

 * **SMOTEENN** is the name of an algorithm that combines oversampling with undersampling. SMOTEENN uses the SMOTE oversampling technique mentioned in the second model to oversample the minority class using newly created data points, but cleans the data using Edited Nearest Neighbors (ENN) algorithm to drop any data points if the nearest neighbors belong to the other class. The goal with this technique is to reduce the impact of outliers on the final model and try to minimize the downsides of both oversampling and undersampling techniques. The SMOTEENN model seemed XXXXXX accurate compared to the first three data learning models because the balanced accuracy score was XXXXXX. The precision was XXXXXXXX and the recall score was XXXXXXXX.

 * **Balanced Random Forest Classifier** is an ensemble learning techniques that creates simple decision trees based on the features and aggregates the resulting votes to make a classification decision. For this model, I used 100 trees (or estimators). By using the _Balanced_ Random Forest Classifier, the model creates a bootstrap sample that is balanced between high risk and low risk data points by randomly undersampling. Compared to the other models, this model provided XXXXX on the balanced accuracy score. The precision XXXX to XXXXXX, while recall was XXXXXX impacted to XXXXXXX.

 * **Easy Ensemble AdaBoost Classifier** was the final machine learning model tested. This models works by creating samples of the training dataset that are balanced because all minority class examples are included, while only a random selection of the majority class are used. Instead of running multiple decision trees at once like in the previous example, the AdaBoost component means that we focus on running one decision tree at a time and adjusting in subsequent models to minimize errors. The balanced accuracy score was XXXXXXXX. The precision was XXXXXXXX and the recall score was XXXXXXXX.
 
___
## Summary
After running all six machine learning models, XXXXXXXX had the highest accurracy score, XXXXXX the best precision, and XXXXXX the best recall score. I would recommend using XXXXXX model because XXXXXXXXXXXXXXXXX.
