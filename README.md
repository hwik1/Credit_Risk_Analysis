# Credit Risk Analysis
Module 17 Challenge prepared by Hannah Wikum - April 2022
___
## Resources
Data Source: LoanStats_2019Q1.csv (provided)

Software: Jupyter Notebook, Python 3.7.11 with Pandas 1.3.5, NumPy 1.20.3, SciPy 1.7.3, Scikit-learn 1.0.2
___
## Overview
### Background
This analysis was performed to create a machine learning model to identify credit risk. A CSV of various loan statistics was provided by LendingClub, a peer-to-peer lending service company. This file included information like the loan amout, annual income, home ownership status, and much more. The goal was to create a machine learning model using various oversampling, undersampling, and ensemble learning methods to determine which model best identified the target of loan status. For this challenge, I used Naive Random Oversampling, SMOTE Oversampling, Cluster Centroid Undersampling, SMOTEENN (combination of over and undersampling), Balanced Random Forest Classifier, and then Easy Ensemble AdaBoost Classifier.

### Methodology
To create the models, I started by reading the data from the CSV into a Pandas DataFrame in my Juptyer Notebook. To prep the data, I used provided code to drop null columns and rows, convert interest rates to a numerical value, and changed the values in the target column of loan status to either low risk (for any value where the loan is current) or high risk (anything late, in default, or in grace period). Next step was to define features and the target. For the features (X), I used pd.get_dummies to convert columns from the DataFrame with string values to numerical so they could be handled by the machine learning model. I dropped the loan status column because that is what we want the model to predict. Next, I created the target value (y) by assigning the loan status column to y and using ravel() to convert to an array. Finally, I imported train_test_split from sklearn.model_selection and split the data into X_train, x_test, y_train, and y_test. For all models, I used a random state of 1 for consistency.

The steps described above set-up the data to be resampled with various machine learning models. To train models using oversampling, undersampling, a combination, or ensemble learning, I resampled the data, confirmed the counter between low risk and high risk changed as expected, trained the model using the resampled data, and then displayed the balanced accuracy score, confusion matrix, and imbalanced classification report to check the efficacy of each model.
___
## Results
After running all six machine learning models, here is a comparison of the balanced accuracy score, precision, and recall scores by model:

 * **Random Oversampling** is a technique that involves randomly sampling from the minority high risk class so you have equal amounts of minority inputs as the majority low risk. In this case, we started with 51,366 low risk training data points, so random oversampling increased the amount of high risk to be equal. As you can see from the picture of my code below, the balanced accuracy score was 0.645. For purposes of this analysis, I am going to focus on the precision and recall score for the high risk class because it is most important to correctly predict high risk loans. The precision for high risk was only 0.01, which means that if a high risk data point was identified, it was only actually high risk about 1% of the time. The recall score for high risk was 0.70, which means that if a loan was high risk, it was only correctly identified as such 70% of the time. In conclusion, this was not a great machine learning model because it was overly aggressive, yet still missed some high risk data points.

  _Random Oversampling Model Results_
 
  ![image](https://user-images.githubusercontent.com/93058069/162595993-c42c6ed2-80ee-4f45-8fff-925d67c2d70c.png)

 * **SMOTE Oversampling** is similar to Random Oversampling, but instead involves creating synthetic data points for the minority class in order to have enough data points in the minority class to equal the majority. Since we started with the same 51,366 low risk training data points, SMOTE oversampling also yielded 51,366 high risk training points so the two formerly imbalanced classes were equal. The balanced accuracy score was 0.662, which was slightly higher than the first model. The precision for high risk was 0.01 (similar to the first model) and the recall score was 0.63. The overall recall was better than the first oversampling technique at 0.69 compared to 0.59 because this model was less aggressive and fewer low risk were misclassified as high risk.

  _SMOTE Oversampling Model Results_

 ![image](https://user-images.githubusercontent.com/93058069/162596029-3ec87465-0c4a-480b-a8ac-432f34a05cfc.png)

 * **Cluster Centroids** is an undersampling technique where we scale down the number of data points in the majority class by identifying clusters of low risk values to scale down the majority data to the size of the minority. Since high risk only had 246 entries in the training data, low risk entries were reduced down to 246, as seen in the counter. The balanced accuracy score on this model was 0.544, which is definitely poorer compared to the oversampling methods. The precision was the same as the other two models and the recall score was 0.69 for high risk. This model was super aggressive because more low risk loans were inaccurately labeled as high than were correctly identified as low. Therefore, the overall recall score suffered at only 0.40.

 _Cluster Centroids Undersampling Model Results_

![image](https://user-images.githubusercontent.com/93058069/162596079-e41134f4-251c-4a07-97c2-1534f701da92.png)

 * **SMOTEENN** is the name of an algorithm that combines oversampling with undersampling. SMOTEENN uses the SMOTE oversampling technique mentioned in the second model to oversample the minority class using newly created data points, but cleans the data using Edited Nearest Neighbors (ENN) algorithm to drop any data points if the nearest neighbors belong to the other class. The goal with this technique is to reduce the impact of outliers on the final model and try to minimize the downsides of both oversampling and undersampling techniques. The SMOTEENN model generated the best confusion matrix of the first four models in terms of correctly identifying the most high risk loans. The balanced accuracy score was 0.645. The precision scores were the same as the other models. The high risk recall was 0.77, which is higher than anything generated with the first three models, but the overall recall was only 0.59 because it was still overly aggressive on classifying low risk data points.
 
  _SMOTEENN Model Results_
 
 ![image](https://user-images.githubusercontent.com/93058069/162596203-04091b4f-ceca-4b7a-9dd1-578e980a30c0.png)

 * **Balanced Random Forest Classifier** is an ensemble learning technique that creates simple decision trees based on the features and aggregates the resulting votes to make a classification decision. For this model, I used 100 trees (or estimators). By using the Balanced Random Forest Classifier, the model creates a bootstrap sample that is balanced between high risk and low risk data points by randomly undersampling. Compared to the first four models, this model had the highest balanced accuracy score at 0.803. That is a clear improvement over the first four models. The overall precision was the same as all other models at 0.99, but the high risk was slightly better at 0.03 (still not great). Finally, the recall for high risk was 0.73 with overall at 0.87.

  _Balanced Random Forest Classifier Model Results_
 
  ![image](https://user-images.githubusercontent.com/93058069/162596298-4715cb4e-beb7-4bd9-be25-83ac8952978b.png)

 * **Easy Ensemble AdaBoost Classifier** was the final machine learning model tested. This models works by creating samples of the training dataset that are balanced because all minority class examples are included, while only a random selection of the majority class are used. Instead of running multiple decision trees at once like in the previous example, the AdaBoost component means that we focus on running one decision tree at a time and adjusting in subsequent models to minimize errors. This model had the best score across all metrics. The balanced accuracy score was 0.932. 93 out of 101 high risk data points were correctly identified in the confusion matrix. The total precision score of 0.99 was the same as other models becasuue it is so highly weighted toward the number of low risk, but high risk precision reached 0.09. Recall for high risk, low risk, and overall was all over 0.90 at 0.92, 0.94, and 0.94, respectively.

  _Easy Ensemble AdaBoost Classifier Model Results_
 
  ![image](https://user-images.githubusercontent.com/93058069/162596343-b8064a13-4743-4c9f-a1c9-2c25ef1bf830.png)

___
## Summary
After running all six machine learning models, the ensemble models clearly performed better than the oversampling, undersampling, or combo models. Those models were all overly-aggressive in misclassifying low risk as high risk, yet still missed a lot of the true high risk data points. The scores improved for the ensemble models, but there was still a clear winner. The Easy Ensemble AdaBoost Classifier had the highest balanced accuracy, precision, and recall scores. It was most likely to correctly identify high risk and low risk - fewer than 1,000 low risk were misclassified as high. However, if I had to make a recommendation to LendingClub on which model to use, I would not recommend any of them. Although the Easy Ensemble AdaBoost Classifier was the best of the six we tested, it still misclassified 959 or 5.6% of true low risk loans as high risk. That could cause the company to give the customer uncompetitive terms or outright deny the loan, even if it could have been profitable. The risk of late payments or defaults are inherent with lending, but you need a large base of profitable, low risk loans to counteract it.
