 Loan Approval Prediction using Logistic Regression and XGBoost
Overview
This project implements a machine learning model to predict loan approval based on applicant details. It combines Logistic Regression (LR) and XGBoost (XGB) for improved accuracy.

Features
Data Preprocessing: Handles missing values, encodes categorical data, and scales numerical features.
Model Training:
Uses Logistic Regression to predict probabilities.
Feeds these probabilities as additional features to an XGBoost Classifier for final prediction.
Evaluation: Reports accuracy, F1-score, precision, and confusion matrix.
User Input Function: Allows manual input for loan prediction.
Dataset
The model expects a CSV file named loan_approval_dataset.csv.
Features used:
Applicant Info: Education, Self-Employment, Dependents, Income
Loan Details: Loan Amount, Loan Term, CIBIL Score
Asset Details: Residential, Commercial, Luxury, and Bank Assets
