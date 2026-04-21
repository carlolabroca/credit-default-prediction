# Credit Default Prediction

End-to-end machine learning project on the UCI Credit Card Default dataset (30,000 records).

## Business Problem
Predict which customers are likely to default on their credit card payment next month, enabling financial institutions to proactively manage credit risk.

## Dataset
- **Source:** UCI Machine Learning Repository — Default of Credit Card Clients
- **Size:** 30,000 records, 24 features
- **Target:** Binary — default payment next month (22% positive class)

## Project Structure
- Exploratory Data Analysis (EDA)
- Feature Engineering (bill_trend, pay_ratio, total_paid, total_delay)
- Multicollinearity analysis
- Model training and evaluation

## Models & Results

| Model | AUC |
|---|---|
| Logistic Regression (default) | 0.728 |
| Logistic Regression (balanced) | 0.729 |
| Random Forest (default) | 0.760 |
| Random Forest (optimized) | 0.780 |

## Key Findings
- PAY_0 (recent payment status) and total_delay are the most predictive features
- Class imbalance addressed via class_weight='balanced'
- GridSearchCV used for hyperparameter tuning

## Tech Stack
Python · Pandas · Scikit-learn · Matplotlib · Seaborn
