# CT1 MLOps Individual Assignment - Bank Term Deposit Prediction

## 1. Business Problem

A leading bank runs marketing campaigns to offer term deposits to customers. Contacting every customer is expensive and time consuming, and only a small share of customers accept the offer. The objective is to build a model that predicts which customers are most likely to subscribe, so the bank can prioritize those leads and improve campaign efficiency.

## 2. Dataset

The dataset contains client demographics, banking information, previous campaign interactions, and macroeconomic indicators, along with the target variable `y` indicating whether the client subscribed to the term deposit ("yes" or "no").

## 3. Model

- Algorithm: Logistic Regression with class weighting for imbalance
- Preprocessing:
  - Numeric features scaled using StandardScaler
  - Categorical features one hot encoded
- Evaluation metrics:
  - ROC AUC
  - Precision, recall, F1
  - Confusion matrix

The final model is saved as `bank_term_deposit_model.joblib`.

<img width="670" height="681" alt="Screenshot 2025-12-16 at 11 37 25 PM" src="https://github.com/user-attachments/assets/cca16f99-fe7b-44d1-a494-f61711e0b169" />


## 4. Local Setup

```bash
pip install -r requirements.txt
python app.py
# API will run at http://localhost:8000
