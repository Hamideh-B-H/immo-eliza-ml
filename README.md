

# 🏡 Real Estate Price Prediction in Belgium

A Machine Learning Pipeline for Accurate Property Valuation

## 📌 Project Overview

This project builds a full machine-learning pipeline to predict real estate prices in Belgium. The dataset was collected through a web-scraping project on Immovlan, completed collaboratively with three team members.

The goal is to create a model that can:

Clean and preprocess raw housing data

Train and compare different regression models

Evaluate performance using clear metrics

Generate reliable predictions for unseen properties

Save the best model for future use

The final model is stored as a .pkl file, allowing fast reuse without retraining.


## ⚙️ Environment & Libraries

This project runs on Python 3 and uses the following major libraries:
| Purpose           | Libraries               |
| ----------------- | ----------------------- |
| Data handling     | `pandas`, `numpy`       |
| Visualisation     | `matplotlib`, `seaborn` |
| Machine learning  | `scikit-learn`          |
| Gradient boosting | `xgboost`               |
| Model storage     | `pickle`                |


## 🏷️ Features

The most predictive variables extracted from the housing dataset:

number_rooms – Total number of rooms

living_area – Property size (m²)

property_type_name → encoded as property_house (house=1, apartment=0)

state_mapped → encoded as state_ready (ready to move in=1, to renovate=0)

postal_code – Target-encoded to capture location price effects


## 🤖 Models Trained

We trained and compared multiple regression models:

1️⃣ Linear Regression

A baseline model to establish initial performance.

2️⃣ Decision Tree Regressor

Captures non-linear patterns in the data.

3️⃣ XGBoost Regressor

A high-performance gradient boosting model.
This model achieved the best results and is saved as best_model.pkl for deployment.

Models were evaluated using:

MSE (Mean Squared Error)

MAE (Mean Absolute Error)

R² Score

## How to Run

1. Clone the repository:
```
git clone https://github.com/Hamideh-B-H/immo-eliza-ml
```
2. Install required libraries:
```
pip install -r requirements.txt
```
3. Run the notebook:
```
jupyter notebook final_notebook.ipynb
```
## Results Summary

### XGBoost achieved the best performance:

Validation R² ≈ 69

MAE ≈ 0.18

MSE ≈ 0.064

-Linear Regression and Decision Tree performed worse but provide useful baselines.

### XGBoost hyperparameters were tuned to reduce overfitting:
```
max_depth=3, n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5
```

## ⏱️ Timeline

This project took one week for completion.


## 📌 Personal Situation

This project was done as part of the AI Boocamp at BeCode.org. 

Connect with me on [LinkedIn](https://www.linkedin.com/in/hamideh-be/ ).