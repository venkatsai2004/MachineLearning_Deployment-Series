from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import pandas as pd

def load_model(model_path):
    model = joblib.load(model_path)
    return model

def train_logistic_regression(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

def train_random_forest(X, y):
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

def train_xgboost(X, y):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    return model

def evaluate_model(model, X, y):
    predictions = model.predict(X)
    accuracy = (predictions == y).mean()
    return accuracy

def save_model(model, model_path):
    joblib.dump(model, model_path)