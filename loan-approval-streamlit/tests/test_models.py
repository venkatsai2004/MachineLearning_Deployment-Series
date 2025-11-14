import pytest
import pandas as pd
from src.models import train_model, predict_model
from sklearn.metrics import accuracy_score
import joblib

# Load the model for testing
model = joblib.load('models/loan_model.joblib')

def test_train_model():
    # Sample data for training
    data = pd.DataFrame({
        'loan_amount': [1000, 2000, 3000],
        'income': [3000, 4000, 5000],
        'loan_approved': [0, 1, 0]
    })
    
    # Train the model
    trained_model = train_model(data)
    
    # Check if the model is not None
    assert trained_model is not None

def test_predict_model():
    # Sample input for prediction
    sample_input = pd.DataFrame({
        'loan_amount': [1500],
        'income': [3500]
    })
    
    # Make a prediction
    prediction = predict_model(model, sample_input)
    
    # Check if the prediction is in the expected range (0 or 1)
    assert prediction in [0, 1]

def test_model_accuracy():
    # Sample test data
    test_data = pd.DataFrame({
        'loan_amount': [1000, 2000, 3000],
        'income': [3000, 4000, 5000],
        'loan_approved': [0, 1, 0]
    })
    
    # Separate features and target
    X_test = test_data[['loan_amount', 'income']]
    y_test = test_data['loan_approved']
    
    # Make predictions
    predictions = predict_model(model, X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    
    # Check if accuracy is above a certain threshold
    assert accuracy >= 0.5  # Adjust threshold as necessary for your model