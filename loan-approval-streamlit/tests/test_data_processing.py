import pandas as pd
import numpy as np
import pytest
from src.data_processing import clean_data, preprocess_data

def test_clean_data():
    # Test with a sample DataFrame
    data = {
        'loan_amount': [1000, 2000, np.nan, 4000],
        'income': [5000, 6000, 7000, 8000],
        'loan_approved': [1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    cleaned_df = clean_data(df)
    
    # Check if NaN values are handled
    assert cleaned_df['loan_amount'].isnull().sum() == 0
    assert cleaned_df.shape[0] == 3  # One row should be dropped

def test_preprocess_data():
    # Test with a sample DataFrame
    data = {
        'loan_amount': [1000, 2000, 3000],
        'income': [5000, 6000, 7000],
        'loan_approved': [1, 0, 1]
    }
    df = pd.DataFrame(data)
    processed_df = preprocess_data(df)
    
    # Check if the processed DataFrame has the expected columns
    expected_columns = ['loan_amount', 'income', 'loan_approved', 'dti']
    assert all(col in processed_df.columns for col in expected_columns)