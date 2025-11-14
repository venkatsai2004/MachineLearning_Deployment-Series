import pandas as pd
import numpy as np

def load_data(file_path):
    """Load the loan approval dataset."""
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    """Clean the dataset by handling missing values and dropping unnecessary columns."""
    df.drop(['name'], axis=1, inplace=True)
    df.dropna(inplace=True)
    return df

def preprocess_data(df):
    """Preprocess the data by encoding categorical variables and scaling numerical features."""
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    # Encode categorical variables
    df['loan_approved'] = le.fit_transform(df['loan_approved'])
    df['income_bracket'] = le.fit_transform(df['income_bracket'])
    df['loan_bracket'] = le.fit_transform(df['loan_bracket'])
    
    # Create new features
    df['dti'] = df['loan_amount'] / df['income']
    
    return df

def get_feature_target(df, target_column):
    """Separate features and target variable."""
    X = df.drop([target_column], axis=1)
    y = df[target_column]
    return X, y