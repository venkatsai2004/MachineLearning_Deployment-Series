def load_data(file_path):
    import pandas as pd
    return pd.read_csv(file_path)

def save_model(model, file_path):
    import joblib
    joblib.dump(model, file_path)

def load_model(file_path):
    import joblib
    return joblib.load(file_path)

def display_dataframe(df):
    import streamlit as st
    st.dataframe(df)

def plot_histogram(df, column):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], bins=30, kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()