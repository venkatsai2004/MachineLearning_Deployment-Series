import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_matrix(df):
    plt.figure(figsize=(12, 8))
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()

def plot_loan_approval_distribution(df):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='loan_approved', data=df)
    plt.title('Loan Approval Distribution')
    plt.xlabel('Loan Approved (0 = Yes, 1 = No)')
    plt.ylabel('Count')
    plt.show()

def plot_income_bracket_distribution(df):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='income_bracket', data=df)
    plt.title('Income Bracket Distribution')
    plt.xlabel('Income Bracket')
    plt.ylabel('Count')
    plt.show()

def plot_loan_bracket_distribution(df):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='loan_bracket', data=df)
    plt.title('Loan Bracket Distribution')
    plt.xlabel('Loan Bracket')
    plt.ylabel('Count')
    plt.show()