import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.title("Loan Approval Prediction")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("## Raw Data Preview")
    st.dataframe(df.head(10))
    
    # Training on just these features
    features = ["city", "loan_amount", "credit_score", "years_employed"]
    # Encode city as index for simplicity (for production use LabelEncoder)
    df["city_encoded"] = df["city"].astype("category").cat.codes
    X = df[["city_encoded", "loan_amount", "credit_score", "years_employed"]]
    y = df["loan_approved"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    st.write("## Predict Loan Approval")
    # Manual entry form
    city_entry = st.text_input("City")
    loan_amount_entry = st.number_input("Loan Amount", min_value=0, value=10000)
    credit_score_entry = st.number_input("Credit Score", min_value=0, max_value=1000, value=500)
    years_employed_entry = st.number_input("Years Employed", min_value=0, value=1)

    # Encode entered city same as training encoding
    known_cities = list(df["city"].astype("category").cat.categories)
    if city_entry in known_cities:
        city_code = known_cities.index(city_entry)
    else:
        city_code = 0  # Default or show warning

    new_data = pd.DataFrame([{
        "city_encoded": city_code,
        "loan_amount": loan_amount_entry,
        "credit_score": credit_score_entry,
        "years_employed": years_employed_entry
    }])
    
    if st.button("Predict Loan Approval"):
        pred = model.predict(new_data)[0]
        st.write(f"Prediction: {'Approved' if pred else 'Not Approved'}")
