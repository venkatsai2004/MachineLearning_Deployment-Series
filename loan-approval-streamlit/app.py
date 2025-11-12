import streamlit as st
import pandas as pd
import mlflow.pyfunc
import datetime
import os

# Load MLflow production model only ONCE
model = mlflow.pyfunc.load_model("models:/LoanApprovalModel/4")

def log_inference(input_row, pred):
    import pandas as pd
    import datetime

    log_entry = input_row.copy()
    log_entry["prediction"] = pred[0]
    log_entry["timestamp"] = datetime.datetime.now().isoformat()

    file_exists = os.path.isfile("prediction_log.csv")
    pd.DataFrame([log_entry]).to_csv(
        "prediction_log.csv", mode="a", header=not file_exists, index=False
    )

st.title("Loan Approval Prediction")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("## Raw Data Preview")
    st.dataframe(df.head(10))
    
    # Feature encoding
    df["city_encoded"] = df["city"].astype("category").cat.codes

    st.write("## Predict Loan Approval")
    city_entry = st.text_input("City")
    loan_amount_entry = st.number_input("Loan Amount", min_value=0, value=10000)
    credit_score_entry = st.number_input("Credit Score", min_value=0, max_value=1000, value=500)
    years_employed_entry = st.number_input("Years Employed", min_value=0, value=1)

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
        pred = model.predict(new_data)
        st.write(f"Prediction: {'Approved' if pred[0] else 'Not Approved'}")
        log_inference(new_data.iloc[0], pred)
