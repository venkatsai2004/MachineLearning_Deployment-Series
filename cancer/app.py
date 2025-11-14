import streamlit as st
import pandas as pd
import joblib
import mlflow.pyfunc
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os

model=mlflow.pyfunc.load_model("models:/Cancer_Recurrence_Prediction/1")
def log_inference(input_row,prediction):
    log_entry=input_row.copy()
    log_entry['prediction']=prediction[0]
    log_entry['timestamp']=datetime.datetime.now().isoformat()
    
    file_exists=os.path.isfile("predictions_log.csv")
    pd.DataFrame([log_entry]).to_csv(
        'predictions_log.csv',mode="a",header=not file_exists,index=False
    )
st.image("thyroid img.jpg", width=140)


st.title("Cancer Recurrence Prediction")
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("## Raw Data Preview")
    st.dataframe(df.head(10))
    st.write("## Data Statistics")
    st.write(df.describe())
    st.write("## Visualization")
    for feature in ["Age", "Gender", "Smoking", "Focality", "Thyroid Function", "Response"]:
        if feature in df:
            st.bar_chart(df[feature])

    st.write("## Feature importance")
    feature_importance = {
        'Feature': ['Response', 'Risk', 'N', 'Adenopathy', 'T', 'Stage', 'Focality', 'M', 'Smoking'],
        'Importance': [0.879182, 0.733376, 0.632323, 0.603966, 0.556201, 0.479112, 0.383776, 0.354360, 0.333243]
    }
    df_imp = pd.DataFrame(feature_importance)
    df_imp = df_imp.sort_values('Importance', ascending=True)
    plt.figure(figsize=(8,6))
    sns.barplot(x='Importance', y='Feature', data=df_imp, palette='mako')
    plt.title('Feature Importance for Cancer Recurrence Prediction', fontsize=14)
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

    
# Input widgets
with st.sidebar:
    st.subheader("Patient Input")
    gender = st.selectbox("Gender", ['F', 'M'])
    smoking = st.selectbox("Smoking", ['No', 'Yes'])
    focality = st.selectbox("Focality", ['Uni-Focal', 'Multi-Focal'])
    response = st.selectbox(
        "Response",
        ['Excellent', 'Indeterminate', 'Biochemical Incomplete', 'Structural Incomplete']
    )

# All required columns in the same order as used to train the model
columns_needed = [
    'Age', 'Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy', 'Physical Examination',
    'Adenopathy', 'Pathology', 'Focality', 'Risk', 'T', 'N', 'M', 'Stage', 'Response'
]

# Encoding maps
encoding_maps = {
    "Gender": {'F': 0, 'M': 1},
    "Smoking": {'No': 0, 'Yes': 1},
    "Hx Smoking": {'No': 0, 'Yes': 1},
    "Hx Radiothreapy": {'No': 0, 'Yes': 1},
    "Physical Examination": {
        'Normal': 4,
        'Multinodular goiter': 0,
        'Single nodular goiter-right': 1,
        'Single nodular goiter-left': 2,
        'Diffuse goiter': 3
    },
    "Adenopathy": {
        'No': 0,
        'Right': 1,
        'Left': 2,
        'Bilateral': 3,
        'Extensive': 4,
        'Posterior': 5
    },
    "Pathology": {
        'Papillary': 0,
        'Micropapillary': 1,
        'Follicular': 2,
        'Hurthel cell': 3
    },
    "Focality": {'Uni-Focal': 0, 'Multi-Focal': 1},
    "Risk": {'Low':0, 'Intermediate':1, 'High':2},
    "T": {'T1a':1, 'T1b':2, 'T2':3, 'T3a':4, 'T3b':5, 'T4a':6, 'T4b':7},
    "N": {'N0':0, 'N1a':1, 'N1b':2},
    "M": {'M0':0, 'M1':1},
    "Stage": {'I':1, 'II':2, 'III':3, 'IVA':4, 'IVB':5},
    "Response": {
        'Excellent': 0,
        'Indeterminate': 1,
        'Biochemical Incomplete': 2,
        'Structural Incomplete': 3
    }
}

# Fill in user-chosen and default values
user_inputs = {
    "Age": 50,
    "Gender": gender,
    "Smoking": smoking,
    "Hx Smoking": 'No',
    "Hx Radiothreapy": 'No',
    "Physical Examination": 'Normal',
    "Adenopathy": 'No',
    "Pathology": 'Papillary',
    "Focality": focality,
    "Risk": 'Low',
    "T": 'T1a',
    "N": 'N0',
    "M": 'M0',
    "Stage": 'I',
    "Response": response,
}

# Encode as per maps (leave unchanged if not in encoding_maps)
encoded_data = {
    col: encoding_maps[col].get(user_inputs[col], user_inputs[col])
    if col in encoding_maps else user_inputs[col]
    for col in columns_needed
}

# Build ordered dataframe for model
input_df = pd.DataFrame([encoded_data], columns=columns_needed)

st.write("## Model Input (encoded)")
st.write(input_df)

if st.button("Predict Cancer Recurrence"):
    model = joblib.load("model/XGB_model.joblib")
    prediction = model.predict(input_df)
    result = "Cured" if prediction[0] == 0 else "Not Cured"
    st.success(f"Prediction: {result}")
    log_inference(user_inputs,prediction)
