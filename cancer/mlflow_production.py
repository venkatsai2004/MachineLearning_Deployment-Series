import mlflow.pyfunc
module=mlflow.pyfunc.load_model("models:/Cancer_Recurrence_Prediction/1")
import pandas as pd
import numpy as np  

model =mlflow.pyfunc.load_model("models:/Cancer_Recurrence_Prediction/1")


features_for_model = ['Gender', 'Smoking', 'Risk', 'Focality', 'Response']
new_data = pd.read_csv("predictions_log.csv")

# Optional: Apply encoding if values are still strings
encoding_maps = {
    "Gender": {'F': 0, 'M': 1},
    "Smoking": {'No': 0, 'Yes': 1},
    "Risk": {'Low': 0, 'Intermediate': 1, 'High': 2},
    "Focality": {'Uni-Focal': 0, 'Multi-Focal': 1},
    "Response": {'Excellent': 0, 'Indeterminate': 1, 
                 'Biochemical Incomplete': 2, 'Structural Incomplete': 3}
}
for col, mapping in encoding_maps.items():
    if col in new_data.columns:
        new_data[col] = new_data[col].map(mapping)

# Select only the features for prediction
X_new = new_data[features_for_model]
result = model.predict(X_new)
print("Predictions Cancer Recurrence:", result)



# form flask type predictions
# from flask import Flask, request, jsonify
# import mlflow.pyfunc
# import pandas as pd

# app = Flask(__name__)
# model = mlflow.pyfunc.load_model("models:/LoanApprovalModel/Production")

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get input data from the request
#     input_data = request.json  # e.g., {"city_encoded": 1, "loan_amount": 20000, ...}

#     # Convert to DataFrame with the correct columns
#     new_data = pd.DataFrame([input_data])

#     # Make prediction
#     result = model.predict(new_data)

#     # Return the result
#     return jsonify({"prediction": result.tolist()})

# if __name__ == '__main__':
#     app.run()
