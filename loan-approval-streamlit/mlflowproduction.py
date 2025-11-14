import mlflow.pyfunc
model = mlflow.pyfunc.load_model("models:/LoanApprovalModel/Production")
# Use model.predict(new_data) for inference

import mlflow.pyfunc
import pandas as pd

model = mlflow.pyfunc.load_model("models:/LoanApprovalModel/Production")

# Example test input (adjust as needed)
new_data = pd.DataFrame([{
    "city_encoded": 1,            # replace with valid values for your data
    "loan_amount": 20000,
    "credit_score": 750,
    "years_employed": 4
}])

result = model.predict(new_data)
print("Predicted loan approval:", result)
