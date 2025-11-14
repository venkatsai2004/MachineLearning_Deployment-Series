import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient

# If prediction_log.csv has NO header row:
# df = pd.read_csv("prediction_log.csv", names=["city_encoded", "loan_amount", "credit_score", "years_employed", "prediction", "timestamp"])
# If it DOES have headers:
df = pd.read_csv("prediction_log.csv")

X = df[["city_encoded", "loan_amount", "credit_score", "years_employed"]]
y = df["prediction"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    mlflow.sklearn.log_model(model, "model")
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/model"
    registered_model = mlflow.register_model(model_uri, "LoanApprovalModel")
    print(f"Model registered: {registered_model.name}, version {registered_model.version}")

client = MlflowClient()
client.transition_model_version_stage(
    name=registered_model.name,
    version=registered_model.version,
    stage="Production"
)
