import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient

# Load your previous predictions as training data
df = pd.read_csv("predictions_log.csv")

# Encoding maps (ensure consistent encoding with previous)
encoding_maps = {
    "Gender": {'F': 0, 'M': 1},
    "Smoking": {'No': 0, 'Yes': 1},
    "Risk": {'Low': 0, 'Intermediate': 1, 'High': 2},
    "Focality": {'Uni-Focal': 0, 'Multi-Focal': 1},
    "Response": {
        'Excellent': 0,
        'Indeterminate': 1,
        'Biochemical Incomplete': 2,
        'Structural Incomplete': 3
    }
}

# Map categorical columns
for col, mapping in encoding_maps.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

# Prepare features and target from predictions
x = df[['Gender', 'Smoking', 'Risk', 'Focality', 'Response']]  # Features
y = df['prediction']  # Use prediction as target

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    model = RandomForestClassifier(random_state=42)
    model.fit(x_train, y_train)
    
    accuracy = model.score(x_test, y_test)
    mlflow.log_metric("accuracy", float(accuracy))
    
    mlflow.sklearn.log_model(model, "model")
    
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/model"
    register_model = mlflow.register_model(model_uri, "Cancer_Recurrence_Prediction")
    
    print(f"Model {register_model.name} with version {register_model.version} registered")

client = MlflowClient()
# Optional: You can skip or update the stage as per your MLflow version
# client.transition_model_version_stage(
#     name="Cancer_Recurrence_Prediction",
#     version=register_model.version,
#     stage="Production"
# )
