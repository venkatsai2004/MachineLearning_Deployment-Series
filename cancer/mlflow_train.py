import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, precision_recall_curve
import pandas as pd

df = pd.read_csv(r"D:\data scinece master\EDA3\differentiated_thyroid_cancer_recurrence.csv")

# Map strings to numbers for all categorical columns AND the target
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
    },
    "Recurred": {'No': 0, 'Yes': 1}
}
for col, mapping in encoding_maps.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

x = df[['Gender','Smoking','Risk','Focality','Response']]
y = df['Recurred']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    rf = RandomForestClassifier(random_state=42).fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)
    
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    plt.savefig("Confusion_Matrix.png")
    mlflow.log_artifact("Confusion_Matrix.png")
    
    y_proba = rf.predict_proba(x_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f'AUC={roc_auc:.2f}')
    ax2.legend()
    plt.savefig("ROC_Curve.png")
    mlflow.log_artifact("ROC_Curve.png")
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    fig3, ax3 = plt.subplots()
    ax3.plot(recall, precision, label=f'AUC={pr_auc:.2f}')
    ax3.legend()
    plt.savefig("Precision_Recall_Curve.png")
    mlflow.log_artifact("Precision_Recall_Curve.png")
    
    mlflow.sklearn.log_model(rf,"model")
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri,"Cancer_Recurrence_Prediction")
    print(f"Model {result.name} is registered with version {result.version}")
