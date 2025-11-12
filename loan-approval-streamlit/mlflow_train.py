import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,roc_curve,auc,confusion_matrix,precision_recall_curve
import numpy as np
import pandas as pd

df =pd.read_csv("D:\data scinece master\EDA4\loan_approval.csv")
df["city_encoded"]=df["city"].astype("category").cat.codes
x=df[['city_encoded','loan_amount','credit_score','years_employed']]
y=df['loan_approved']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


with mlflow.start_run():
    model=RandomForestClassifier(random_state=42).fit(x_train,y_train)
    y_pred=model.predict(x_test)
    accuracy=accuracy_score(y_test,y_pred)
    mlflow.log_metric("accuracy",accuracy)
    
    y_pred=model.predict(x_test)
    cm=confusion_matrix(y_test,y_pred)
    fig,ax=plt.subplots()
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    
    y_proba=model.predict_proba(x_test)[:,1]
    fpr,tpr,thresholds=roc_curve(y_test,y_proba)
    roc_curve=auc(fpr,tpr)
    fig2,ax2=plt.subplots()
    ax2.plot(fpr,tpr,label=f'AUC = {roc_curve:.2f}')
    ax2.legend()
    plt.savefig("roc_curve.png")
    mlflow.log_artifact("roc_curve.png")
    
    precision,recall,thresholds=precision_recall_curve(y_test,y_proba)
    precision_recall_curve=auc(recall,precision)
    fig3,ax3=plt.subplots()
    ax3.plot(recall,precision,label=f'AUC = {precision_recall_curve:.2f}')
    ax3.legend()
    plt.savefig("precision_recall_curve.png")
    mlflow.log_artifact("precision_recall_curve.png")
    
    mlflow.sklearn.log_model(model,"model")
    run_id=mlflow.active_run().info.run_id
    model_uri=f"runs:/{run_id}/model"
    result=mlflow.register_model(model_uri,"LoanApprovalModel")
    print(f"Model Registered with ID at {result.version}")
    
from mlflow.tracking import MlflowClient
client = MlflowClient()
client.transition_model_version_stage(
    name="LoanApprovalModel",
    version=4, 
    stage="Production"
)
