import json
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd

# read artifacts placed by Argo (match your WorkflowTemplate paths)
with open("/data/params.json") as f:
    params = json.load(f)

lr = float(params["lr"])          # ✅ convert
acc = float(params.get("acc", 0)) # optional

df = pd.read_csv("/data/dataset.csv")
X = df.drop("label", axis=1)
y = df["label"]

model = LogisticRegression(C=1.0/lr, max_iter=200)
model.fit(X, y)

joblib.dump(model, "/tmp/model.pkl")
print("trained final model with lr=", lr, "selected acc=", acc)
