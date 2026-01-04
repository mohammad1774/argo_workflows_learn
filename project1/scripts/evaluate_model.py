import joblib 
import json 
import pandas as pd 
from sklearn.metrics import accuracy_score

model = joblib.load("/tmp/model.pkl")
df = pd.read_csv("/data/dataset.csv")

X = df.drop("label",axis=1)
y = df["label"]

preds = model.predict(X)
acc = accuracy_score(y,preds)

metrics = {
    "accuracy": acc
}

with open("/tmp/metrics.json", "w") as f:
    json.dump(metrics,f)

with open("/tmp/accuracy.txt", "w") as f:
    f.write(str(acc))

print("Accuracy:", acc)