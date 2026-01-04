import pandas as pd 
import joblib 
import sys 
from sklearn.linear_model import LogisticRegression

lr = float(sys.argv[1])

df = pd.read_csv('/data/dataset.csv')
X = df.drop('label',axis=1)
y = df['label']

model = LogisticRegression(C=1/lr, max_iter=200)
model.fit(X,y)


joblib.dump(model, '/tmp/model.pkl')
print("model.pkl saved with lr=",lr)