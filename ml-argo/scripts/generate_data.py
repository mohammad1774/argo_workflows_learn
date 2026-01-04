import pandas as pd 
from sklearn.datasets import make_classification

X, y  = make_classification(
    n_samples=100,
    n_features=4,
    n_classes=2,
    random_state=42
)

df = pd.DataFrame(X, columns=['f1','f2','f3','f4'])
df['label'] = y
df.to_csv('dataset.csv',index=False)

print("dataset.csv created")