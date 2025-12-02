# model_train.py
# OPTIONAL: create a tiny demo ML model using synthetic data.
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import joblib

def make_synthetic_data(n=500):
    np.random.seed(42)
    severity = np.random.choice([3,2,1,0], size=n)  # 3=Critical..0=Low
    age = np.random.exponential(scale=30, size=n)  # days
    exploit = np.random.binomial(1, 0.2, size=n)
    asset = np.random.randint(1,6,size=n)
    # ground truth: high priority if severity high or exploit or asset critical
    label = ((severity>=3) | (exploit==1) | (asset>=5)).astype(int)
    df = pd.DataFrame({
        "severity": severity,
        "age": age,
        "exploit": exploit,
        "asset": asset,
        "label": label
    })
    return df

def train_and_save(path="prioritizer.joblib"):
    df = make_synthetic_data()
    X = df[["severity","age","exploit","asset"]]
    y = df["label"]
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    joblib.dump(clf, path)
    print("Saved model to", path)

if __name__ == "__main__":
    train_and_save()
