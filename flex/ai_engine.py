import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

MODEL_PATH = os.path.join("models", "priority_model.joblib")

def featurize_patch_meta(meta):
    """
    Convert patch meta dict into numeric features for the model.
    Features:
     - severity (map): low=0,medium=1,high=2,critical=3
     - priority (if provided)
     - change_files_count (if provided)
     - age_days (if provided)
    """
    severity_map = {"low":0, "medium":1, "high":2, "critical":3}
    sev = severity_map.get(meta.get("severity","medium").lower(), 1)
    priority = float(meta.get("priority", 1))
    change_files = int(meta.get("change_files_count", 1))
    age_days = float(meta.get("age_days", 30))
    return np.array([sev, priority, change_files, age_days]).reshape(1, -1)

def load_or_train_model():
    os.makedirs("models", exist_ok=True)
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    # Train a synthetic model (example) - you should replace with real data
    X = []
    y = []
    import random
    for _ in range(300):
        sev = random.choice([0,1,2,3])
        priority = random.uniform(0.5, 5.0)
        change_files = random.randint(1, 200)
        age = random.randint(1, 365)
        score = sev * 2.5 + priority * 0.5 + np.log1p(change_files) * 0.2 + (365 - age) * 0.002
        X.append([sev, priority, change_files, age])
        y.append(score + random.uniform(-0.5,0.5))
    X = np.array(X)
    y = np.array(y)
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model

def score_patch(model, patch):
    meta = patch.get("meta", patch)
    X = featurize_patch_meta(meta)
    return model.predict(X)[0]
