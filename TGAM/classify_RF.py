import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import pickle

def Classify_by_RF(feature_data, KSS_annotation):
    X = feature_data.copy()
    y = KSS_annotation.copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rf_dep_range = range(7, 13) # choose depth from 7 to 12
    rf_err = []
    for depth in rf_dep_range:
        rf_model = RandomForestClassifier(n_estimators=10, max_depth=depth, random_state=42)
        scores = cross_val_score(rf_model, X_scaled, y, cv=10, scoring='accuracy')
        rf_err.append(1 - scores.mean())
    dep_min = rf_err.index(min(rf_err)) + 7
    print(f"dep_min:{dep_min}")
    RF = RandomForestClassifier(n_estimators=10, max_depth=dep_min, random_state=42)
    scores = cross_val_score(RF, X_scaled, y, cv=10, scoring='accuracy')
    RF.fit(X_scaled, y)
    # save the model as pkl file
    pickle.dump(RF, open('model_RF.pkl', 'wb'))
    print(f"RF accuracy:{scores.mean()},accuracy standard:{scores.std()}")
