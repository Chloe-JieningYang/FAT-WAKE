import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import pickle

'''
feature_data   : numpy with shape (data num,feature num)
KSS_annotation : numpy with shape (data num,1)
'''


def Classify_by_KNN(feature_data, KSS_annotation):
    X = feature_data.copy()
    y = KSS_annotation.copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    k_range = range(1, 100)
    k_error = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_scaled, y, cv=10, scoring='accuracy')
        k_error.append(1 - scores.mean())
    k_min = k_error.index(min(k_error)) + 1
    knn = KNeighborsClassifier(n_neighbors=k_min)
    knn.fit(X_scaled,y)
    # save the model as pkl file
    pickle.dump(knn, open('model_KNN.pkl', 'wb'))
    scores = cross_val_score(knn, X_scaled, y, cv=10, scoring='accuracy')
    print("KNN acc:", scores.mean(), "acc_std", scores.std())
    return scores
