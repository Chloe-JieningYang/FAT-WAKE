import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut


'''
feature_data   : numpy with shape (data num,feature num)
KSS_annotation : numpy with shape (data num,1)
kernel         : 'linear'/'rbf'
'''
def Classify_by_SVM(feature_data, KSS_annotation, kernel):
    X = feature_data.copy()
    y = KSS_annotation.copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    acc = []
    C = 1.0  # parameter
    gamma = 'scale'
    loo = RepeatedKFold(n_splits=10, n_repeats=2)
    for i, (train_index, test_index) in enumerate(loo.split(X)):
        train_X, train_y = X_scaled[train_index], y[train_index]
        test_X, test_y = X_scaled[test_index], y[test_index]
        svm = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=C, gamma=gamma))
        svm.fit(train_X, train_y)
        predict_y = svm.predict(test_X)
        acc.append(accuracy_score(test_y, predict_y))
    acc = np.array(acc)
    print(f"average acc of SVM with {kernel}:{np.mean(acc)},standard of acc:{np.std(acc)}")
