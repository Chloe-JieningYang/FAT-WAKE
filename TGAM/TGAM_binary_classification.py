import numpy as np
import classify_SVM as cSVM
import classify_KNN as cKNN
import classify_RF as cRF
import preprocess as pre
import file_list as flist

rest_start = 70  # the start time of useful data(5s)
rest_end = 94
sample_rate = 512
spE_num = 4
sp_num = 15
reget_features = 0  # choose whether to re-get the features. If not, the features will be directly read from X.txt.
X_path = "X.txt"

wake_kss, fat_kss = pre.get_KSS("../no_kss.xlsx", rest_start, rest_end)
kss = np.append(wake_kss, fat_kss, axis=0)

if reget_features:
    spEw, spw = pre.get_features(flist.wake_file_list, rest_start, rest_end, sample_rate)
    spEf, spf = pre.get_features(flist.fat_file_list, rest_start, rest_end, sample_rate)
    spE = np.append(spEw, spEf, axis=0).reshape(-1, spE_num)
    sp = np.append(spw, spf, axis=0).reshape(-1, sp_num)

    print(f"shape of alpha:{sp.shape}")
    X = np.concatenate((spE, sp), axis=1)

    print(f"shape of X:{X.shape}")
    # print(f"shape of X:{X.shape}")
    # print(f"shape of kss:{kss.shape}")

    # save X
    np.savetxt(X_path, X)

else:
    # just read X from txt
    X = np.loadtxt(X_path)

cSVM.Classify_by_SVM(X, kss, 'linear')
cSVM.Classify_by_SVM(X, kss, 'rbf')
cKNN.Classify_by_KNN(X, kss)
cRF.Classify_by_RF(X, kss)
