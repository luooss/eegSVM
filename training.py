import time
import numpy as np
from dset import EEGDataset
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier


dir = "/Users/luoshuai/Documents/bcmi/SEED/seed"
eeg_dataset = EEGDataset(dir)
eeg_dataset.describeData()
train_data = eeg_dataset.train_data
train_label = eeg_dataset.train_label
test_data = eeg_dataset.test_data
test_label = eeg_dataset.test_label

train_data = StandardScaler().fit_transform(train_data)
test_data = StandardScaler().fit_transform(test_data)

class SVMoneVSoneTrainingApp:
    def __init__(self):
        self.model = BaggingClassifier(SVC(kernel="rbf", gamma=2, C=1, probability=False), max_samples=0.1, n_estimators=10, n_jobs=-1)
        # self.model = BaggingClassifier(SVC(kernel="linear", probability=False, class_weight='balanced'), max_samples=0.1, n_estimators=10)
    
    def main(self):
        start = time.time()
        self.model.fit(train_data, train_label)
        end = time.time()
        print('Time: {:.4f} seconds'.format(end - start))
        test_pred = self.model.predict(test_data)
        pred_accuracy = accuracy_score(test_label, test_pred)
        print(np.unique(test_pred))
        print(pred_accuracy)


class SVMoneVSrestTrainingApp:
    def __init__(self):
        pass


class MinMaxSVMTrainingApp:
    def __init__(self):
        pass