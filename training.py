import time
from types import new_class
import numpy as np
from dset import EEGDataset
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix, plot_confusion_matrix
# from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
import matplotlib.pyplot as plt


dir = "/home/PublicDir/luoshuai/data/SEED"
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
        self.model = OneVsOneClassifier(SVC(kernel="linear", probability=False, class_weight='balanced'), n_jobs=-1)
    
    def main(self):
        print('='*30)
        print('SVM OneVsOne\n')
        start = time.time()
        self.model.fit(train_data, train_label)
        end = time.time()
        print('Training time: {:.4f} seconds\n'.format(end - start))
        train_pred = self.model.predict(train_data)
        test_pred = self.model.predict(test_data)
        train_accuracy = accuracy_score(train_label, train_pred)
        test_accuracy = accuracy_score(test_label, test_pred)
        print('Training accuracy: {:.4f}%'.format(train_accuracy*100))
        print('Test accuracy: {:.4f}%\n'.format(test_accuracy*100))
        # report = classification_report(test_label, test_pred)
        # print(report)
        conf_matrix = plot_confusion_matrix(self.model, test_data, test_label, normalize='true', cmap='Blues')
        conf_matrix.figure_.suptitle("OneVsOne test set confusion matrix")
        plt.savefig('./onevsone_test_confusion_matrix.png')


class SVMoneVSrestTrainingApp:
    def __init__(self):
        self.model = OneVsRestClassifier(SVC(kernel="linear", probability=False, class_weight='balanced'), n_jobs=-1)
    
    def main(self):
        print('='*30)
        print('SVM OneVsRest\n')
        start = time.time()
        self.model.fit(train_data, train_label)
        end = time.time()
        print('Training time: {:.4f} seconds\n'.format(end - start))
        train_pred = self.model.predict(train_data)
        test_pred = self.model.predict(test_data)
        train_accuracy = accuracy_score(train_label, train_pred)
        test_accuracy = accuracy_score(test_label, test_pred)
        print('Training accuracy: {:.4f}%'.format(train_accuracy*100))
        print('Test accuracy: {:.4f}%\n'.format(test_accuracy*100))
        # report = classification_report(test_label, test_pred)
        # print(report)
        conf_matrix = plot_confusion_matrix(self.model, test_data, test_label, normalize='true', cmap='Blues')
        conf_matrix.figure_.suptitle("OneVsRest test set confusion matrix")
        plt.savefig('./onevsrest_test_confusion_matrix.png')


class MinMaxSVMTrainingApp:
    def __init__(self):
        self.model = []
    
    def main(self):
        print('='*30)
        print('Min-Max SVM\n')
        self.divideDataByClass()
        self.train()
        self.predict(test_data, test_label)

    def predict(self, x, y):
        self.pred = []
        for i in range(3):
            self.pred.append([])
            for j in range(3):
                if i == j:
                    continue
                else:
                    p_ = self.model[i+j-1].predict_proba(x)[:, 0]
                    p = p_ if i < j else 1 - p_
                    self.pred[i].append(p.reshape(-1, 1))
        
        self.min_gate = []
        for i in range(3):
            m = np.concatenate(self.pred[i], axis=1).min(axis=1)
            self.min_gate.append(m.reshape(-1, 1))
        
        self.max_gate = np.argmax(np.concatenate(self.min_gate, axis=1), axis=1) - 1
        
        pred_accuracy = accuracy_score(y, self.max_gate)
        print('Prediction accuracy: {:.4f}%'.format(pred_accuracy*100))

        cm = confusion_matrix(y, self.max_gate, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp = disp.plot(cmap='Blues')
        disp.figure_.suptitle("Min-Max SVM test set confusion matrix")
        plt.savefig('./minmax_test_confusion_matrix.png')


    def train(self):
        for i in range(3):
            for j in range(i+1, 3):
                x_train = np.concatenate((self.train_data_by_class[i], self.train_data_by_class[j]), axis=0)
                y_train = np.concatenate((self.train_label_by_class[i], self.train_label_by_class[j]), axis=0)
                print('Training ({:d}, {:d}): '.format(i, j), end='')
                start = time.time()
                self.model.append(SVC(kernel="linear", probability=True, class_weight='balanced').fit(x_train, y_train))
                end = time.time()
                print('cost time {:.4f} seconds'.format(end-start))

    def divideDataByClass(self):
        self.train_data_by_class = []
        self.train_label_by_class = []
        for c in [-1, 0, 1]:
            c_indices = train_label == c
            self.train_data_by_class.append(train_data[c_indices])
            self.train_label_by_class.append(train_label[c_indices])