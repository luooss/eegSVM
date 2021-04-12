import numpy as np
import os


class EEGDataset:
    def __init__(self, dir):
        for f in os.listdir(dir):
            if not f.endswith(".npy"):
                continue

            fname = f[:-4]
            print("Processing {:s}".format(f))
            if fname == "train_data":
                self.train_data = np.load(os.path.join(dir, f))
            elif fname == "train_label":
                self.train_label = np.load(os.path.join(dir, f))
            elif fname == "test_data":
                self.test_data = np.load(os.path.join(dir, f))
            elif fname == "test_label":
                self.test_label = np.load(os.path.join(dir, f))
            
        print("Dataset load done")
    
    def describeData(self):
        print("Training Data size: {:d}".format(self.train_data.shape[0]))
        print("Test Data size: {:d}".format(self.test_data.shape[0]))