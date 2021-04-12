import numpy as np
import os


class EEGDataset:
    def __init__(self, dir):
        for f in os.listdir(dir):
            if not f.endswith(".npy"):
                continue

            fname = f[:-4]
            print("Loading {:s}".format(f))
            if fname == "train_data":
                self.train_data = np.load(os.path.join(dir, f))
            elif fname == "train_label":
                self.train_label = np.load(os.path.join(dir, f))
            elif fname == "test_data":
                self.test_data = np.load(os.path.join(dir, f))
            elif fname == "test_label":
                self.test_label = np.load(os.path.join(dir, f))
            
        print("Dataset load done\n")
    
    def describeData(self):
        print("Training data:")
        print("Total: {:d}, Class -1: {:d}, Class 0: {:d}, Class 1: {:d}\n".format(self.train_label.shape[0],
                                                                                 (self.train_label==-1).sum(),
                                                                                 (self.train_label==0).sum(),
                                                                                 (self.train_label==1).sum()))
        print("Test data:")
        print("Total: {:d}, Class -1: {:d}, Class 0: {:d}, Class 1: {:d}\n".format(self.test_label.shape[0],
                                                                                 (self.test_label==-1).sum(),
                                                                                 (self.test_label==0).sum(),
                                                                                 (self.test_label==1).sum()))