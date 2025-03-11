import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import scipy.stats as stats
import statsmodels.stats.diagnostic as smd
import pandas as pd

digit=3

class DataPrep():
    def __init__(self, path):
        self.path= path

    def get_data(self, digit=3):
        with h5py.File(self.path, 'r') as hf:
            train_images = np.array(hf["train"]["data"])  # Image data
            train_labels = np.array(hf["train"]["target"])  # Labels

        
        digit_data = train_images[train_labels == digit]

        # Flatten images (16x16) into vectors (256-dimensional)
        x = digit_data.reshape(digit_data.shape[0], -1)
        return x

    def center_data(self, x):
        # Normalize the data (mean centering)
        scaler = StandardScaler(with_std=False)
        x_centered = scaler.fit_transform(x)
        return x_centered

# if __name__=="__main__":
#     parser= argparse.ArgumentParser()
#     parser.add_argument("path", type=str, help="enter path")
#     args = parser.parse_args()
#     file_path =  args.path # "/usps.h5"
#     prep_obj= DataPrep(file_path)
#     x= prep_obj.get_data(digit= digit)
#     x_centered= prep_obj.center_data(x)
#     print(x_centered.shape)