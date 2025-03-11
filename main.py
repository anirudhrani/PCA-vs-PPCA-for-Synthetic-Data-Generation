import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import scipy.stats as stats
import statsmodels.stats.diagnostic as smd
import pandas as pd

from src import utils
from src.data_prep import DataPrep
from src.pca import PCA
from src.ppca import PPCA



if __name__=="__main__":
    parser= argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="enter path")
    parser.add_argument("--q", type=int, help="number of principal components")
    parser.add_argument("--digit", type=int, help="number between 0 and 9")
    args = parser.parse_args()

    file_path = args.path
    q= args.q
    digit= args.digit

    prep_obj= DataPrep(file_path)
    x= prep_obj.get_data(digit= digit)
    x_centered= prep_obj.center_data(x)
    

    pca= PCA(x_centered, q=q)
    x_new_pca= pca.generate_synthetic_data()
    print(x_new_pca.shape)

    utils.sample_random_and_plot(x_new_pca)

    # Mean-centered synthetic data matrix
    X_hat_mean = np.mean(x_new_pca, axis=0)
    X_hat_centered = x_new_pca - X_hat_mean

    # Empirical covariance matrix of synthetic data
    cov_X_hat = (X_hat_centered.T @ X_hat_centered) / (x_new_pca.shape[0] - 1)

    print(f"Shape of Covariance Matrix for Synthetic Data using PCA: {cov_X_hat.shape}")
    # print("Sample Covariance Matrix of Synthetic Data:\n", cov_X_hat)

    pca_L2= utils.l2_real_synthetic(x_centered= x_centered, x_synth= x_new_pca)
    print(f"L2 PCA: {pca_L2}")

    ppca= PPCA(x_centered, q=q)
    x_new_ppca= ppca.generate_synthetic_data()
    print(x_new_ppca.shape)

    utils.sample_random_and_plot(x_new_ppca)

    # Mean-centered synthetic data matrix
    X_hat_mean = np.mean(x_new_ppca, axis=0)
    X_hat_centered = x_new_ppca - X_hat_mean

    # Empirical covariance matrix of synthetic data
    cov_X_hat = (X_hat_centered.T @ X_hat_centered) / (x_new_ppca.shape[0] - 1)

    print(f"Shape of Covariance Matrix for Synthetic Data using PCA: {cov_X_hat.shape}")
    # print("Sample Covariance Matrix of Synthetic Data:\n", cov_X_hat)

    ppca_L2= utils.l2_real_synthetic(x_centered= x_centered, x_synth= x_new_ppca)
    print(f"L2 PPCA: {ppca_L2}")