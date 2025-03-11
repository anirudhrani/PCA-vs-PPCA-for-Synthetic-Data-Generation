import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import scipy.stats as stats
import statsmodels.stats.diagnostic as smd
import pandas as pd

class PCA():
    def __init__(self, x_centered, q):
        self.q= q
        self.x= x_centered
        _, _, _= self.svd()
        _, _, _, _= self.reconstruct_data()

    def reconstruct_data(self):
        self.U_q = self.U[:, :self.q]
        self.S_q = np.diag(self.S[:self.q])
        self.Vt_q = self.Vt[:self.q, :]
        self.X_q = self.U_q @ self.S_q @ self.Vt_q
        return self.X_q, self.U_q, self.S_q, self.Vt_q

    def svd(self,):
        self.U, self.S, self.Vt = np.linalg.svd(self.x, full_matrices=False)
        return self.U, self.S, self.Vt
    
    def generate_synthetic_data(self,):
        zero_array = np.zeros_like(self.U_q)
        n_samples = self.X_q.shape[0]
        mu_gauss, sigma_gauss = stats.norm.fit(self.U_q)

        U_sampled_q = stats.norm.rvs(loc=mu_gauss, scale=sigma_gauss, size= zero_array.shape)
        X_hat_q = U_sampled_q @ self.S_q @ self.Vt_q
        
        mean_vector = np.mean(self.x, axis=0)
        mse = np.mean((self.x - X_hat_q) ** 2)
        sigma = (mse)
        W = np.random.normal(loc=0, scale=sigma, size=X_hat_q.shape)

        x_new_pca = X_hat_q + mean_vector  + W
       
        return x_new_pca
