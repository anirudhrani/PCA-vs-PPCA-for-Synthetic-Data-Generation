import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import scipy.stats as stats
import statsmodels.stats.diagnostic as smd
import pandas as pd



class PPCA():
    def __init__(self, x_centered, q):
        self.q=q
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
    
    def generate_synthetic_data(self):
        n_samples = self.X_q.shape[0]
        z_new = np.random.randn(n_samples, self.q)

        x_mean = np.mean(self.X_q, axis=0)

        d = self.X_q.shape[1]
        sigma_sq = (1 / (d - self.q)) * np.sum((self.S[self.q:]**2))
        sigma = np.sqrt(sigma_sq)

        epsilon = np.random.normal(loc=0, scale=sigma, size=(n_samples, d))

        x_new = z_new @ self.Vt_q + x_mean + epsilon
        return x_new
