import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import scipy.stats as stats
import statsmodels.stats.diagnostic as smd
import pandas as pd
digit=3

def svd(x):
        U, S, Vt = np.linalg.svd(x, full_matrices=False)
        return U, S, Vt

def l2_real_synthetic(x_centered, x_synth):
    C_real = np.corrcoef(x_centered, rowvar=True)  
    C_synthetic = np.corrcoef(x_synth, rowvar=True)
    L2_diff = np.mean((C_real - C_synthetic) ** 2)
    return L2_diff

def sample_random_and_plot(x_new_pca):
    random_indices = np.random.choice(x_new_pca.shape[0], size=5, replace=False)

    fig, axes = plt.subplots(1, 5, figsize=(10, 2))

    for i, idx in enumerate(random_indices):
        img_2d = x_new_pca[idx].reshape(16, 16)
        
        # Display the image
        axes[i].imshow(img_2d, cmap='gray')
        axes[i].axis('off') 

    plt.tight_layout()
    plt.show()