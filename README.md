# Synthetic Image Generation using PCA & PPCA

This project explores techniques in dimensionality reduction and probabilistic modeling to analyze and generate handwritten digit images using the USPS dataset. The core objective is to build a structured understanding of principal component analysis (PCA) and its probabilistic counterpart (PPCA), while validating the effectiveness of synthetic data generation through statistical measures and visual evaluation.

## Overview

The project investigates how low-dimensional representations can efficiently capture essential features in high-dimensional image data. By leveraging singular value decomposition (SVD) and probabilistic sampling, it not only reduces dimensional complexity but also generates synthetic digit images that closely resemble real data.

## Key Highlights

- 📉 **Dimensionality Reduction with PCA**  
  Data is first centered and decomposed using SVD to extract dominant components. The impact of dimensionality (`q`) is visualized through singular values, explained variance, and reconstruction error (MSE).

- 📊 **Visual and Statistical Analysis**  
  Distributions of principal components are modeled using Gaussian and Laplace fits. Distribution fitting quality is evaluated using Kolmogorov-Smirnov and Anderson-Darling tests, along with parameter extraction for interpretability.

- 🧠 **Synthetic Image Generation via PCA**  
  A generative model is implemented by sampling from fitted distributions of PCA projections, incorporating noise and mean reconstruction. Synthetic samples are visually compared against true samples and analyzed through correlation matrix heatmaps and L2 similarity metrics.

- 🔍 **Validation via Cross-Correlation**  
  Feature-level correlation structures of real and synthetic datasets are compared using empirical covariance and correlation matrices. Heatmaps and L2 differences provide a quantitative measure of how well synthetic data preserves original structure.

- 🌀 **Probabilistic PCA (PPCA)**  
  PPCA is implemented to introduce stochasticity in the generation process by sampling latent variables and adding Gaussian noise. Synthetic digits are generated and evaluated similarly, enabling a direct comparison between deterministic PCA and probabilistic PPCA models.

- ⚖️ **Comparative Evaluation of PCA vs PPCA**  
  A final analysis compares the generative quality, robustness, and practical utility of PCA vs PPCA in terms of structural preservation and randomness, offering insights into model selection for synthetic data tasks.

## Getting Started

Install requirements:
```bash
pip install -r requirements.txt  
```
Run:
```bash 
python main.py --path usps.h5 --q 5 --digit 3
```