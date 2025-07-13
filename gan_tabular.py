import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from ctgan import CTGAN

# 1. Load and preprocess data
data = pd.read_csv('data/data.csv', header=None)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Train CTGAN
ctgan = CTGAN(epochs=300)
ctgan.fit(pd.DataFrame(X_scaled, columns=[f'f{i}' for i in range(X.shape[1])]))

# 3. Generate synthetic data
num_samples = 1000
synthetic_data = ctgan.sample(num_samples)

# 4. Visualize real vs synthetic data (PCA)
pca = PCA(n_components=2)
real_pca = pca.fit_transform(X_scaled)
synth_pca = pca.transform(synthetic_data)
plt.figure(figsize=(8, 6))
plt.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.5, label='Real', s=10)
plt.scatter(synth_pca[:, 0], synth_pca[:, 1], alpha=0.5, label='Synthetic', s=10)
plt.legend()
plt.title('PCA: Real vs Synthetic Data')
plt.savefig('visualizations/gan_pca.png')
plt.close()

# 5. Visualize real vs synthetic data (t-SNE)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
real_tsne = tsne.fit_transform(X_scaled)
synth_tsne = tsne.fit_transform(synthetic_data)
plt.figure(figsize=(8, 6))
plt.scatter(real_tsne[:, 0], real_tsne[:, 1], alpha=0.5, label='Real', s=10)
plt.scatter(synth_tsne[:, 0], synth_tsne[:, 1], alpha=0.5, label='Synthetic', s=10)
plt.legend()
plt.title('t-SNE: Real vs Synthetic Data')
plt.savefig('visualizations/gan_tsne.png')
plt.close()

# 6. Feature distribution comparison
for i in range(X.shape[1]):
    plt.figure(figsize=(6, 3))
    sns.kdeplot(X_scaled[:, i], label='Real', fill=True)
    sns.kdeplot(synthetic_data.iloc[:, i], label='Synthetic', fill=True)
    plt.title(f'Feature {i} Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'visualizations/gan_feature_{i}_dist.png')
    plt.close()

# 7. Save synthetic data
synthetic_data.to_csv('data/synthetic_data.csv', index=False)

print('CTGAN training and visualization complete. Synthetic data saved to data/synthetic_data.csv.') 