import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap


def plot_target_distribution(df, target_col='target'):
    """Plot the distribution of the target variable."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_col, data=df)
    plt.title('Target Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.grid(axis='y')
    return plt.gcf()


def plot_numeric_distributions(df, numeric_columns):
    """Plot histograms for numeric features."""
    n_cols = 3
    n_rows = int(np.ceil(len(numeric_columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()
    for i, col in enumerate(numeric_columns):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i])
        axes[i].set_title(col)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    return fig


def compute_pca(X_scaled, feature_names, n_components=2, top_n=40):
    """Run PCA and return projected data and top contributing features."""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    components = pca.components_
    importances = np.sum(components[:n_components, :] ** 2, axis=0)
    top_idx = np.argsort(importances)[-top_n:]
    top_components = components[:, top_idx]
    top_features = feature_names[top_idx]
    return X_pca, top_components, top_features


def plot_pca_biplot(X_pca, components, feature_names):
    """Plot PCA biplot with arrows for top features."""
    plt.figure(figsize=(8, 8))
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    for i, varname in enumerate(feature_names):
        plt.arrow(0, 0, components[0, i], components[1, i], alpha=0.5)
        plt.text(components[0, i] * 1.1, components[1, i] * 1.1, varname,
                 fontsize=8, ha='center', va='center')
    plt.title('PCA Biplot')
    return plt.gcf()


def compute_tsne(X_scaled, perplexity=30, random_state=42):
    """Compute t-SNE projection."""
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    X_tsne = tsne.fit_transform(X_scaled)
    return X_tsne


def compute_umap(X_scaled, n_neighbors=15, min_dist=0.1, random_state=42):
    """Compute UMAP projection."""
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    X_umap = reducer.fit_transform(X_scaled)
    return X_umap


def plot_2d_projection(X_proj, labels, title='2D Projection'):
    """Plot a 2D projection (e.g., t-SNE or UMAP) with color-coded labels."""
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_proj[:, 0], X_proj[:, 1], c=labels, cmap='coolwarm', alpha=0.5)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    return plt.gcf()
