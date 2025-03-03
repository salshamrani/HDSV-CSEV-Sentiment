import numpy as np
from datasets import load_from_disk
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load the dataset
yelp_with_embeddings = load_from_disk("Datasets/yelp_with_embeddings")

def reduce_dimensions(embeddings, n_components=2):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(embeddings)
    
    tsne = TSNE(n_components=n_components, random_state=42)
    tsne_result = tsne.fit_transform(embeddings)
    
    return pca_result, tsne_result

def plot_embeddings(reduced_embeddings, labels, title):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter)
    plt.title(title)
    plt.show()

def main():
    train_embeddings = np.array(yelp_with_embeddings['train']['embeddings'])
    train_labels = np.array(yelp_with_embeddings['train']['label'])
    
    pca_result, tsne_result = reduce_dimensions(train_embeddings)
    
    plot_embeddings(pca_result, train_labels, 'PCA of Yelp Embeddings')
    plot_embeddings(tsne_result, train_labels, 't-SNE of Yelp Embeddings')

if __name__ == "__main__":
    main()
