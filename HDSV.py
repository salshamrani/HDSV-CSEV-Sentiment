import numpy as np
from datasets import load_from_disk
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the dataset
yelp_with_embeddings = load_from_disk("Datasets/yelp_with_embeddings")

def select_distinctive_samples(embeddings, labels, n_samples=100):
    """
    Select n_samples from each class with the highest distance between them.
    """
    positive_indices = np.where(labels == 1)[0]
    negative_indices = np.where(labels == 0)[0]
    
    positive_embeddings = embeddings[positive_indices]
    negative_embeddings = embeddings[negative_indices]
    
    # Calculate pairwise distances
    distances = euclidean_distances(positive_embeddings, negative_embeddings)
    
    # Get indices of top n_samples distances
    flat_indices = np.argsort(distances.ravel())[-n_samples:]
    pos_indices, neg_indices = np.unravel_index(flat_indices, distances.shape)
    
    selected_positive = positive_indices[pos_indices]
    selected_negative = negative_indices[neg_indices]
    selected_distances = distances[pos_indices, neg_indices]
    
    return selected_positive, selected_negative, selected_distances

def visualize_samples(embeddings, labels, selected_pos, selected_neg):
    """
    Visualize the selected samples using PCA.
    """
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                c=labels, alpha=0.1, cmap='coolwarm')
    plt.scatter(reduced_embeddings[selected_pos, 0], reduced_embeddings[selected_pos, 1], 
                color='red', marker='x', s=50, label='Selected Positive')
    plt.scatter(reduced_embeddings[selected_neg, 0], reduced_embeddings[selected_neg, 1], 
                color='blue', marker='x', s=50, label='Selected Negative')
    plt.title('PCA Visualization of High-Distance Samples')
    plt.legend()
    plt.show()

def main():
    train_embeddings = np.array(yelp_with_embeddings['train']['embeddings'])
    train_labels = np.array(yelp_with_embeddings['train']['label'])
    
    selected_pos, selected_neg, distances = select_distinctive_samples(train_embeddings, train_labels)
    
    print(f"Average distance between selected samples: {np.mean(distances):.4f}")
    print(f"Max distance: {np.max(distances):.4f}")
    print(f"Min distance: {np.min(distances):.4f}")
    
    visualize_samples(train_embeddings, train_labels, selected_pos, selected_neg)

if __name__ == "__main__":
    main()
