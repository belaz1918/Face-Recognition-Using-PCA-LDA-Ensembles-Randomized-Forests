import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

# Incremental PCA
def perform_incremental_pca(subsets, n_components):
    ipca = IncrementalPCA(n_components=n_components)
    for subset in subsets:
        ipca.partial_fit(subset.T)
    return ipca

# Batch PCA
def perform_batch_pca(images, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(images.T)
    return pca

# Evaluate PCA methods
def evaluate_pca(pca_model, images, labels, test_images, test_labels):
    train_projected = pca_model.transform(images.T)
    test_projected = pca_model.transform(test_images.T)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(train_projected, labels)
    predictions = knn.predict(test_projected)
    accuracy = accuracy_score(test_labels, predictions)
    reconstruction_error = mean_squared_error(images.T, pca_model.inverse_transform(train_projected))
    return accuracy, reconstruction_error
