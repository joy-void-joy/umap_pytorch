import torchvision
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from umap_pytorch import PUMAP, load_pumap
import seaborn as sns
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


# Custom dataset wrapper that preprocesses MNIST data
class MNISTDataset(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, index):
        image, label = self.mnist_dataset[index]
        # Original shape: (1, 28, 28) -> keep as (1, 28, 28)
        return image, label


def main():
    # Load the MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, transform=transforms.ToTensor(), download=True
    )

    # Wrap it with our custom dataset (PUMAP will extract just the data, ignoring labels)
    dataset = MNISTDataset(train_dataset)

    # For visualization, get the labels
    labels = [str(train_dataset[i][1]) for i in tqdm(range(len(train_dataset)), desc="Loading labels")]

    # Create and fit PUMAP - now accepts a Dataset directly!
    pumap = PUMAP(
        epochs=5,
        min_dist=1,
        n_neighbors=50,
        num_workers=8,
        decoder=True,
        beta=0.01,
        match_nonparametric_umap=True,
    )
    pumap.fit(dataset)
    pumap.save("yo.pkl")
    pumap = load_pumap("yo.pkl")

    # For transform, we still need to pass a tensor
    X = torch.stack([dataset[i][0] for i in tqdm(range(len(dataset)), desc="Stacking tensors")])
    embedding = pumap.transform(X)
    print(embedding.shape, embedding)
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels, s=0.4)
    plt.savefig("test4.png")

    def regenerate_and_plot(i=6):
        some_points = embedding[np.random.choice(embedding.shape[0], 6)]
        regenerated = pumap.inverse_transform(torch.Tensor(some_points))

        for i in range(6):
            img = regenerated[i, 0]
            img = Image.fromarray(np.uint8(img))
            img.save("image_{}.png".format(i))

    regenerate_and_plot()


if __name__ == "__main__":
    main()


# ==============================================================================
# EXAMPLES FOR LARGE DATASETS (500GB+)
# ==============================================================================

"""
# Example 1: Using graph_sample_size for memory-efficient training
# ----------------------------------------------------------------
# For very large datasets, use graph_sample_size to only load a subset
# of the data for graph construction. The encoder will generalize to all data.

pumap = PUMAP(
    epochs=10,
    n_neighbors=50,
    graph_sample_size=50000,  # Only use 50k samples for graph
    random_state=42,
)
pumap.fit(your_large_dataset)


# Example 2: Using a precomputed graph with faiss
# ------------------------------------------------
from umap_pytorch import get_umap_graph_from_precomputed_knn, get_knn_from_faiss
import faiss

# Assuming you have a faiss index and your data
# faiss_index = faiss.read_index('your_index.bin')
# data = np.array(...)  # Your data as numpy array

n_neighbors = 50
knn_indices, knn_dists = get_knn_from_faiss(faiss_index, data, n_neighbors)
graph = get_umap_graph_from_precomputed_knn(
    knn_indices, knn_dists, n_samples=len(data), n_neighbors=n_neighbors
)

pumap = PUMAP(epochs=10, n_neighbors=n_neighbors)
pumap.fit(dataset, precomputed_graph=graph)


# Example 3: Using a precomputed graph with hnswlib
# --------------------------------------------------
from umap_pytorch import get_umap_graph_from_precomputed_knn, get_knn_from_hnswlib
import hnswlib

# Assuming you have an hnswlib index and your data
# hnsw_index = hnswlib.Index(space='l2', dim=dim)
# hnsw_index.load_index('your_index.bin')
# data = np.array(...)  # Your data as numpy array

n_neighbors = 50
knn_indices, knn_dists = get_knn_from_hnswlib(hnsw_index, data, n_neighbors)
graph = get_umap_graph_from_precomputed_knn(
    knn_indices, knn_dists, n_samples=len(data), n_neighbors=n_neighbors
)

pumap = PUMAP(epochs=10, n_neighbors=n_neighbors)
pumap.fit(dataset, precomputed_graph=graph)


# Example 4: Using precomputed KNN directly
# ------------------------------------------
# If you already have KNN indices and distances from any source:

from umap_pytorch import get_umap_graph_from_precomputed_knn

# knn_indices: shape (n_samples, n_neighbors) - neighbor indices
# knn_dists: shape (n_samples, n_neighbors) - neighbor distances

graph = get_umap_graph_from_precomputed_knn(
    knn_indices, knn_dists, n_samples=len(dataset), n_neighbors=50
)

pumap = PUMAP(epochs=10)
pumap.fit(dataset, precomputed_graph=graph)
"""
