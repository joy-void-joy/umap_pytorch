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


# Load the MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)

# Wrap it with our custom dataset (PUMAP will extract just the data, ignoring labels)
dataset = MNISTDataset(train_dataset)

# For visualization, get the labels
labels = [str(train_dataset[i][1]) for i in range(len(train_dataset))]

# Create and fit PUMAP - now accepts a Dataset directly!
pumap = PUMAP(epochs=5, min_dist=1, n_neighbors=50, num_workers=8, decoder=True, beta=0.01, match_nonparametric_umap=True)
pumap.fit(dataset)
pumap.save('yo.pkl')
pumap = load_pumap('yo.pkl')

# For transform, we still need to pass a tensor
X = torch.stack([dataset[i][0] for i in range(len(dataset))])
embedding = pumap.transform(X)
print(embedding.shape, embedding)
sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=labels, s=0.4)
plt.savefig('test4.png')


def regenerate_and_plot(i=6):
    some_points = embedding[np.random.choice(embedding.shape[0], 6)]
    regenerated = pumap.inverse_transform(torch.Tensor(some_points))

    for i in range(6):
        img = regenerated[i,0]
        img = Image.fromarray(np.uint8(img))
        img.save("image_{}.png".format(i))

regenerate_and_plot()


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


# Example 2: Using a precomputed hnswlib index
# ------------------------------------------------
# If you have an hnswlib index, you can use it to build the graph
# without loading all data into memory. This is ideal for 500GB+ datasets.

import hnswlib

# Assuming you already have your hnswlib index built:
# hnsw_index = hnswlib.Index(space='l2', dim=your_dim)
# hnsw_index.load_index('your_index.bin')

pumap = PUMAP(
    epochs=10,
    n_neighbors=50,
)
pumap.fit(
    your_large_dataset,
    hnsw_index=hnsw_index,
    knn_batch_size=1000,  # Process 1000 samples at a time for KNN queries
)


# Example 3: Using a precomputed graph
# ------------------------------------
# If you've already computed the UMAP graph yourself, you can pass it directly.

from umap_pytorch import get_umap_graph_from_precomputed_knn

# If you have precomputed KNN indices and distances:
# knn_indices: shape (n_samples, n_neighbors) - neighbor indices
# knn_dists: shape (n_samples, n_neighbors) - neighbor distances

graph = get_umap_graph_from_precomputed_knn(
    knn_indices, knn_dists, n_samples=len(dataset), n_neighbors=50
)

pumap = PUMAP(epochs=10)
pumap.fit(dataset, precomputed_graph=graph)


# Example 4: Building hnswlib index from your dataset
# ----------------------------------------------------
# If you need to build the hnswlib index first:

import hnswlib

# Get dimensionality from first sample
first_sample = dataset[0]
if isinstance(first_sample, tuple):
    first_sample = first_sample[0]
dim = first_sample.flatten().shape[0]

# Create index
hnsw_index = hnswlib.Index(space='l2', dim=dim)
hnsw_index.init_index(max_elements=len(dataset), ef_construction=200, M=16)

# Add elements in batches
batch_size = 10000
for i in range(0, len(dataset), batch_size):
    batch_end = min(i + batch_size, len(dataset))
    batch_data = []
    for j in range(i, batch_end):
        item = dataset[j]
        if isinstance(item, tuple):
            item = item[0]
        batch_data.append(item.flatten().numpy())
    hnsw_index.add_items(np.array(batch_data), list(range(i, batch_end)))

# Save for later use
hnsw_index.save_index('my_index.bin')

# Now use with PUMAP
pumap = PUMAP(epochs=10, n_neighbors=50)
pumap.fit(dataset, hnsw_index=hnsw_index)
"""
