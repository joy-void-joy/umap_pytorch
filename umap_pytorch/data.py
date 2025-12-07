import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class SubsetDataset(Dataset):
    """
    A dataset that wraps another dataset and only exposes a subset of indices.
    Useful for training on a sample of a larger dataset.
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def get_item_from_dataset(dataset, index):
    """Extract data item from a dataset, handling tuple returns (e.g., (data, label))."""
    item = dataset[index]
    if isinstance(item, tuple):
        return item[0]
    return item


def get_graph_elements(graph_, n_epochs):

    graph = graph_.tocoo()
    # eliminate duplicate entries by summing them together
    graph.sum_duplicates()
    # number of vertices in dataset
    n_vertices = graph.shape[1]
    # get the number of epochs based on the size of the dataset
    if n_epochs is None:
        # For smaller datasets we can use more epochs
        if graph.shape[0] <= 10000:
            n_epochs = 500
        else:
            n_epochs = 200
    # remove elements with very low probability
    graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    graph.eliminate_zeros()
    # get epochs per sample based upon edge probability
    epochs_per_sample = n_epochs * graph.data

    head = graph.row
    tail = graph.col
    weight = graph.data

    return graph, epochs_per_sample, head, tail, weight, n_vertices


class UMAPDataset(Dataset):
    def __init__(self, dataset, graph_, n_epochs=200):
        """
        UMAP Dataset that wraps an existing PyTorch Dataset.

        Args:
            dataset: A PyTorch Dataset. If items are tuples (e.g., (data, label)),
                     only the first element (data) is used.
            graph_: The UMAP graph (sparse matrix) defining edge relationships.
            n_epochs: Number of training epochs for edge sampling.
        """
        graph, epochs_per_sample, head, tail, weight, n_vertices = get_graph_elements(graph_, n_epochs)

        self.edges_to_exp, self.edges_from_exp = (
            np.repeat(head, epochs_per_sample.astype("int")),
            np.repeat(tail, epochs_per_sample.astype("int")),
        )
        shuffle_mask = np.random.permutation(np.arange(len(self.edges_to_exp)))
        self.edges_to_exp = self.edges_to_exp[shuffle_mask].astype(np.int64)
        self.edges_from_exp = self.edges_from_exp[shuffle_mask].astype(np.int64)
        self.dataset = dataset

    def __len__(self):
        return len(self.edges_to_exp)

    def __getitem__(self, index):
        edge_to_idx = self.edges_to_exp[index]
        edge_from_idx = self.edges_from_exp[index]
        edges_to_exp = get_item_from_dataset(self.dataset, edge_to_idx)
        edges_from_exp = get_item_from_dataset(self.dataset, edge_from_idx)
        return (edges_to_exp, edges_from_exp)


class MatchDataset(Dataset):
    def __init__(self, dataset, embeddings):
        """
        Dataset for matching parametric UMAP to non-parametric embeddings.

        Args:
            dataset: A PyTorch Dataset. If items are tuples (e.g., (data, label)),
                     only the first element (data) is used.
            embeddings: Pre-computed non-parametric UMAP embeddings.
        """
        self.embeddings = torch.Tensor(embeddings)
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = get_item_from_dataset(self.dataset, index)
        return data, self.embeddings[index]