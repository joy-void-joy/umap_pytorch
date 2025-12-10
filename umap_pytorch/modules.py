from pynndescent import NNDescent
import numpy as np
from sklearn.utils import check_random_state
from umap.umap_ import fuzzy_simplicial_set
import torch
from scipy.sparse import csr_matrix


def convert_distance_to_probability(distances, a=1.0, b=1.0):
    return -torch.log1p(a * distances ** (2 * b))


def compute_cross_entropy(
    probabilities_graph, probabilities_distance, EPS=1e-4, repulsion_strength=1.0
):
    # cross entropy
    attraction_term = -probabilities_graph * torch.nn.functional.logsigmoid(
        probabilities_distance
    )
    repellant_term = (
        -(1.0 - probabilities_graph)
        * (
            torch.nn.functional.logsigmoid(probabilities_distance)
            - probabilities_distance
        )
        * repulsion_strength
    )

    # balance the expected losses between atrraction and repel
    CE = attraction_term + repellant_term
    return attraction_term, repellant_term, CE


def umap_loss(embedding_to, embedding_from, _a, _b, batch_size, negative_sample_rate=5):
    # get negative samples by randomly shuffling the batch
    embedding_neg_to = embedding_to.repeat(negative_sample_rate, 1)
    repeat_neg = embedding_from.repeat(negative_sample_rate, 1)
    embedding_neg_from = repeat_neg[torch.randperm(repeat_neg.shape[0])]
    distance_embedding = torch.cat(
        (
            (embedding_to - embedding_from).norm(dim=1),
            (embedding_neg_to - embedding_neg_from).norm(dim=1),
        ),
        dim=0,
    )

    # convert probabilities to distances
    probabilities_distance = convert_distance_to_probability(distance_embedding, _a, _b)
    # set true probabilities based on negative sampling
    probabilities_graph = torch.cat(
        (torch.ones(batch_size), torch.zeros(batch_size * negative_sample_rate)),
        dim=0,
    )

    # compute cross entropy
    (attraction_loss, repellant_loss, ce_loss) = compute_cross_entropy(
        probabilities_graph.cuda(),
        probabilities_distance.cuda(),
    )
    loss = torch.mean(ce_loss)
    return loss


def get_umap_graph(X, n_neighbors=10, metric="cosine", random_state=None):
    random_state = check_random_state(None) if random_state == None else random_state
    # number of trees in random projection forest
    n_trees = 5 + int(round((X.shape[0]) ** 0.5 / 20.0))
    # max number of nearest neighbor iters to perform
    n_iters = max(5, int(round(np.log2(X.shape[0]))))
    # distance metric

    # get nearest neighbors
    nnd = NNDescent(
        X.reshape((len(X), int(np.prod(np.shape(X)[1:])))),
        n_neighbors=n_neighbors,
        metric=metric,
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
        verbose=True,
    )
    # get indices and distances
    neighbor_graph = nnd.neighbor_graph
    assert neighbor_graph is not None
    knn_indices, knn_dists = neighbor_graph

    # build fuzzy_simplicial_set
    result = fuzzy_simplicial_set(
        X=X,
        n_neighbors=n_neighbors,
        metric=metric,
        random_state=random_state,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
    )
    umap_graph = result[0]

    return umap_graph


def get_umap_graph_from_precomputed_knn(
    knn_indices, knn_dists, n_samples, n_neighbors=10, random_state=None
):
    """
    Build a UMAP graph from precomputed KNN indices and distances.

    This is useful when you have an external KNN index (like hnswlib) and want
    to build the UMAP graph without loading all data into memory.

    Args:
        knn_indices: Array of shape (n_samples, n_neighbors) with neighbor indices.
        knn_dists: Array of shape (n_samples, n_neighbors) with neighbor distances.
        n_samples: Total number of samples.
        n_neighbors: Number of neighbors used.
        random_state: Random state for reproducibility.

    Returns:
        UMAP graph as a sparse matrix.
    """
    random_state = (
        check_random_state(None)
        if random_state is None
        else check_random_state(random_state)
    )

    # Create a dummy X just for the fuzzy_simplicial_set call
    # It's only used for shape, not actual computation when knn_indices/dists are provided
    X_dummy = np.zeros((n_samples, 1))

    result = fuzzy_simplicial_set(
        X=X_dummy,
        n_neighbors=n_neighbors,
        metric="precomputed",
        random_state=random_state,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
    )
    umap_graph = result[0]

    return umap_graph


def get_knn_from_hnswlib(hnsw_index, data, n_neighbors=10):
    """
    Get KNN indices and distances from an hnswlib index.

    Args:
        hnsw_index: An hnswlib index that has been built on the data.
        data: Query data as numpy array of shape (n_samples, n_features).
        n_neighbors: Number of neighbors to retrieve.

    Returns:
        Tuple of (knn_indices, knn_dists) arrays.
    """
    labels, distances = hnsw_index.knn_query(data, k=n_neighbors)
    return labels.astype(np.int64), distances.astype(np.float32)


def get_knn_from_faiss(faiss_index, data, n_neighbors=10):
    """
    Get KNN indices and distances from a faiss index.

    Args:
        faiss_index: A faiss index that has been built on the data.
        data: Query data as numpy array of shape (n_samples, n_features).
        n_neighbors: Number of neighbors to retrieve.

    Returns:
        Tuple of (knn_indices, knn_dists) arrays.
    """
    distances, labels = faiss_index.search(data.astype(np.float32), n_neighbors)
    return labels.astype(np.int64), distances.astype(np.float32)
