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
        X.reshape((len(X), np.product(np.shape(X)[1:]))),
        n_neighbors=n_neighbors,
        metric=metric,
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
        verbose=True,
    )
    # get indices and distances
    knn_indices, knn_dists = nnd.neighbor_graph

    # get indices and distances
    knn_indices, knn_dists = nnd.neighbor_graph
    # build fuzzy_simplicial_set
    umap_graph, sigmas, rhos = fuzzy_simplicial_set(
        X=X,
        n_neighbors=n_neighbors,
        metric=metric,
        random_state=random_state,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
    )

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

    umap_graph, sigmas, rhos = fuzzy_simplicial_set(
        X=X_dummy,
        n_neighbors=n_neighbors,
        metric="precomputed",
        random_state=random_state,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
    )

    return umap_graph


def get_knn_from_hnswlib_batched(
    hnsw_index, dataset, get_item_fn, n_neighbors=10, batch_size=1000, verbose=True
):
    """
    Query an hnswlib index in batches to get KNN for a dataset.

    This function processes the dataset in batches to avoid loading all data
    into memory at once. It's suitable for very large datasets.

    Args:
        hnsw_index: An hnswlib index that has been built on the data.
        dataset: A PyTorch Dataset.
        get_item_fn: Function to extract data from dataset items (handles tuples).
        n_neighbors: Number of neighbors to retrieve.
        batch_size: Number of samples to process at once.
        verbose: Whether to print progress.

    Returns:
        Tuple of (knn_indices, knn_dists) arrays.
    """
    n_samples = len(dataset)
    knn_indices = np.zeros((n_samples, n_neighbors), dtype=np.int64)
    knn_dists = np.zeros((n_samples, n_neighbors), dtype=np.float32)

    n_batches = (n_samples + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)

        if verbose and batch_idx % 10 == 0:
            print(
                f"Processing batch {batch_idx + 1}/{n_batches} (samples {start_idx}-{end_idx})"
            )

        # Load batch data
        batch_data = []
        for i in range(start_idx, end_idx):
            item = get_item_fn(dataset, i)
            if isinstance(item, torch.Tensor):
                item = item.numpy()
            batch_data.append(item.flatten())

        batch_data = np.array(batch_data, dtype=np.float32)

        # Query hnswlib index
        labels, distances = hnsw_index.knn_query(batch_data, k=n_neighbors)

        knn_indices[start_idx:end_idx] = labels
        knn_dists[start_idx:end_idx] = distances

    return knn_indices, knn_dists


def get_umap_graph_from_hnswlib(
    hnsw_index,
    dataset,
    get_item_fn,
    n_neighbors=10,
    batch_size=1000,
    random_state=None,
    verbose=True,
):
    """
    Build a UMAP graph using an hnswlib index for KNN queries.

    This allows building a UMAP graph on very large datasets without loading
    all data into memory. The hnswlib index should have been built on the
    same data.

    Args:
        hnsw_index: An hnswlib index built on the dataset.
        dataset: A PyTorch Dataset.
        get_item_fn: Function to extract data from dataset items.
        n_neighbors: Number of neighbors for UMAP graph.
        batch_size: Batch size for KNN queries.
        random_state: Random state for reproducibility.
        verbose: Whether to print progress.

    Returns:
        UMAP graph as a sparse matrix.
    """
    if verbose:
        print(
            f"Querying hnswlib index for {len(dataset)} samples with {n_neighbors} neighbors..."
        )

    knn_indices, knn_dists = get_knn_from_hnswlib_batched(
        hnsw_index, dataset, get_item_fn, n_neighbors, batch_size, verbose
    )

    if verbose:
        print("Building fuzzy simplicial set...")

    return get_umap_graph_from_precomputed_knn(
        knn_indices, knn_dists, len(dataset), n_neighbors, random_state
    )
