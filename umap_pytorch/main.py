import pytorch_lightning as pl
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import mse_loss
import torch.nn.functional as F

from torch.utils.data import Subset
from umap_pytorch.data import UMAPDataset, MatchDataset, get_item_from_dataset
from umap_pytorch.modules import (
    get_umap_graph,
    umap_loss,
    get_umap_graph_from_hnswlib,
    get_umap_graph_from_precomputed_knn,
)
from umap_pytorch.model import default_encoder, default_decoder

from umap.umap_ import find_ab_params
import dill
from umap import UMAP


def extract_data_sample_from_dataset(dataset, sample_size=None, random_state=None):
    """
    Extract a sample of data from a dataset for graph construction.

    For large datasets, this avoids loading all data into memory by only
    extracting a representative sample for building the UMAP graph.

    Args:
        dataset: A PyTorch Dataset.
        sample_size: Number of samples to extract. If None, extracts all data.
        random_state: Random state for reproducible sampling.

    Returns:
        Tuple of (data_tensor, sample_indices) where sample_indices maps
        to the original dataset indices.
    """
    n_samples = len(dataset)

    if sample_size is None or sample_size >= n_samples:
        # Extract all data (only for small datasets)
        indices = list(range(n_samples))
        items = [get_item_from_dataset(dataset, i) for i in indices]
        return torch.stack(items), indices

    # Sample a subset of the data
    rng = np.random.default_rng(random_state)
    indices = rng.choice(n_samples, size=sample_size, replace=False).tolist()
    indices.sort()  # Sort for potential cache efficiency

    items = [get_item_from_dataset(dataset, i) for i in indices]
    return torch.stack(items), indices


""" Model """


class Model(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        encoder: nn.Module,
        decoder=None,
        beta=1.0,
        min_dist=0.1,
        reconstruction_loss=F.binary_cross_entropy_with_logits,
        match_nonparametric_umap=False,
    ):
        super().__init__()
        self.lr = lr
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta  # weight for reconstruction loss
        self.match_nonparametric_umap = match_nonparametric_umap
        self.reconstruction_loss = reconstruction_loss
        self._a, self._b = find_ab_params(1.0, min_dist)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        if not self.match_nonparametric_umap:
            (edges_to_exp, edges_from_exp) = batch
            embedding_to, embedding_from = (
                self.encoder(edges_to_exp),
                self.encoder(edges_from_exp),
            )
            encoder_loss = umap_loss(
                embedding_to,
                embedding_from,
                self._a,
                self._b,
                edges_to_exp.shape[0],
                negative_sample_rate=5,
            )
            self.log("umap_loss", encoder_loss, prog_bar=True)

            if self.decoder:
                recon = self.decoder(embedding_to)
                recon_loss = self.reconstruction_loss(recon, edges_to_exp)
                self.log("recon_loss", recon_loss, prog_bar=True)
                return encoder_loss + self.beta * recon_loss
            else:
                return encoder_loss

        else:
            data, embedding = batch
            embedding_parametric = self.encoder(data)
            encoder_loss = mse_loss(embedding_parametric, embedding)
            self.log("encoder_loss", encoder_loss, prog_bar=True)
            if self.decoder:
                recon = self.decoder(embedding_parametric)
                recon_loss = self.reconstruction_loss(recon, data)
                self.log("recon_loss", recon_loss, prog_bar=True)
                return encoder_loss + self.beta * recon_loss
            else:
                return encoder_loss


""" Datamodule """


class Datamodule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        batch_size,
        num_workers,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )


class PUMAP:
    def __init__(
        self,
        encoder=None,
        decoder=None,
        n_neighbors=10,
        min_dist=0.1,
        metric="euclidean",
        n_components=2,
        beta=1.0,
        reconstruction_loss=F.binary_cross_entropy_with_logits,
        random_state=None,
        lr=1e-3,
        epochs=10,
        batch_size=64,
        num_workers=1,
        num_gpus=1,
        match_nonparametric_umap=False,
        graph_sample_size=None,
        data_shape=None,
    ):
        """
        Parametric UMAP implementation using PyTorch.

        Args:
            encoder: Neural network encoder. If None, uses default MLP.
            decoder: Neural network decoder. If None, no reconstruction.
                     If True, uses default decoder.
            n_neighbors: Number of neighbors for KNN graph.
            min_dist: Minimum distance in embedding space.
            metric: Distance metric for KNN.
            n_components: Dimensionality of output embedding.
            beta: Weight for reconstruction loss.
            reconstruction_loss: Loss function for decoder.
            random_state: Random state for reproducibility.
            lr: Learning rate.
            epochs: Number of training epochs.
            batch_size: Training batch size.
            num_workers: DataLoader workers.
            num_gpus: GPU devices.
            match_nonparametric_umap: If True, train to match non-parametric UMAP.
            graph_sample_size: Number of samples to use for graph construction.
                               If None, uses all data (not recommended for large datasets).
                               For datasets > 100k samples, consider using 10000-50000.
            data_shape: Shape of a single data item (excluding batch dimension).
                        Required when using precomputed_graph. Inferred from dataset otherwise.
        """
        self.encoder = encoder
        self.decoder = decoder
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.n_components = n_components
        self.beta = beta
        self.reconstruction_loss = reconstruction_loss
        self.random_state = random_state
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_gpus = num_gpus
        self.match_nonparametric_umap = match_nonparametric_umap
        self.graph_sample_size = graph_sample_size
        self.data_shape = data_shape

    def fit(
        self, dataset, precomputed_graph=None, hnsw_index=None, knn_batch_size=1000
    ):
        """
        Fit the parametric UMAP model.

        Args:
            dataset: A PyTorch Dataset. Each item should be a data tensor or
                     a tuple where the first element is the data tensor.
            precomputed_graph: Optional precomputed UMAP graph (sparse matrix).
                               If provided, skips graph construction entirely.
            hnsw_index: Optional hnswlib index for efficient KNN queries.
                        Use this for very large datasets (>100k samples) to avoid
                        loading all data into memory. The index should be built
                        on the same data.
            knn_batch_size: Batch size for querying the hnswlib index.
        """
        trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=self.epochs)

        # Determine data shape for encoder/decoder initialization
        if self.data_shape is not None:
            data_shape = self.data_shape
        else:
            # Get shape from first item in dataset
            first_item = get_item_from_dataset(dataset, 0)
            data_shape = first_item.shape
            print(f"Inferred data shape: {data_shape}")

        encoder = (
            default_encoder(data_shape, self.n_components)
            if self.encoder is None
            else self.encoder
        )

        if self.decoder is None or isinstance(self.decoder, nn.Module):
            decoder = self.decoder
        elif self.decoder == True:
            decoder = default_decoder(data_shape, self.n_components)

        if not self.match_nonparametric_umap:
            self.model = Model(
                self.lr,
                encoder,
                decoder,
                beta=self.beta,
                min_dist=self.min_dist,
                reconstruction_loss=self.reconstruction_loss,
            )

            if precomputed_graph is not None:
                print("Using precomputed graph")
                graph = precomputed_graph
            elif hnsw_index is not None:
                # Use hnswlib index for efficient KNN - no need to load all data
                print(
                    f"Building UMAP graph using hnswlib index (batch_size={knn_batch_size})..."
                )
                graph = get_umap_graph_from_hnswlib(
                    hnsw_index,
                    dataset,
                    get_item_from_dataset,
                    n_neighbors=self.n_neighbors,
                    batch_size=knn_batch_size,
                    random_state=self.random_state,
                    verbose=True,
                )
            else:
                # Extract sample for graph construction
                n_dataset = len(dataset)
                sample_size = self.graph_sample_size

                if sample_size is not None and sample_size < n_dataset:
                    print(
                        f"Sampling {sample_size} points from {n_dataset} for graph construction..."
                    )
                    X_sample, sample_indices = extract_data_sample_from_dataset(
                        dataset, sample_size=sample_size, random_state=self.random_state
                    )
                    print("Building UMAP graph from sample...")
                    graph = get_umap_graph(
                        X_sample,
                        n_neighbors=self.n_neighbors,
                        metric=self.metric,
                        random_state=self.random_state,
                    )

                    # Create a subset dataset for training (only sampled points)
                    dataset = Subset(dataset, sample_indices)
                else:
                    print(
                        f"Extracting all {n_dataset} samples for graph construction..."
                    )
                    X_all, _ = extract_data_sample_from_dataset(
                        dataset, sample_size=None
                    )
                    print("Building UMAP graph...")
                    graph = get_umap_graph(
                        X_all,
                        n_neighbors=self.n_neighbors,
                        metric=self.metric,
                        random_state=self.random_state,
                    )

            trainer.fit(
                model=self.model,
                datamodule=Datamodule(
                    UMAPDataset(dataset, graph), self.batch_size, self.num_workers
                ),
            )
        else:
            # For match_nonparametric_umap mode, sample if needed
            n_dataset = len(dataset)
            sample_size = self.graph_sample_size

            if sample_size is not None and sample_size < n_dataset:
                print(
                    f"Sampling {sample_size} points from {n_dataset} for non-parametric UMAP..."
                )
                X_sample, sample_indices = extract_data_sample_from_dataset(
                    dataset, sample_size=sample_size, random_state=self.random_state
                )
                dataset = Subset(dataset, sample_indices)
            else:
                print(f"Extracting all {n_dataset} samples for non-parametric UMAP...")
                X_sample, _ = extract_data_sample_from_dataset(
                    dataset, sample_size=None
                )

            print("Fitting Non parametric Umap")
            non_parametric_umap = UMAP(
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                metric=self.metric,
                n_components=self.n_components,
                random_state=self.random_state,
                verbose=True,
            )
            non_parametric_embeddings = non_parametric_umap.fit_transform(
                torch.flatten(X_sample, 1, -1).numpy()
            )
            self.model = Model(
                self.lr,
                encoder,
                decoder,
                beta=self.beta,
                reconstruction_loss=self.reconstruction_loss,
                match_nonparametric_umap=self.match_nonparametric_umap,
            )
            print("Training NN to match embeddings")
            trainer.fit(
                model=self.model,
                datamodule=Datamodule(
                    MatchDataset(dataset, non_parametric_embeddings),
                    self.batch_size,
                    self.num_workers,
                ),
            )

    @torch.no_grad()
    def transform(self, X):
        print(
            f"Reducing array of shape {X.shape} to ({X.shape[0]}, {self.n_components})"
        )
        return self.model.encoder(X).detach().cpu().numpy()

    @torch.no_grad()
    def inverse_transform(self, Z):
        return self.model.decoder(Z).detach().cpu().numpy()

    def save(self, path):
        with open(path, "wb") as oup:
            dill.dump(self, oup)
        print(f"Pickled PUMAP object at {path}")


def load_pumap(path):
    print("Loading PUMAP object from pickled file.")
    with open(path, "rb") as inp:
        return dill.load(inp)


if __name__ == "__main__":
    pass
