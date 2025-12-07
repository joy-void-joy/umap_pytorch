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
        # Remove the channel dimension squeeze and keep shape consistent
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
    