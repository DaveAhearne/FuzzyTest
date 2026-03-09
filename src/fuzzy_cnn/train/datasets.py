import torchvision
import torch
from torch.utils.data import DataLoader, Subset
from fuzzy_cnn.common.config import DATA_DIR, settings
from fuzzy_cnn.common.preprocessing import get_train_transforms, get_inference_transforms

# We set the download flag on both the loaders to true so we grab them if they don't exist
def get_train_val_loaders() -> tuple[DataLoader, DataLoader]:
    cifar_10_data_train = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=get_train_transforms())
    cifar_10_data_val = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=get_inference_transforms())

    n = len(cifar_10_data_train)
    # We do a 90/10 split on 
    train_size = int(n * 0.9)
    indices = torch.randperm(n)
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    train_subset = Subset(cifar_10_data_train, train_indices)
    val_subset = Subset(cifar_10_data_val, val_indices)

    return (
        DataLoader(train_subset, batch_size=settings.train_batch_size, shuffle=True, num_workers=settings.num_workers),
        DataLoader(val_subset, batch_size=settings.validation_batch_size, shuffle=False, num_workers=settings.num_workers)
    )

def get_test_loader() -> DataLoader:
    # Here we're grabbing the test set, we turn shuffling off, it doesn't help a ton but it makes testing a bit more deterministic
    cifar_10_data = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=get_inference_transforms())
    return DataLoader(cifar_10_data, batch_size=settings.test_batch_size, shuffle=False, num_workers=settings.num_workers)
