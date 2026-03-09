import torchvision
from torch.utils.data import DataLoader
from fuzzy_cnn.common.config import DATA_DIR, settings
from fuzzy_cnn.common.preprocessing import get_train_transforms, get_inference_transforms

# We set the download flag on both the loaders to true so we grab them if they don't exist

def get_train_loader() -> DataLoader:
    # Train gets shuffled and set to train true because we want the train set, and we want random order during training which helps
    cifar_10_data = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=get_train_transforms())
    return DataLoader(cifar_10_data, batch_size=settings.train_batch_size, shuffle=True, num_workers=settings.num_workers)

def get_test_loader() -> DataLoader:
    # Here we're grabbing the test set, we turn shuffling off, it doesn't help a ton but it makes testing a bit more deterministic
    cifar_10_data = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=get_inference_transforms())
    return DataLoader(cifar_10_data, batch_size=settings.test_batch_size, shuffle=False, num_workers=settings.num_workers)
