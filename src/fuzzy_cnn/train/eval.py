import torch
from .model import CIFAR10ClassifierModel
from torch.utils.data import DataLoader

def evaluate_model(model: CIFAR10ClassifierModel, test_loader: DataLoader) -> float:
    # Don't build the dependency graph, we're not training so it's not useful here
    # and it just adds overhead
    with torch.no_grad():
        model.eval()

        correct = 0
        total = 0

        # We don't need the epochs just go through the whole set once
        for images, labels in test_loader:
            output = model(images)

            # after the first dimension which is the batch, the largest is going to be the label guess
            # so we just flatten to that
            batch_predicions = torch.argmax(output, dim=1)

            # Because the batch predictions and the labels are both tensors we can just compare them directly
            correct += (batch_predicions == labels).sum().item()

            total += labels.size(0)

        accuracy = 100 * (correct / total)
        return accuracy