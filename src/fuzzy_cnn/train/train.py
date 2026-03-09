from torch import nn
import torch
from fuzzy_cnn.common.config import CHECKPOINT_DIR, settings
from fuzzy_cnn.common.io import save_checkpoint
from fuzzy_cnn.train.datasets import get_test_loader, get_train_val_loaders
from fuzzy_cnn.train.eval import evaluate_model
from .model import CIFAR10ClassifierModel

def train_model() -> None:
    model = CIFAR10ClassifierModel()
    train_loader, val_loader = get_train_val_loaders()
    test_loader = get_test_loader()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Adam is a good default optimizer, cross entropy makes the most sense
    # because we want to measure how correct our probability guess is for a set of choices
    # and it penalizes confident wrong guesses
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, settings.train_epochs + 1):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            # We need to zero out the gradients from other runs, otherwise these accumulate
            # so adam would work on this batch, then this batch and the previous and so on
            optimizer.zero_grad()

            # Make predictions and calculate the loss
            output = model(images)
            loss = loss_fn(output, labels)

            # Do the backpropagation from the loss and step the network with the optimizer
            loss.backward() # this computes the gradient & adds them to the running total, which is why we need zero_grad
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation phase - eval() disables dropout and switches BatchNorm to use
        # running statistics rather than batch statistics
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        # Switch off gradient tracking, we just want the results now
        with torch.no_grad():
            for images, labels in val_loader:
                output = model(images)
                loss = loss_fn(output, labels)
                val_loss += loss.item()
                predicted = output.argmax(dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total

        print(f"Epoch {epoch}/{settings.train_epochs} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")
        save_checkpoint(model, optimizer, epoch, CHECKPOINT_DIR / f"train_cnn_{epoch}.pt")

    # Test how good our final result is
    test_accuracy = evaluate_model(model, test_loader)
    print(f"Final test accuracy: {test_accuracy:.2f}%")

    save_checkpoint(model, optimizer, epoch, CHECKPOINT_DIR / "final.pt")