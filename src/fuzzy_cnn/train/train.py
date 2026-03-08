from torch import nn
import torch
from fuzzy_cnn.common.config import CHECKPOINT_DIR, LEARNING_RATE, TRAIN_EPOCHS
from fuzzy_cnn.common.io import save_checkpoint
from fuzzy_cnn.train.datasets import get_train_loader
from .model import CIFAR10ClassifierModel

def train() -> None:
    model = CIFAR10ClassifierModel()
    train_loader = get_train_loader()

    # Adam is a good default optimizer, cross entropy makes the most sense
    # because we want to measure how correct our probability guess is for a set of choices
    # and it penalizes confident wrong guesses
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, TRAIN_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            # Make predictions and calculate the loss
            output = model(images)
            loss = loss_fn(output, labels)

            # We need to zero out the gradients from other runs, otherwise these accumulate
            # so adam would work on this batch, then this batch and the previous and so on
            optimizer.zero_grad() 

            # Do the backpropagation from the loss and step the network with the optimizer
            loss.backward() # this computes the gradient & adds them to the running total, which is why we need zero_grad
            optimizer.step()  

            running_loss += loss.item()
    
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch}/{TRAIN_EPOCHS} - Loss: {avg_loss:.4f}")
    
    # TODO: We'll just save at the end for now, but really it would be better if we saved more checkpoints on the epochs
    save_checkpoint(model, optimizer, epoch, CHECKPOINT_DIR / "final.pt")