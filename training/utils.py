"""
Training utilities for training models, and visualizing results.

Provides functions for training a PyTorch model using a provided
data loader, optimizer, and loss criterion, and saving the resulting
learning curve as a figure.
"""


from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def train(
        train_loader: DataLoader,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        epochs: int,
        validation_loader: DataLoader | None = None,
) -> dict[str, list[float]]:
    """
    Train a PyTorch model and optionally evaluate it on a validation dataset.

    The model is trained for the specified number of epochs. For each epoch,
    the average training loss is calculated across all batches. If a
    validation data loader is provided, the model is also evaluated on the
    validation set after the training phase of each epoch.

    During validation, gradient computation is disabled and the model is
    switched to evaluation mode. The function returns the recorded training
    and validation losses for each epoch.
    """
    print("Initializing training:")

    training_history = []
    validation_history = []

    for epoch in range(epochs):
        print(f"Currently at epoch {epoch + 1}...")

        # Training
        model.train()

        epoch_loss = 0.0

        for x, y in train_loader:
            optimizer.zero_grad()

            logits = model(x)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        epoch_loss /= len(train_loader)
        training_history.append(epoch_loss)

        # Validation
        if validation_loader is not None:
            model.eval()

            validation_loss = 0.0

            with torch.no_grad():
                for x, y in validation_loader:
                    logits = model(x)

                    loss = criterion(
                        logits.reshape(-1, logits.size(-1)),
                        y.reshape(-1),
                    )

                    validation_loss += loss.item()

            validation_loss /= len(validation_loader)
            validation_history.append(validation_loss)

            print(
                f"Training loss: {epoch_loss:.4f} | "
                f"Validation loss: {validation_loss:.4f}"
            )
        else:
            print(f"Training loss: {epoch_loss:.4f}")

    return {
        "training": training_history,
        "validation": validation_history,
    }

def save_learning_curves(
        histories: dict[str, list[float]],
        path: Path,
) -> None:
    """
    Save training and validation loss histories as a learning-curve plot.

    Each entry in histories is plotted as a separate curve, using the
    dictionary key as the curve label. The resulting figure is saved to the
    specified path.
    """
    plt.figure(figsize=(7, 5))

    for name, history in histories.items():
        plt.plot(
            range(1, len(history) + 1),
            history,
            marker="o",
            label=name,
        )

    plt.title("Learning Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()