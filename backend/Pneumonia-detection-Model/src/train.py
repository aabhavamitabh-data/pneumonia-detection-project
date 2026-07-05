# Trains a model (SimpleCNN or ResNet18) on the Covid19-dataset,
# tracks loss/accuracy per epoch, and saves the best model to disk.

import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import SimpleCNN, get_resnet_model
import os

# ----------------------------
# CONFIG - change these for your hyperparameter experiments
# ----------------------------
MODEL_TYPE = "resnet"        # "simple" or "resnet"
OPTIMIZER_NAME = "adam"      # "adam", "sgd", or "rmsprop"
LEARNING_RATE = 0.001
EPOCHS = 15
USE_CLASS_WEIGHTS = True     # helps with our imbalanced classes (Covid: 111 vs 70/70)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def get_optimizer(name, params, lr):
    if name == "adam":
        return optim.Adam(params, lr=lr)
    elif name == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9)
    elif name == "rmsprop":
        return optim.RMSprop(params, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def compute_class_weights(train_loader, num_classes):
    """Counts samples per class in the training set and returns inverse-frequency
    weights, so the loss function penalizes mistakes on minority classes more."""
    counts = torch.zeros(num_classes)
    for _, labels in train_loader:
        for label in labels:
            counts[label] += 1
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes  # normalize
    return weights.to(DEVICE)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    train_loader, test_loader, classes = get_dataloaders()
    num_classes = len(classes)

    # Build model
    if MODEL_TYPE == "simple":
        model = SimpleCNN(num_classes=num_classes)
    elif MODEL_TYPE == "resnet":
        model = get_resnet_model(num_classes=num_classes, freeze_backbone=True)
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

    model = model.to(DEVICE)

    # Loss function - weighted if class imbalance handling is on
    if USE_CLASS_WEIGHTS:
        weights = compute_class_weights(train_loader, num_classes)
        print("Class weights:", weights)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # Only optimize parameters that require gradients (matters for frozen ResNet backbone)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = get_optimizer(OPTIMIZER_NAME, trainable_params, LEARNING_RATE)

    best_val_acc = 0.0
    os.makedirs("models", exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, test_loader, criterion)

        print(f"Epoch {epoch}/{EPOCHS} | "
              f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
              f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        # Save the best model seen so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = f"models/best_{MODEL_TYPE}_{OPTIMIZER_NAME}.pt"
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best model saved ({val_acc:.4f}) to {save_path}")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
    