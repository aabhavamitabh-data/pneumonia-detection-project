import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import SimpleCNN
import os
 
# ----------------------------
# CONFIG
# ----------------------------
OPTIMIZER_NAME = "rmsprop"
LEARNING_RATE = 0.001
EPOCHS = 40
USE_CLASS_WEIGHTS = True
SAVE_PATH = "models/best_simple_rmsprop.pt"
 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
print(f"Experiment: SimpleCNN + {OPTIMIZER_NAME.upper()}, {EPOCHS} epochs")
 
 
def compute_class_weights(train_loader, num_classes):
    """Counts samples per class and returns inverse-frequency weights,
    so the loss function penalizes mistakes on minority classes more."""
    counts = torch.zeros(num_classes)
    for _, labels in train_loader:
        for label in labels:
            counts[label] += 1
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return weights.to(DEVICE)
 
 
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
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
    return running_loss / total, correct / total
 
 
def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total
 
 
def main():
    train_loader, test_loader, classes = get_dataloaders()
    num_classes = len(classes)
 
    model = SimpleCNN(num_classes=num_classes).to(DEVICE)
 
    if USE_CLASS_WEIGHTS:
        weights = compute_class_weights(train_loader, num_classes)
        print("Class weights:", weights)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
 
    # RMSprop with momentum - a classic, simpler optimizer than Adam.
    # Often needs more careful tuning / more epochs to converge well,
    # which is exactly what we're testing here.
    optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
 
    best_val_acc = 0.0
    os.makedirs("models", exist_ok=True)
 
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, test_loader, criterion)
 
        print(f"Epoch {epoch}/{EPOCHS} | "
              f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
              f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")
 
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  -> New best model saved ({val_acc:.4f}) to {SAVE_PATH}")
 
    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
 
 
if __name__ == "__main__":
    main()