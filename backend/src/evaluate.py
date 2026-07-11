# Loads a saved model and computes detailed evaluation metrics:
# confusion matrix, per-class precision/recall/F1, and overall report.

import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from model import get_resnet_model, SimpleCNN

MODEL_PATH = "models/best_simple_adam_20ep.pt"
MODEL_TYPE = "simple"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_type, num_classes, path):
    if model_type == "resnet":
        model = get_resnet_model(num_classes=num_classes, freeze_backbone=True)
    elif model_type == "simple":
        model = SimpleCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def get_predictions(model, loader):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return all_labels, all_preds


def main():
    train_loader, test_loader, classes = get_dataloaders()
    num_classes = len(classes)

    model = load_model(MODEL_TYPE, num_classes, MODEL_PATH)
    y_true, y_pred = get_predictions(model, test_loader)

    print("Classes:", classes)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    print(cm)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {MODEL_TYPE}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
    