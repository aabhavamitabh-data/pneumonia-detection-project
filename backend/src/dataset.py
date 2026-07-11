


import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DATA_DIR = "data/Covid19-dataset"
IMG_SIZE = 224          # standard input size for pretrained CNNs
BATCH_SIZE = 16         # small dataset -> small batch size works better

# ImageNet mean/std, since we'll fine-tune an ImageNet-pretrained backbone
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Training transforms include augmentation, since our dataset is small
# and augmentation helps the model generalize instead of memorizing
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# Test transforms are deterministic - no augmentation, just resize + normalize
test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def get_dataloaders():
    train_ds = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_transforms)
    test_ds = datasets.ImageFolder(f"{DATA_DIR}/test", transform=test_transforms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, train_ds.classes


if __name__ == "__main__":
    # Quick sanity check when running this file directly
    train_loader, test_loader, classes = get_dataloaders()
    print("Classes:", classes)
    print("Train batches:", len(train_loader))
    print("Test batches:", len(test_loader))

    # Peek at one batch to confirm shapes are correct
    images, labels = next(iter(train_loader))
    print("Batch image shape:", images.shape)   # expect [16, 3, 224, 224]
    print("Batch label shape:", labels.shape)   # expect [16]