"""
Dataset loading for GTSRB (primary) and CIFAR-10 (secondary generalisation check).

IMPORTANT: transforms deliberately do NOT normalise. Images stay in [0, 1] so
that adversarial attacks operate in valid pixel space. Normalisation is applied
inside the model wrapper (see models.NormalizedModel).
"""
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import config


def _train_transform():
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),          # -> [0, 1]
    ])


def _eval_transform():
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),          # -> [0, 1]
    ])


def get_datasets(name=None):
    """Return (train_ds, val_ds, test_ds)."""
    name = name or config.DATASET

    if name == "gtsrb":
        full_train = datasets.GTSRB(
            root=config.DATA_DIR, split="train",
            transform=_train_transform(), download=True,
        )
        test_ds = datasets.GTSRB(
            root=config.DATA_DIR, split="test",
            transform=_eval_transform(), download=True,
        )
    elif name == "cifar10":
        full_train = datasets.CIFAR10(
            root=config.DATA_DIR, train=True,
            transform=_train_transform(), download=True,
        )
        test_ds = datasets.CIFAR10(
            root=config.DATA_DIR, train=False,
            transform=_eval_transform(), download=True,
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")

    n_val = int(0.1 * len(full_train))
    n_train = len(full_train) - n_val
    g = torch.Generator().manual_seed(config.SEED)
    train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=g)

    return train_ds, val_ds, test_ds


def get_loaders(name=None, batch_size=None):
    batch_size = batch_size or config.BATCH_SIZE
    train_ds, val_ds, test_ds = get_datasets(name)

    num_workers = 2 if config.DEVICE.type == "cuda" else 0
    mk = lambda ds, sh: DataLoader(
        ds, batch_size=batch_size, shuffle=sh,
        num_workers=num_workers, pin_memory=(config.DEVICE.type == "cuda"),
    )
    return mk(train_ds, True), mk(val_ds, False), mk(test_ds, False)


# Human-readable GTSRB class names, indexed 0-42.
GTSRB_CLASSES = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing for vehicles over 3.5t", "Right-of-way at intersection",
    "Priority road", "Yield", "Stop", "No vehicles",
    "Vehicles over 3.5t prohibited", "No entry", "General caution",
    "Dangerous curve left", "Dangerous curve right", "Double curve",
    "Bumpy road", "Slippery road", "Road narrows on the right", "Road work",
    "Traffic signals", "Pedestrians", "Children crossing", "Bicycles crossing",
    "Beware of ice/snow", "Wild animals crossing", "End of all limits",
    "Turn right ahead", "Turn left ahead", "Ahead only",
    "Go straight or right", "Go straight or left", "Keep right", "Keep left",
    "Roundabout mandatory", "End of no passing",
    "End of no passing by vehicles over 3.5t",
]

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def class_names(name=None):
    name = name or config.DATASET
    return GTSRB_CLASSES if name == "gtsrb" else CIFAR10_CLASSES
