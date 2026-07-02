import torch

from pathlib import Path

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from src.config import Config


class DatasetLoader:
    """
    Handles dataset loading and DataLoader creation.
    """

    def __init__(self):

        config = Config()

        self.processed_data_path = Path(
            config.data["processed_data_path"]
        )

        self.image_size = config.data["image_size"]

        self.batch_size = config.data["batch_size"]

        self.num_workers = config.data["num_workers"]


    def get_transforms(self):
        """
        Creates image transformations for training,
        validation and testing.
        """

        train_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        test_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        return train_transform, test_transform
    

    def get_datasets(self):
        """
        Creates train, validation and test datasets.
        """

        train_transform, test_transform = self.get_transforms()

        train_dataset = datasets.ImageFolder(
            root=self.processed_data_path / "train",
            transform=train_transform
        )

        val_dataset = datasets.ImageFolder(
            root=self.processed_data_path / "val",
            transform=test_transform
        )

        test_dataset = datasets.ImageFolder(
            root=self.processed_data_path / "test",
            transform=test_transform
        )

        print("=" * 50)
        print("Datasets Loaded Successfully")
        print("=" * 50)

        print(f"Training Images   : {len(train_dataset)}")
        print(f"Validation Images : {len(val_dataset)}")
        print(f"Testing Images    : {len(test_dataset)}")

        print("\nClass Mapping")
        print(train_dataset.class_to_idx)

        return train_dataset, val_dataset, test_dataset
    

    def get_dataloaders(self):
        """
        Creates DataLoaders for train, validation and test datasets.
        """

        train_dataset, val_dataset, test_dataset = self.get_datasets()

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()

        )

        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )

        print("=" * 50)
        print("DataLoaders Created Successfully")
        print("=" * 50)

        print(f"Training Batches   : {len(train_loader)}")
        print(f"Validation Batches : {len(val_loader)}")
        print(f"Testing Batches    : {len(test_loader)}")

        return train_loader, val_loader, test_loader
    
    def run(self):
        """
        Executes the complete data loading pipeline.
        """

        train_loader, val_loader, test_loader = self.get_dataloaders()

        return train_loader, val_loader, test_loader