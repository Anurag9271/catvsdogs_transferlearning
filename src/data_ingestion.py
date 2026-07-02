import shutil
import random
from pathlib import Path
from PIL import Image
from src.config import Config


class DataIngestion:
    """
    Handles dataset validation and preparation.
    """

    def __init__(self):
        config = Config()

        # Dataset paths
        self.raw_data_path = Path(config.data["raw_data_path"])
        self.processed_data_path = Path(config.data["processed_data_path"])

        # Split ratios
        self.train_split = config.data["train_split"]
        self.val_split = config.data["val_split"]
        self.test_split = config.data["test_split"]

        self.random_seed = config.data["random_seed"]
        self.classes = config.data["classes"]

    def validate_dataset(self) -> None:
        """
        Validate dataset structure.
        """

        if not self.raw_data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.raw_data_path}"
            )

        print("=" * 50)
        print("Dataset Validation Successful")
        print("=" * 50)

        total_images = 0

        for category in self.classes:

            folder = self.raw_data_path / category

            if not folder.exists():
                raise FileNotFoundError(
                    f"{category} folder not found."
                )

            image_count = len(list(folder.glob("*")))

            total_images += image_count

            print(f"{category}: {image_count}")

        print(f"Total: {total_images}")


    def create_processed_directories(self):
        """
        Create train, validation, and test directories.
        """

        for split in ["train", "val", "test"]:
            for category in self.classes:

                folder = self.processed_data_path / split / category

                folder.mkdir(parents=True, exist_ok=True)

        print("Processed directories created successfully.")

    def get_image_paths(self, category: str) -> list[Path]:
        """
        Returns all image paths for the given category.
        """

        category_path = self.raw_data_path / category

        image_paths = [
            image
            for image in category_path.iterdir()
            if image.is_file()
        ]

        return image_paths
    
    def split_images(self, image_paths: list[Path]):
        """
        Split images into train, validation and test.
        """

        random.Random(self.random_seed).shuffle(image_paths)

        total_images = len(image_paths)

        train_end = int(total_images * self.train_split)

        val_end = train_end + int(total_images * self.val_split)

        train_images = image_paths[:train_end]

        val_images = image_paths[train_end:val_end]

        test_images = image_paths[val_end:]

        return train_images, val_images, test_images

    def copy_images(self,image_paths: list[Path],split: str,category: str):
        """
        Copies images to the processed dataset.

        Args:
            image_paths: List of image paths.
            split: train / val / test
            category: Cat / Dog
        """

        destination = self.processed_data_path / split / category

        for image_path in image_paths:

            try:

                with Image.open(image_path) as img:
                    img.verify()

                shutil.copy(image_path, destination)

            except Exception:

                print(f"Skipping corrupted image: {image_path}")

    def clear_processed_directory(self):
        """
        Removes processed dataset if it already exists.
        """

        if self.processed_data_path.exists():

            shutil.rmtree(self.processed_data_path)

            print("Old processed dataset removed.")


    def run(self) -> None:

        print("\nStarting Data Ingestion Pipeline...\n")

        self.validate_dataset()

        self.clear_processed_directory()

        self.create_processed_directories()

        for category in self.classes:

            print(f"\nProcessing {category}")

            image_paths = self.get_image_paths(category)

            train_images, val_images, test_images = self.split_images(image_paths)

            self.copy_images(train_images, "train", category)

            self.copy_images(val_images, "val", category)

            self.copy_images(test_images, "test", category)

        print("\nData Ingestion Completed Successfully.")