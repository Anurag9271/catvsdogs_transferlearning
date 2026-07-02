import torch
import torch.nn as nn
import torch.optim as optim

import mlflow
import mlflow.pytorch

from pathlib import Path

from src.config import Config
from src.dataset import DatasetLoader
from src.model import ModelBuilder


class Trainer:
    """
    Handles model training and validation.
    """

    def __init__(self):

        config = Config()

        self.epochs = config.training["epochs"]
        self.learning_rate = config.training["learning_rate"]

        self.model_path = Path(
            config.paths["model_path"]
        )

        mlflow.set_experiment(
            "CatsVsDogs_TransferLearning"
        )

    def setup_device(self):
        """
        Selects the available device for training.
        """

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        print("=" * 50)
        print("Device Configuration")
        print("=" * 50)
        print(f"Using Device : {device}")

        return device

    def setup_loss_function(self):
        """
        Creates the loss function.
        """

        criterion = nn.CrossEntropyLoss()

        print("=" * 50)
        print("Loss Function")
        print("=" * 50)
        print(f"Loss : {criterion}")

        return criterion

    def setup_optimizer(self, model):
        """
        Creates the optimizer.
        """

        optimizer = optim.Adam(
            filter(
                lambda parameter: parameter.requires_grad,
                model.parameters()
            ),
            lr=self.learning_rate
        )

        print("=" * 50)
        print("Optimizer")
        print("=" * 50)
        print(f"Optimizer     : Adam")
        print(f"Learning Rate : {self.learning_rate}")

        return optimizer

    def train_one_epoch(
        self,
        model,
        train_loader,
        criterion,
        optimizer,
        device
    ):
        """
        Train the model for one epoch.
        """

        model.train()

        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            _, predictions = torch.max(outputs, dim=1)

            correct_predictions += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

        epoch_loss = running_loss / len(train_loader)

        epoch_accuracy = (
            correct_predictions / total_samples
        ) * 100

        return epoch_loss, epoch_accuracy

    def validate_one_epoch(
        self,
        model,
        val_loader,
        criterion,
        device
    ):
        """
        Validate the model for one epoch.
        """

        model.eval()

        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                running_loss += loss.item()

                _, predictions = torch.max(outputs, dim=1)

                correct_predictions += (
                    predictions == labels
                ).sum().item()

                total_samples += labels.size(0)

        epoch_loss = running_loss / len(val_loader)

        epoch_accuracy = (
            correct_predictions / total_samples
        ) * 100

        return epoch_loss, epoch_accuracy

    def save_best_model(
        self,
        model,
        val_accuracy,
        best_accuracy
    ):
        """
        Saves the model if validation accuracy improves.
        """

        if val_accuracy > best_accuracy:

            self.model_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            torch.save(
                model.state_dict(),
                self.model_path
            )

            # Log model artifact to MLflow
            mlflow.log_artifact(
                str(self.model_path)
            )

            print(
                f"Best model saved "
                f"({val_accuracy:.2f}%)"
            )

            return val_accuracy

        return best_accuracy

    def train(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device
    ):
        """
        Complete training loop.
        """

        best_accuracy = 0.0

        with mlflow.start_run():

            # -----------------------------
            # Log Hyperparameters
            # -----------------------------

            mlflow.log_param(
                "model",
                "ResNet50"
            )

            mlflow.log_param(
                "epochs",
                self.epochs
            )

            mlflow.log_param(
                "learning_rate",
                self.learning_rate
            )

            mlflow.log_param(
                "batch_size",
                train_loader.batch_size
            )

            # -----------------------------
            # Training Loop
            # -----------------------------

            for epoch in range(self.epochs):

                train_loss, train_accuracy = self.train_one_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    device
                )

                val_loss, val_accuracy = self.validate_one_epoch(
                    model,
                    val_loader,
                    criterion,
                    device
                )

                # -----------------------------
                # Log Metrics
                # -----------------------------

                mlflow.log_metric(
                    "train_loss",
                    train_loss,
                    step=epoch
                )

                mlflow.log_metric(
                    "train_accuracy",
                    train_accuracy,
                    step=epoch
                )

                mlflow.log_metric(
                    "val_loss",
                    val_loss,
                    step=epoch
                )

                mlflow.log_metric(
                    "val_accuracy",
                    val_accuracy,
                    step=epoch
                )

                best_accuracy = self.save_best_model(
                    model,
                    val_accuracy,
                    best_accuracy
                )

                print("=" * 60)
                print(
                    f"Epoch [{epoch + 1}/{self.epochs}]"
                )
                print(
                    f"Train Loss : {train_loss:.4f}"
                )
                print(
                    f"Train Acc  : {train_accuracy:.2f}%"
                )
                print(
                    f"Val Loss   : {val_loss:.4f}"
                )
                print(
                    f"Val Acc    : {val_accuracy:.2f}%"
                )

            # Log Best Accuracy
            mlflow.log_metric(
                "best_validation_accuracy",
                best_accuracy
            )

            print("=" * 60)
            print("Training Completed")

    def run(self):
        """
        Executes the complete training pipeline.
        """

        loader = DatasetLoader()

        train_loader, val_loader, _ = loader.run()

        builder = ModelBuilder()

        model = builder.run()

        device = self.setup_device()

        model = model.to(device)

        criterion = self.setup_loss_function()

        optimizer = self.setup_optimizer(model)

        self.train(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device
        )