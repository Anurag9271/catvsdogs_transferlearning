import torch
import mlflow

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.config import Config
from src.dataset import DatasetLoader
from src.model import ModelBuilder


class Evaluator:
    """
    Evaluates the trained model on the test dataset.
    """

    def __init__(self):

        config = Config()

        self.model_path = Path(
            config.paths["model_path"]
        )

        mlflow.set_experiment(
            "CatsVsDogs_TransferLearning"
        )

    def setup_device(self):
        """
        Selects the available device.
        """

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        print("=" * 50)
        print("Evaluation Device")
        print("=" * 50)
        print(f"Using Device : {device}")

        return device

    def load_model(self, device):
        """
        Loads the trained model.
        """

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        builder = ModelBuilder()

        model = builder.run()

        model.load_state_dict(
            torch.load(
                self.model_path,
                map_location=device
            )
        )

        model.to(device)

        model.eval()

        print("=" * 50)
        print("Model Loaded Successfully")
        print("=" * 50)

        return model

    def evaluate(
        self,
        model,
        test_loader,
        device
    ):
        """
        Evaluates the model.
        """

        predictions = []

        labels_list = []

        with torch.no_grad():

            for images, labels in test_loader:

                images = images.to(device)

                labels = labels.to(device)

                outputs = model(images)

                _, predicted = torch.max(
                    outputs,
                    dim=1
                )

                predictions.extend(
                    predicted.cpu().numpy()
                )

                labels_list.extend(
                    labels.cpu().numpy()
                )

        accuracy = accuracy_score(
            labels_list,
            predictions
        )

        precision = precision_score(
            labels_list,
            predictions
        )

        recall = recall_score(
            labels_list,
            predictions
        )

        f1 = f1_score(
            labels_list,
            predictions
        )

        confusion = confusion_matrix(
            labels_list,
            predictions
        )

        report = classification_report(
            labels_list,
            predictions,
            target_names=["Cat", "Dog"]
        )

        print("=" * 50)
        print("Evaluation Results")
        print("=" * 50)

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        print("\nConfusion Matrix")
        print(confusion)

        print("\nClassification Report")
        print(report)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    def log_metrics(self, metrics):
        """
        Logs evaluation metrics to MLflow.
        """

        with mlflow.start_run(run_name="Evaluation"):

            mlflow.log_metric(
                "test_accuracy",
                metrics["accuracy"]
            )

            mlflow.log_metric(
                "test_precision",
                metrics["precision"]
            )

            mlflow.log_metric(
                "test_recall",
                metrics["recall"]
            )

            mlflow.log_metric(
                "test_f1_score",
                metrics["f1_score"]
            )

    def run(self):
        """
        Executes the evaluation pipeline.
        """

        loader = DatasetLoader()

        _, _, test_loader = loader.run()

        device = self.setup_device()

        model = self.load_model(device)

        metrics = self.evaluate(
            model,
            test_loader,
            device
        )

        self.log_metrics(metrics)

        print("=" * 50)
        print("Evaluation Completed Successfully")
        print("=" * 50)