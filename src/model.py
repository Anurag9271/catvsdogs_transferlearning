import torch
import torch.nn as nn
from torchvision import models

from src.config import Config


class ModelBuilder:
    """
    Builds a pretrained ResNet50 model for transfer learning.
    """

    def __init__(self):

        config = Config()

        self.model_name = config.model["name"]

        self.pretrained = config.model["pretrained"]

        self.num_classes = config.model["num_classes"]

    def load_pretrained_model(self):
        """
        Loads the pretrained ResNet50 model.
        """

        if self.model_name == "resnet50":

            if self.pretrained:
                weights = models.ResNet50_Weights.DEFAULT
            else:
                weights = None

            model = models.resnet50(weights=weights)

        else:
            raise ValueError(
                f"Unsupported model: {self.model_name}"
            )

        print("=" * 50)
        print("Pretrained Model Loaded Successfully")
        print("=" * 50)
        print(f"Model Name : {self.model_name}")
        print(f"Pretrained : {self.pretrained}")

        return model
    
    def freeze_layers(self, model):
        """
        Freezes all pretrained layers.
        """

        for parameter in model.parameters():
            parameter.requires_grad = False

        print("=" * 50)
        print("Pretrained Layers Frozen")
        print("=" * 50)

        return model
    
    def replace_classifier(self, model):
        """
        Replaces the final fully connected layer.
        """

        in_features = model.fc.in_features

        model.fc = nn.Linear(
            in_features=in_features,
            out_features=self.num_classes
        )

        print("=" * 50)
        print("Classifier Replaced Successfully")
        print("=" * 50)
        print(f"Input Features : {in_features}")
        print(f"Output Classes : {self.num_classes}")

        return model
    

    def print_model_summary(self, model):
        """
        Prints model parameter statistics.
        """

        total_params = sum(
            parameter.numel()
            for parameter in model.parameters()
        )

        trainable_params = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        frozen_params = total_params - trainable_params

        print("=" * 50)
        print("Model Summary")
        print("=" * 50)
        print(f"Total Parameters     : {total_params:,}")
        print(f"Trainable Parameters : {trainable_params:,}")
        print(f"Frozen Parameters    : {frozen_params:,}")


    def get_model(self):
        """
        Builds and returns the transfer learning model.
        """

        model = self.load_pretrained_model()

        model = self.freeze_layers(model)

        model = self.replace_classifier(model)

        self.print_model_summary(model)

        return model
    
    def run(self):
        """
        Executes the model building pipeline.
        """

        return self.get_model()