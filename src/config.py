import yaml


class Config:
    """
    Reads project configuration from params.yaml
    """

    def __init__(self, config_path="params.yaml"):

        with open(config_path, "r") as file:
            params = yaml.safe_load(file)

        self.data = params["data"]
        self.training = params["training"]
        self.model = params["model"]
        self.paths = params["paths"]

    def get(self):
        return {
            "data": self.data,
            "training": self.training,
            "model": self.model,
            "paths": self.paths
        }