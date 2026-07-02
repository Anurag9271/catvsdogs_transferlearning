from src.model import ModelBuilder
from src.train import Trainer


def main():

    builder = ModelBuilder()

    model = builder.run()

    trainer = Trainer()

    trainer.setup_optimizer(model)


if __name__ == "__main__":
    main()