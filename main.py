from src.dataset import DatasetLoader


def main():

    loader = DatasetLoader()

    train_loader, val_loader, test_loader = loader.run()


if __name__ == "__main__":
    main()