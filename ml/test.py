import argparse

import torch
from model import Config, LitEMNIST, TransformBuilder
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy
from torchvision import datasets


def load_model(checkpoint_path: str, config: Config, device: torch.device):
    """Load the model from checkpoint and move to the correct device."""
    model = LitEMNIST.load_from_checkpoint(checkpoint_path, config=config)
    model.to(device)  # Move model to device
    return model


def get_test_dataloader(data_dir: str, batch_size: int, num_workers: int):
    """Create a DataLoader for the test set."""
    transform_builder = TransformBuilder(input_size=(28, 28))
    transform = transform_builder.build_transform(train=False)

    test_dataset = datasets.EMNIST(
        root=data_dir,
        split="byclass",  # Choose 'byclass' or 'bymerge' based on your requirement
        train=False,
        download=True,
        transform=transform,
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    return test_dataloader


def test_model(
    model: torch.nn.Module, test_dataloader: DataLoader, device: torch.device
):
    """Test the model on the test dataset and return results."""
    model.eval()  # Set the model to evaluation mode
    model.to(device)  # Ensure model is on the correct device

    correct = 0
    total = 0
    test_loss = 0
    criterion = torch.nn.CrossEntropyLoss()  # Assuming the model uses CrossEntropyLoss

    # Define accuracy metric
    accuracy_metric = Accuracy(task="multiclass", num_classes=62).to(device)

    # Run inference on the test set
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader, 1):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(
                device
            )  # Move data to device

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            # Calculate accuracy
            accuracy_metric.update(outputs, targets)

            # Calculate correct predictions
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            avg_loss = test_loss / i
            accuracy = accuracy_metric.compute()

            # Print the current progress
            print(
                f"Test Loss: {avg_loss:.4f} | Test Accuracy: {accuracy:.4f}", end="\r"
            )

    return avg_loss, accuracy


def main(
    checkpoint_path: str, data_dir: str, batch_size: int = 256, num_workers: int = 4
):
    # Set device and config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config(data_dir=data_dir, batch_size=batch_size)

    # Load model
    model = load_model(checkpoint_path, config, device)

    # Get test data loader
    test_dataloader = get_test_dataloader(data_dir, batch_size, num_workers)

    # Test model
    avg_loss, accuracy = test_model(model, test_dataloader, device)

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the EMNIST model on a dataset.")
    parser.add_argument(
        "--checkpoint_path", type=str, help="Path to the checkpoint file."
    )
    parser.add_argument(
        "--data_dir", type=str, help="Path to the directory containing the test data."
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for testing."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for the DataLoader.",
    )

    args = parser.parse_args()

    main(args.checkpoint_path, args.data_dir, args.batch_size, args.num_workers)
