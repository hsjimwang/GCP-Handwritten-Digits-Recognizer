import os
import typing
from dataclasses import dataclass
from typing import Tuple

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import EMNIST
from torchvision.transforms import InterpolationMode

# EMNIST ByClass label map
EMNIST_BLCLASS_LABELS = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
]


@dataclass
class Config:
    """Configuration options for the Lightning EMNIST example.

    :param data_dir: The path to the directory where the EMNIST dataset is stored. Defaults to the value of
            the 'PATH_DATASETS' environment variable or '.' if not set.
    :param save_dir: The path to the directory where the training logs will be saved. Defaults to 'logs/'.
    :param batch_size: The batch size to use during training. Defaults to 256 if a GPU is available,
            or 64 otherwise.
    :param max_epochs: The maximum number of epochs to train the model for. Defaults to 3.
    :param accelerator: The accelerator to use for training. Can be one of "cpu", "gpu", "tpu", "ipu", "auto".
    :param devices: The number of devices to use for training. Defaults to 1.

    Examples:
        This dataclass can be used to specify the configuration options for training a PyTorch Lightning model on the
        EMNIST dataset. A new instance of this dataclass can be created as follows:
        >>> config = Config()
        The default values for each argument are shown in the documentation above. If desired, any of these values can be
        overridden when creating a new instance of the dataclass:
        >>> config = Config(batch_size=128, max_epochs=5)
    """

    data_dir: str = os.environ.get("PATH_DATASETS", ".")
    save_dir: str = "logs/"
    batch_size: int = 256 if torch.cuda.is_available() else 64
    max_epochs: int = 3
    accelerator: str = "auto"
    devices: int = 1


class TransformBuilder:
    """Builder class for dataset transforms."""

    def __init__(
        self,
        input_size: typing.Tuple[int, int],
        interpolation: InterpolationMode = InterpolationMode.BICUBIC,
        antialias: bool = True,
    ):
        """Initializes a transform builder instance.

        :param input_size: The input size of the model. (h, w)
        :param interpolation: The interpolation mode for resizing images.
        :param antialias: Whether to use antialiasing when resizing images.
        """
        self.input_size = input_size
        self.interpolation = interpolation
        self.antialias = antialias

    def _build_train_transform(self) -> typing.Callable:
        """Builds a transform for training data.

        :return: The training transform as a :class:`typing.Callable`.
        """
        transform = transforms.Compose(
            [
                transforms.Lambda(
                    lambda img: transforms.functional.rotate(img, -90)
                ),  # rotate 90 degrees counterclockwise
                transforms.Lambda(
                    lambda img: transforms.functional.hflip(img)
                ),  # horizontal flip
                transforms.Resize((28, 28)),  # Resize to 28x28
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.RandomRotation(10),
                transforms.RandomAffine(10),
                transforms.RandomPerspective(
                    distortion_scale=0.2, p=0.5, interpolation=2
                ),
                # transforms.RandomApply(
                #     [transforms.RandomCrop(28, padding=4)], p=0.5
                # ),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                # transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
                # transforms.GaussianBlur(23),
                # transforms.RandomAdjustSharpness(sharpness_factor=2)
            ]
        )
        return transform

    def _build_val_transform(self) -> typing.Callable:
        """Builds a transform for validation data.

        :return: The validation transform as a :class:`typing.Callable`.
        """
        transform = transforms.Compose(
            [
                transforms.Lambda(
                    lambda img: transforms.functional.rotate(img, -90)
                ),  # rotate 90 degrees counterclockwise
                transforms.Lambda(
                    lambda img: transforms.functional.hflip(img)
                ),  # horizontal flip
                transforms.Resize((28, 28)),  # Resize to 28x28
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        return transform

    def build_transform(self, train: bool = False) -> typing.Callable:
        """Builds a transform for training or validation data.

        :param train: Whether to build a transform for training data.
        :return: The transform as a :class:`typing.Callable`.
        """
        return self._build_train_transform() if train else self._build_val_transform()


class SimpleMLP(nn.Module):
    """A simple multi-layer perceptron (MLP) model for image classification."""

    def __init__(
        self, channels: int, width: int, height: int, hidden_size: int, num_classes: int
    ):
        """Initializes the SimpleMLP model.

        :param channels: Number of input channels.
        :param width: Width of the input image.
        :param height: Height of the input image.
        :param hidden_size: Number of neurons in the hidden layers.
        :param num_classes: Number of output classes.
        """
        super(SimpleMLP, self).__init__()

        # Define the sequential layers
        self.model = nn.Sequential(
            # Flatten input tensor
            nn.Flatten(),
            # Fully connected layer 1: input size = channels * width * height, output size = hidden_size
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            # Fully connected layer 2: input size = hidden_size, output size = hidden_size
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            # Output layer: input size = hidden_size, output size = num_classes
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)


class SimpleCNN(nn.Module):
    """A simple convolutional neural network (CNN) model for image classification."""

    def __init__(self, num_classes: int):
        """Initializes the SimpleCNN model."""
        super(SimpleCNN, self).__init__()

        # First convolutional layer: input channels = 1 (grayscale image), output channels = 32, kernel size = 3x3, stride = 1, padding = 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)

        # Second convolutional layer: input channels = 32, output channels = 64, kernel size = 3x3, stride = 1, padding = 1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Max pooling layer: kernel size = 2x2, stride = 2
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers after flattening
        # Calculate the flattened size after convolution and pooling: 28x28 -> 14x14 -> 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 128 hidden neurons
        self.fc2 = nn.Linear(
            128, num_classes
        )  # Output layer with 62 classes (for EMNIST)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        # Apply convolution and ReLU activation for the first layer
        x = F.relu(self.conv1(x))

        # Apply max pooling after the first convolution
        x = self.pool(x)

        # Apply convolution and ReLU activation for the second layer
        x = F.relu(self.conv2(x))

        # Apply max pooling after the second convolution
        x = self.pool(x)

        # Flatten the output from 2D to 1D for the fully connected layers
        x = x.view(-1, 64 * 7 * 7)  # Flatten the output to a 1D vector

        # Apply fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))

        # Apply the final output layer
        x = self.fc2(x)

        return x


class LitEMNIST(pl.LightningModule):
    """A PyTorch Lightning Module for training and evaluating a model on the EMNIST dataset."""

    def __init__(
        self,
        config: Config,
        hidden_size: int = 64,
        learning_rate: float = 2e-4,
        num_workers: int = 4,
    ):
        """Initializes the model with the given configuration.

        :param config: Configuration object containing various settings.
        :param hidden_size: Size of the hidden layer, defaults to 64.
        :param learning_rate: Learning rate for the optimizer, defaults to 0.0002.
        :param num_workers: Number of workers for data loading, defaults to 4.
        """
        super().__init__()

        self.config = config
        self.data_dir = config.data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.num_workers = num_workers

        self.num_classes = 62
        self.dims = (1, 28, 28)

        # Initialize TransformBuilder with the desired input size
        self.transform_builder = TransformBuilder(input_size=(28, 28))

        # Intialize the model instance
        self.model = SimpleMLP(
            channels=1,
            width=28,
            height=28,
            hidden_size=128,
            num_classes=self.num_classes,
        )
        self.model = SimpleCNN(self.num_classes)

        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model and apply log softmax activation.

        :param x: Input tensor to the model.
        :return: Output tensor after applying the model and log softmax activation.
        """
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Perform a single training step.

        :param batch: A tuple containing the input tensor and the target tensor.
        :return: The computed loss for the batch.
        """
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Perform a single validation step.

        :param batch: A tuple containing the input tensor and the target tensor.
        :return: None.
        """

        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Perform a single test step.

        :param batch: A tuple containing the input data (x) and the target labels (y).
        :return: None.
        """

        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer to use for training."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    # ------------------------------------- #
    # DATA RELATED HOOKS
    # ------------------------------------- #

    def setup(self, stage: str = None) -> None:
        """Set up the data for training, validation, and testing."""
        if stage == "fit" or stage is None:
            emnist_full = EMNIST(
                self.data_dir,
                train=True,
                transform=self.transform_builder.build_transform(train=True),
                split="byclass",
                download=True,
            )
            print(len(emnist_full))
            dataset_size = len(emnist_full)

            # Split the dataset into 85% training, 15% validation
            train_size = int(dataset_size * 0.85)
            val_size = dataset_size - train_size

            self.emnist_train, self.emnist_val = random_split(
                emnist_full, [train_size, val_size]
            )

        if stage == "test" or stage is None:
            self.emnist_test = EMNIST(
                self.data_dir,
                train=False,
                transform=self.transform_builder.build_transform(train=False),
                split="byclass",
                download=True,
            )

    def train_dataloader(self) -> DataLoader:
        """Create the data loader for training data."""
        return DataLoader(
            self.emnist_train,
            batch_size=self.config.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Create the data loader for validation data."""
        return DataLoader(
            self.emnist_val,
            batch_size=self.config.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Create the data loader for test data."""
        return DataLoader(
            self.emnist_test,
            batch_size=self.config.batch_size,
            num_workers=self.num_workers,
        )
