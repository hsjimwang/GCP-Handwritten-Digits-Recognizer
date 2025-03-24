import argparse
import os
import sys
import typing

import pytorch_lightning as pl
import torch
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(__file__))
from model import EMNIST_BLCLASS_LABELS, Config, LitEMNIST

DEFAULT_CONFIG = Config(
    data_dir="./data", max_epochs=1, batch_size=1, accelerator="auto", devices=1
)


def load_model(
    checkpoint_path: str = None,
    config: Config = DEFAULT_CONFIG,
    num_workers: int = 4,
    device: str = "cpu",
):
    """Load a trained model from a checkpoint.

    :param checkpoint_path: Path to the checkpoint file. If None, a default checkpoint will be used.
    :param config: Configuration object for the model. Defaults to DEFAULT_CONFIG.
    :param num_workers: Number of workers to use for data loading. Defaults to 4.
    :param device : Device to load the model on. Can be 'cpu' or 'cuda'. Defaults to 'cpu'.
    :return: The trained model loaded from the checkpoint, set to evaluation mode and moved to the specified device.
    """
    model = LitEMNIST.load_from_checkpoint(
        checkpoint_path, config=config, num_workers=num_workers
    )
    model.eval()  # Set model to evaluation mode
    model.to(device)  # Ensure the model is on the correct device
    return model


def preprocess_image(image_path: str) -> torch.Tensor:
    # Define image transformations to match training conditions
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),  # Resize to 28x28
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize((0.1307,), (0.3081,)),  # Normalize with MNIST mean/std
        ]
    )
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)  # Add batch dimension


def predict_digits(
    model: pl.LightningModule, input_image: Image.Image
) -> typing.Tuple[str, float]:
    """Predicts the digits from a given input image using the provided model.

    :param model: The trained PyTorch Lightning model used for prediction.
    :param input_image: The input image containing digits to be recognized.
                        It is assumed to be a 28x28n image.
    :return: A tuple containing:
            - str: The concatenated string of predicted digits/letters.
            - float: The average confidence of the predictions.
    """

    #  Assume input_image is a 28x28n image, first split it
    n = input_image.width // 28  # Assume the image width is 28x28n
    slices = [input_image.crop((i * 28, 0, (i + 1) * 28, 28)) for i in range(n)]

    # Define necessary preprocessing transformations
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),  # Resize to 28x28
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # Prediction results and confidence values
    predictions = []
    confidences = []  # Store the confidence values for each image

    for img in slices:
        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
        output = model(img_tensor)  # Assume the model output is logits
        _, predicted_label = torch.max(
            output, 1
        )  # Get the predicted class with the highest probability
        confidence = torch.nn.functional.softmax(output, dim=1)[0][
            predicted_label.item()
        ].item()  # Get the confidence for the predicted class
        predictions.append(
            EMNIST_BLCLASS_LABELS[predicted_label.item()]
        )  # Assuming the prediction is a letter, map the digit to a letter
        confidences.append(confidence)

    # Calculate the average confidence
    avg_confidence = sum(confidences) / len(confidences)

    # Concatenate all predicted letters into a string
    predicted_string = "".join(predictions)

    return predicted_string, avg_confidence


def main():
    parser = argparse.ArgumentParser(description="EMNIST Handwritten Digits Predictor.")
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to image for prediction"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )
    args = parser.parse_args()

    # Automatically use GPU if available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create a config matching the training setup
    config = Config(
        data_dir="./data", max_epochs=1, batch_size=1, accelerator="auto", devices=1
    )

    # Load model and preprocess image
    model = load_model(args.checkpoint_path, config, args.num_workers)

    # Perform prediction
    image_pil = Image.open(args.image_path)
    results = predict_digits(model, image_pil)

    print(f"Predicted digit: {results}")


if __name__ == "__main__":
    main()
