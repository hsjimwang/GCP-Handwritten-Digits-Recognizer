import argparse
import os
import random
import struct
import typing

import numpy as np
from model import EMNIST_BLCLASS_LABELS
from PIL import Image
from torchvision import transforms


# Function to read IDX files
def read_idx(filename: str) -> np.ndarray:
    """Reads an IDX file and returns its contents as a NumPy array.

    :param filename: The path to the IDX file to be read.
    :return: A NumPy array containing the data from the IDX file. The shape of the array
        depends on the type of data:
        - For image data (magic number 2051), the array has shape (num_items, rows, cols).
        - For label data (magic number 2049), the array has shape (num_items,).
    """

    with open(filename, "rb") as f:
        # Read header information
        (magic_number,) = struct.unpack(">I", f.read(4))
        if magic_number == 2051:  # Magic number for image data
            num_items, rows, cols = struct.unpack(">III", f.read(12))
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(
                num_items, rows, cols
            )
        elif magic_number == 2049:  # Magic number for label data
            (num_items,) = struct.unpack(">I", f.read(4))
            data = np.frombuffer(f.read(), dtype=np.uint8)
        else:
            raise ValueError("Unsupported IDX file format.")
    return data


# Function to generate concatenated images
def generate_concatenated_images(
    images: typing.List[np.ndarray],
    labels: typing.List[np.ndarray],
    output_dir: str = None,
    count: int = 10,
):
    """Generate concatenated images from a list of images and labels, apply transformations, and save the results.
    This function randomly selects a number of images and their corresponding labels, applies a series of transformations
    (rotation and horizontal flip), concatenates the transformed images horizontally, and saves the resulting image to the
    specified output directory.

    :param images: List of images as numpy arrays.
    :param labels: List of labels corresponding to the images.
    :param output_dir: Directory where the concatenated images will be saved. If None, the current directory is used.
    :param count: Number of concatenated images to generate, defaults to 10.
    :return: None
    """

    os.makedirs(output_dir, exist_ok=True)

    # Define transformations (rotate 90 degrees counterclockwise and horizontal flip)
    transform = transforms.Compose(
        [
            transforms.Lambda(
                lambda img: transforms.functional.rotate(img, -90)
            ),  # Rotate counterclockwise by 90 degrees
            transforms.Lambda(
                lambda img: transforms.functional.hflip(img)
            ),  # Horizontal flip
        ]
    )

    for i in range(count):
        # Randomly select the number of letters (between 2 and 8)
        num_letters = random.randint(2, 8)

        # Randomly choose letters and their corresponding images
        indices = random.sample(range(len(labels)), num_letters)
        letters = [
            EMNIST_BLCLASS_LABELS[label] for label in labels[indices]
        ]  # 'A' -> 65 is the ASCII code for 'A'
        selected_images = [images[idx] for idx in indices]

        # Apply transformations to images
        transformed_images = [
            transform(Image.fromarray(img)) for img in selected_images
        ]

        # Concatenate images
        total_width = sum(img.width for img in transformed_images)
        height = transformed_images[0].height
        result_img = Image.new("L", (total_width, height))

        current_x = 0
        for img in transformed_images:
            result_img.paste(img, (current_x, 0))
            current_x += img.width

        # Save the result
        output_path = os.path.join(output_dir, f"{''.join(letters)}.png")
        result_img.save(output_path)
        print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate concatenated letter images from EMNIST."
    )
    parser.add_argument(
        "--image_file",
        type=str,
        required=True,
        help="Path to the EMNIST training image file (e.g., emnist-byclass-train-images-idx3-ubyte).",
    )
    parser.add_argument(
        "--label_file",
        type=str,
        required=True,
        help="Path to the EMNIST training label file (e.g., emnist-byclass-train-labels-idx1-ubyte).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save concatenated images.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of concatenated images to generate.",
    )

    args = parser.parse_args()

    # Read images and labels
    images = read_idx(args.image_file)
    labels = read_idx(args.label_file)

    # Generate concatenated images
    generate_concatenated_images(images, labels, args.output_dir, args.count)


if __name__ == "__main__":
    main()
