import argparse

import pytorch_lightning as pl
from model import Config, LitEMNIST
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint


def train_digit_recognizer(
    data_dir: str,
    checkpoint_dir: str,
    max_epochs: int = 30,
    batch_size: int = 256,
    num_workers: int = 4,
):
    # Init our Config
    config = Config(
        data_dir=data_dir,
        max_epochs=max_epochs,
        batch_size=batch_size,
        accelerator="auto",
        devices=1,
    )

    # Init our model
    emnist_model = LitEMNIST(config=config, num_workers=num_workers)

    # Init checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch={epoch:02d}-val_loss={val_loss:.2f}-val_acc={val_acc:.3f}",
        monitor="val_loss",
        save_top_k=3,  # Save all checkpoints
        save_last=True,  # Save the last checkpoint
    )

    # Initialize a trainer
    trainer = pl.Trainer(
        logger=loggers.TensorBoardLogger(checkpoint_dir, name="logs"),
        accelerator=config.accelerator,
        devices=config.devices,
        max_epochs=config.max_epochs,
        callbacks=[checkpoint_callback],
    )

    # Train the model
    trainer.fit(emnist_model)
    trainer.test(ckpt_path="best")


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        default="./data",
        help="Directory to save the MNIST dataset",
    )
    argparser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory to save model checkpoints",
    )
    argparser.add_argument(
        "--max_epochs", type=int, default=30, help="Number of epochs to train the model"
    )
    argparser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for training"
    )
    argparser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )
    args = argparser.parse_args()

    train_digit_recognizer(
        args.data_dir,
        args.checkpoint_dir,
        args.max_epochs,
        args.batch_size,
        args.num_workers,
    )


if __name__ == "__main__":
    main()
