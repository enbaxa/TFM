"""
This module contains the API for the training of the model.

This module contains the following functions:
    * configure_dataset - Configure the dataset for training the model.
    * get_dataloaders - Get the dataloaders for the training and testing of the model.
    * train - Train the model on the dataset.

This module also contains the following dataclass:
    * ConfigRun - A dataclass to store the configuration for the model training.

This module requires the following modules:
    * pandas
    * torch
    * logging
    * dataclasses
    * dataset_define
    * learning_model
    * set_logger
    * torch.utils.data

This module contains the following classes:
    * ConfigRun

This module contains the following functions:
    * configure_dataset
    * get_dataloaders
    * train

If you want to see the documentation of the function or class, please, scroll down to the end of this file.

"""

import logging
from dataclasses import dataclass

import pandas as pd
import torch
from dataset_define import CategoricDataset
from learning_model import CategoricNeuralNetwork
from torch.utils.data import DataLoader


logger = logging.getLogger("TFM")
printer = logging.getLogger("printer")


@dataclass
class ConfigRun:
    """
    A dataclass to store the configuration for the model training.

    Attributes:
        learning_rate (float): The learning rate for the model.
        batch_size (int): The batch size for the model.
        epochs (int): The number of epochs to train the model.

    Methods:
        None

    """
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 50


def configure_dataset(df: pd.DataFrame, input_columns: list, output_column: list) -> CategoricDataset:
    """
    Configure the dataset for training the model.

    Args:
        df (pd.DataFrame): The data to be used by the dataset.
        input_columns (list): The names of the columns to be used as input.
        output_column (list): The names of the columns to be used as output.

    Returns:
        dataset (CategoricDataset): The configured dataset.
    """
    # statements to make sure the input is correct
    if not isinstance(input_columns, list):
        raise TypeError("input_columns must be a list")
    if not isinstance(output_column, list):
        raise TypeError("output_column must be a list")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if len(input_columns) == 0:
        raise ValueError("input_columns must have at least one element")
    if len(output_column) != 1:
        raise ValueError("output_column must have exactly one element")

    dataset = CategoricDataset(data=df)
    dataset.define_input_output(input_columns=input_columns, output_column=output_column)
    return dataset


def get_dataloaders(dataset: CategoricDataset, batch_size: int = 32, train_size: float = 0.8) -> tuple:
    """
    Get the dataloaders for the training and testing of the model.

    Args:
        dataset (CategoricDataset): The dataset to be used.
        batch_size (int): The batch size to be used.

    Returns:
        train_dataloader (DataLoader): The DataLoader for the training dataset.
        test_dataloader (DataLoader): The DataLoader for the testing dataset.
    """
    train_dataset, test_dataset = dataset.train_test_split(train_size=train_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if train_size == 1.0:
        logger.warning("train_size is 1.0, no test dataset will be created. Its DataLoader will be None.")
        return train_dataloader, None
    else:
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        return train_dataloader, test_dataloader


def train(model: CategoricNeuralNetwork,
          train_dataloader: DataLoader,
          test_dataloader: DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler._LRScheduler = None,
          epochs: int = 50) -> None:
    """
    Train the model on the dataset.

    Args:
        model (CategoricNeuralNetwork): The model to be trained.
        train_dataloader (DataLoader): The DataLoader for the training dataset.
        test_dataloader (DataLoader): The DataLoader for the testing dataset.
        loss_fn (torch.nn.Module): The loss function to be used.
        optimizer (torch.optim.Optimizer): The optimizer to be used.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler to be used.
        epochs (int): The number of epochs to train the model.

    Returns:
        None
    """
    try:
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            model.train_loop(train_dataloader, loss_fn, optimizer, scheduler)
            model.test_loop(test_dataloader, loss_fn)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    finally:
        logger.info("Training stopped.")


if __name__ == '__main__':
    pass
