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
from typing import ClassVar

import pandas as pd
import torch
from dataset_define import CategoricDataset
from learning_model import CategoricNeuralNetwork, StopTraining
from torch.utils.data import DataLoader


logger: logging.Logger = logging.getLogger("TFM")
printer: logging.Logger = logging.getLogger("printer")


@dataclass
class ConfigRun:
    """
    A dataclass to store the configuration for the model training.

    Attributes:
        learning_rate (float): The learning rate for the model.
        batch_size (int): The batch size for the model.
        epochs (int): The number of epochs to train the model.
        f1_target (float): The target F1 score to stop the training.

    Methods:
        None

    """
    learning_rate: ClassVar[float] = 1e-3
    batch_size: ClassVar[int] = 64
    epochs: ClassVar[int] = 50
    f1_target: ClassVar[float] = 0.7


def configure_dataset(
        df: pd.DataFrame,
        input_columns: list,
        output_columns: list
        ) -> CategoricDataset:
    """
    Configure the dataset for training the model.

    Args:
        df (pd.DataFrame): The data to be used by the dataset.
        input_columns (list): The names of the columns to be used as input.
        output_columns (list): The names of the columns to be used as output.

    Returns:
        dataset (CategoricDataset): The configured dataset.
    """
    # statements to make sure the input is correct
    if not isinstance(input_columns, list):
        raise TypeError("input_columns must be a list")
    if not isinstance(output_columns, list):
        raise TypeError("output_columns must be a list")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if len(input_columns) == 0:
        raise ValueError("input_columns must have at least one element")
    if len(output_columns) != 1:
        raise ValueError("output_columns must have exactly one element")
    if any([col not in df.columns for col in input_columns]):
        raise ValueError("input_columns must be columns in the DataFrame")
    if any([col not in df.columns for col in output_columns]):
        raise ValueError("output_columns must be columns in the DataFrame")
    if any([col in output_columns for col in input_columns]):
        raise ValueError("input_columns and output_columns must have different elements")

    # create the dataset
    dataset: CategoricDataset = CategoricDataset(data=df)
    # define the input and output columns and create auxiliary variables
    dataset.define_input_output(input_columns=input_columns, output_columns=output_columns)
    return dataset


def get_dataloaders(dataset: CategoricDataset, batch_size: int = 32, train_size: float = 0.8) -> tuple:
    """
    Get the dataloaders for the training and testing of the model.

    Args:
        dataset (CategoricDataset): The dataset to be used.
        batch_size (int): The batch size to be used.

    Returns:
        train_dataloader (DataLoader): The DataLoader for the training dataset.
        test_dataloader (DataLoader or None): The DataLoader for the testing dataset.
    """
    train_dataset, test_dataset = dataset.train_test_split(train_size=train_size)
    train_dataloader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    if train_size == 1.0:
        logger.warning("train_size is 1.0, no test dataset will be created. Its DataLoader will be None.")
        return train_dataloader, None
    test_dataloader: DataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_dataloader, test_dataloader


def create_model(
        dataset: CategoricDataset,
        use_input_embedding: bool = True,
        use_output_embedding: bool = False,
        max_hidden_neurons: int = 512,
        hidden_layers: int = 1,
        train_nlp_embedding: bool = False
        ) -> CategoricNeuralNetwork:
    """
    Create an instance of the CategoricNeuralNetwork model.

    Args:
        dataset (CategoricDataset): The dataset to be used.
        use_input_embedding (bool): Whether to use an embedding layer for the input.
        use_output_embedding (bool): Whether to use an embedding layer for the output.
        max_hidden_neurons (int): The number of neurons in the hidden layers.
        hidden_layers (int): The number of hidden layers.
        train_nlp_embedding (bool): Whether to train the NLP embedding layer.


    Returns:
        model (CategoricNeuralNetwork): The model to be used.
    """
    assert dataset.already_configured, "The dataset must be configured before creating the model."
    model: CategoricNeuralNetwork = CategoricNeuralNetwork(
        category_mappings=dataset.category_mappings,
        use_input_embedding=use_input_embedding,
        use_output_embedding=use_output_embedding,
        max_hidden_neurons=max_hidden_neurons,
        hidden_layers=hidden_layers,
        train_nlp_embedding=train_nlp_embedding
        )
    model.to(model.device)
    return model


def train(model: CategoricNeuralNetwork,
          train_dataloader: DataLoader,
          test_dataloader: DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler._LRScheduler = None,
          epochs: int = 50,
          f1_target: float = None
          ) -> None:
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
        f1_target (float): The target F1 score to stop the training.
                           if None use the value in model.f1_target
                           If not none, overwrite the value in model.f1_target

    Returns:
        None
    """
    printer.debug(
        f"\nDetails on the training:"
        f"\nModel: {model}"
        f"\nLoss Function: {loss_fn}"
        f"\nOptimizer: {optimizer}"
        f"\nScheduler: {scheduler}"
        f"\nEpochs: {epochs}"
        f"\nf1_target = {f1_target}"
        f"\n\nConfig: {ConfigRun.learning_rate = } {ConfigRun.batch_size = } {ConfigRun.epochs = }"
        )
    try:
        model.f1_target = f1_target if f1_target is not None else model.f1_target
        for t in range(epochs):
            printer.info(f"\nEpoch {t+1}\n-------------------------------")
            model.train_loop(train_dataloader, loss_fn, optimizer)
            f1_score = model.test_loop(test_dataloader, loss_fn)
        if scheduler is not None:
            scheduler.step(f1_score)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except StopTraining:
        printer.info("Training Stopped")


if __name__ == '__main__':
    pass
