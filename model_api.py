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

If you want to see the documentation of the function or class,
please, scroll down to the end of this file.

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
    A dataclass to store the default configuration for the model training.

    Attributes:
        learning_rate (float): The learning rate for the model.
        batch_size (int): The batch size for the model.
        epochs (int): The number of epochs to train the model.
        f1_target (float): The target F1 score to stop the training.
        max_hidden_neurons (int): The maximum number of hidden neurons.
        hidden_layers (int): The number of hidden layers.

    Methods:
        None

    """
    learning_rate: ClassVar[float] = 1e-3
    batch_size: ClassVar[int] = 64
    epochs: ClassVar[int] = 50
    f1_target: ClassVar[float] = 0.7
    precision_target: ClassVar[float] = 0.7
    recall_target: ClassVar[float] = 0.7
    max_hidden_neurons: ClassVar[int] = 2058
    hidden_layers: ClassVar[int] = 2
    monitor_f1: ClassVar[bool] = True
    monitor_precision: ClassVar[bool] = False
    monitor_recall: ClassVar[bool] = False
    model_uses_input_embedding: ClassVar[bool] = True
    model_uses_output_embedding: ClassVar[bool] = False
    model_trains_nlp_embedding: ClassVar[bool] = False
    nlp_model_name: ClassVar[str] = "distilbert-base-uncased"


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
    dataset.define_input_output(input_columns=input_columns, output_columns=output_columns, store_input_features=False)
    return dataset


def get_dataloaders(
        dataset: CategoricDataset,
        train_size: float = 0.8,
        batch_size: int = None,
        aggregate_outputs: bool = False
        ) -> tuple:
    """
    Get the dataloaders for the training and testing of the model.

    Args:
        dataset (CategoricDataset): The dataset to be used.
        train_size (float): The proportion of the dataset to be used for training.
        batch_size (int): The batch size to be used.
        aggregate_outputs (bool): Whether to aggregate the outputs.
                                  This is useful if some input appears multiple times
                                    with different outputs.

    Returns:
        train_dataloader (DataLoader): The DataLoader for the training dataset.
        test_dataloader (DataLoader or None): The DataLoader for the testing dataset.
    """
    if batch_size is None:
        batch_size = ConfigRun.batch_size

    train_dataset, test_dataset = dataset.train_test_split(train_size=train_size)
    train_dataloader: DataLoader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
        )

    if aggregate_outputs:
        test_dataset.group_by()
    test_dataloader: DataLoader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
        )
    return train_dataloader, test_dataloader


def create_model(
        dataset: CategoricDataset,
        use_input_embedding: bool = None,
        use_output_embedding: bool = None,
        max_hidden_neurons: int = None,
        hidden_layers: int = None,
        train_nlp_embedding: bool = None,
        nlp_model_name: str = None
        ) -> CategoricNeuralNetwork:
    """
    Create an instance of the CategoricNeuralNetwork model.

    Args:
        dataset (CategoricDataset): The dataset to be used.
        If None, it will use the configuration in ConfigRun.
        use_input_embedding (bool): Whether to use an embedding layer for the input.
        use_output_embedding (bool): Whether to use an embedding layer for the output.
        max_hidden_neurons (int): The number of neurons in the hidden layers.
                                  If None, it will use the configuration in ConfigRun.
        hidden_layers (int): The number of hidden layers.
                                If None, it will use the configuration in ConfigRun.
        train_nlp_embedding (bool): Whether to train the NLP embedding layer.
        nlp_model_name (str): The name of the NLP model to be used.

    Returns:
        model (CategoricNeuralNetwork): The model to be used.
    """
    assert dataset.already_configured, "The dataset must be configured before creating the model."
    if use_input_embedding is None:
        use_input_embedding = ConfigRun.model_uses_input_embedding
    if use_output_embedding is None:
        use_output_embedding = ConfigRun.model_uses_output_embedding
    if train_nlp_embedding is None:
        train_nlp_embedding = ConfigRun.model_trains_nlp_embedding
    if nlp_model_name is None:
        nlp_model_name = ConfigRun.nlp_model_name

    if max_hidden_neurons is None:
        max_hidden_neurons = ConfigRun.max_hidden_neurons
    if hidden_layers is None:
        hidden_layers = ConfigRun.hidden_layers

    printer.debug(
        f"\nDetails on the model:"
        f"\nDataset: {dataset}"
        f"\nuse_input_embedding = {use_input_embedding}"
        f"\nuse_output_embedding = {use_output_embedding}"
        f"\nmax_hidden_neurons = {max_hidden_neurons}"
        f"\nhidden_layers = {hidden_layers}"
        f"\ntrain_nlp_embedding = {train_nlp_embedding}"
        f"\nnlp_model_name = {nlp_model_name}"
        )
    model: CategoricNeuralNetwork = CategoricNeuralNetwork(
        category_mappings=dataset.category_mappings,
        use_input_embedding=use_input_embedding,
        use_output_embedding=use_output_embedding,
        max_hidden_neurons=max_hidden_neurons,
        hidden_layers=hidden_layers,
        train_nlp_embedding=train_nlp_embedding,
        nlp_model_name=nlp_model_name
        )
    model.to(model.device)
    return model


def train(
        model: CategoricNeuralNetwork,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        f1_target: float = None,
        precision_target: float = None,
        recall_target: float = None,
        epochs: int = None,
        monitor_f1: bool = None,
        monitor_precision: bool = None,
        monitor_recall: bool = None
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
        The following will default to the values in ConfigRun:
        f1_target (float): The target F1 score to stop the training.
        precision_target (float): The target precision to stop the training.
        recall_target (float): The target recall to stop the training.
        epochs (int): The number of epochs to train the model.
        monitor_f1 (bool): Whether to monitor the F1 score.
        monitor_precision (bool): Whether to monitor the precision.
        monitor_recall (bool): Whether to monitor the recall.

    Returns:
        None
    """

    if f1_target is None:
        f1_target = ConfigRun.f1_target
    if precision_target is None:
        precision_target = ConfigRun.precision_target
    if recall_target is None:
        recall_target = ConfigRun.recall_target
    if epochs is None:
        epochs = ConfigRun.epochs
    if monitor_f1 is None:
        monitor_f1 = ConfigRun.monitor_f1
    if monitor_precision is None:
        monitor_precision = ConfigRun.monitor_precision
    if monitor_recall is None:
        monitor_recall = ConfigRun.monitor_recall

    printer.debug("".join((
        "\nDetails on the training:",
        f"\nModel: {model}",
        f"\nLoss Function: {loss_fn}",
        f"\nOptimizer: {optimizer}",
        f"\nScheduler: {scheduler}",
        f"\nEpochs: {epochs}",
        f"\nf1_target = {f1_target}" if monitor_f1 else "",
        f"\nprecision_target = {precision_target}" if monitor_precision else "",
        f"\nrecall_target = {recall_target}" if monitor_recall else "",
        )
        ))
    try:
        for t in range(epochs):
            printer.info(f"\nEpoch {t+1}\n-------------------------------")
            model.train_loop(train_dataloader, loss_fn, optimizer)
            f1_score, precision, recall = model.test_loop(test_dataloader, loss_fn)
            if scheduler is not None:
                scheduler.step(f1_score)
            if all([
                f1_score > f1_target if monitor_f1 else True,
                precision > precision_target if monitor_precision else True,
                recall > recall_target if monitor_recall else True
            ]):
                raise StopTraining("Target reached.")
    except StopTraining as e:
        printer.info(e)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")


if __name__ == '__main__':
    pass
