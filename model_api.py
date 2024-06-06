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
    # Model parameters
    max_hidden_neurons: ClassVar[int] = 2058
    hidden_layers: ClassVar[int] = 2
    model_uses_input_embedding: ClassVar[bool] = True
    model_uses_output_embedding: ClassVar[bool] = False
    model_trains_nlp_embedding: ClassVar[bool] = False
    # Training parameters
    learning_rate: ClassVar[float] = 1e-3
    batch_size: ClassVar[int] = 32
    epochs: ClassVar[int] = 15
    train_size: ClassVar[float] = 0.8
    # Optimizer
    optimizer_name: ClassVar[str] = "AdamW"
    nlp_model_name: ClassVar[str] = "huggingface/CodeBERTa-small-v1"
    # Scheduler: Learning rate multiplicative factor
    lr_decay_factor: ClassVar[float] = 0.1
    scheduler_type: ClassVar[str] = None
    # Scheduler: relevant only if Reduce on Plateau
    scheduler_plateau_mode: ClassVar[str] = "max"
    lr_decay_patience: ClassVar[int] = 4
    lr_decay_targets: ClassVar[dict] = {
        "f1": 0.75
    }
    # Scheduler: relevant only if reduce on step
    lr_decay_step: ClassVar[int] = 10


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
        train_size: float = None,
        batch_size: int = None,
        aggregate_outputs: bool = True
        ) -> tuple:
    """
    Get the dataloaders for the training and testing of the model.

    Args:
        dataset (CategoricDataset): The dataset to be used.
        train_size (float): The proportion of the dataset to be used for training.
                            If None, it will use the configuration in ConfigRun.
        batch_size (int): The batch size to be used.
                          If None, it will use the configuration in ConfigRun.

        aggregate_outputs (bool): Whether to aggregate the outputs.
                                  This is useful if some input appears multiple times
                                    with different outputs.

    Returns:
        train_dataloader (DataLoader): The DataLoader for the training dataset.
        test_dataloader (DataLoader or None): The DataLoader for the testing dataset.
    """
    if batch_size is None:
        batch_size = ConfigRun.batch_size
    if train_size is None:
        train_size = ConfigRun.train_size

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
        drop_last=False
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

    info_message = []
    info_message.append("Details on the model:")
    info_message.append(f"Dataset: Trying to relate{dataset.input_columns} to {dataset.output_columns}")
    info_message.append(f"use_input_embedding = {use_input_embedding}")
    info_message.append(f"use_output_embedding = {use_output_embedding}")
    info_message.append(f"max_hidden_neurons = {max_hidden_neurons}")
    info_message.append(f"hidden_layers = {hidden_layers}")
    info_message.append(f"train_nlp_embedding = {train_nlp_embedding}")
    info_message.append(f"nlp_model_name = {nlp_model_name}")

    printer.debug("\n".join(info_message))

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


def get_optimizer(
        model: CategoricNeuralNetwork,
        learning_rate: float = None,
        optimizer_name: str = "AdamW"
        ) -> torch.optim.Optimizer:
    """
    Get the optimizer for the model.

    Args:
        model (CategoricNeuralNetwork): The model to be used.
        learning_rate (float): The learning rate for the optimizer.
                                If None, it will use the configuration in ConfigRun.
        optimizer_name (str): The name of the optimizer to be used.
                                If None, it will use the configuration in ConfigRun.
                                Must be one of "AdamW", "SGD", or "Adam".

    Returns:
        optimizer (torch.optim.Optimizer): The optimizer to be used.
    """
    # check the input
    if learning_rate is None:
        learning_rate = ConfigRun.learning_rate
    if optimizer_name is None:
        optimizer_name = ConfigRun.optimizer_name

    # create the optimizer
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("optimizer_name must be one of 'AdamW', 'SGD', or 'Adam'.")
    return optimizer


def get_scheduler(
        optimizer: torch.optim.Optimizer,
        factor: float = 0.1,
        mode: str = None,
        patience: int = None,
        steps: int = None,
        scheduler_type: str = None
        ) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Get the scheduler for the optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to be used.
        factor (float): The factor for the scheduler.
        mode (str): The mode for the scheduler.
                        Relevant only if the scheduler is ReduceLROnPlateau.
        patience (int): The patience for the scheduler.
                        Relevant only if the scheduler is ReduceLROnPlateau.
        steps (int): The steps for the scheduler.
                        Relevant only if the scheduler is StepLR.
        scheduler_type (str): The type of scheduler to be used.
                                If None, it will use the configuration in ConfigRun.
                                Must be one of "StepLR" or "ReduceLROnPlateau".

    Returns:
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler to be used.
    """
    # check the input
    if scheduler_type is None:
        if ConfigRun.scheduler_type is not None:
            scheduler_type = ConfigRun.scheduler_type
        else:
            scheduler_type = "ReduceLROnPlateau"
            logger.debug("scheduler_type was not set. Using ReduceLROnPlateau.")
    if factor is None:
        factor = ConfigRun.lr_decay_factor
    # create the scheduler
    if scheduler_type == "ReduceLROnPlateau":
        if patience is None:
            patience = ConfigRun.lr_decay_patience
        if mode is None:
            mode = ConfigRun.scheduler_plateau_mode
        # if the scheduler is ReduceLROnPlateau, the steps will be ignored
        if steps is not None:
            logger.warning(
                "Steps is not used in ReduceLROnPlateau scheduler."
                " However it was set in the function call."
                " It will be ignored."
                )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode=mode,
            factor=factor,
            patience=patience
            )
    elif scheduler_type == "StepLR":
        if steps is None:
            steps = ConfigRun.lr_decay_step
        # if the scheduler is StepLR, the patience will be ignored
        if patience is not None:
            logger.warning(
                "Patience is not used in StepLR scheduler."
                " However it was set in the function call."
                " It will be ignored."
                )
        # if the scheduler is StepLR, the mode will be ignored
        if mode is not None:
            logger.warning(
                "Mode is not used in StepLR scheduler."
                " However it was set in the function call."
                " It will be ignored."
                )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=steps,
            gamma=factor
            )
    else:
        raise ValueError("scheduler_type must be one of 'StepLR' or 'ReduceLROnPlateau'.")
    return scheduler


def get_loss_fn(
        use_output_embedding: bool = None
        ) -> torch.nn.Module:
    """
    Get the loss function for the model.

    Args:
        use_output_embedding (bool): Whether to use an embedding layer for the output.
                                     If None, it will use the configuration in ConfigRun.
                                     This decides the loss function to be used.

    Returns:
        loss_fn (torch.nn.Module): The loss function to be used.
    """
    if use_output_embedding is None:
        use_output_embedding = ConfigRun.model_uses_output_embedding

    if use_output_embedding:
        loss_fn = torch.nn.CosineEmbeddingLoss()
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    return loss_fn


def train(
        model: CategoricNeuralNetwork,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        epochs: int = None,
        lr_decay_targets: dict = None
          ) -> None:
    """
    Train the model on the dataset.

    Args:
        Necessary:
            model (CategoricNeuralNetwork): The model to be trained.
            train_dataloader (DataLoader): The DataLoader for the training dataset.
            test_dataloader (DataLoader): The DataLoader for the testing dataset.
            loss_fn (torch.nn.Module): The loss function to be used.
            optimizer (torch.optim.Optimizer): The optimizer to be used.
        Optional (will default to the values in ConfigRun):
            scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler to be used.
            The following will default to the values in ConfigRun:
            epochs (int): The number of epochs to train the model.

    Returns:
        None
    """

    if epochs is None:
        epochs = ConfigRun.epochs

    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        if lr_decay_targets is None:
            lr_decay_targets = ConfigRun.lr_decay_targets

    f1_target = lr_decay_targets.get("f1", None)
    precision_target = lr_decay_targets.get("precision", None)
    recall_target = lr_decay_targets.get("recall", None)

    monitor_f1: bool = f1_target is not None
    monitor_precision: bool = precision_target is not None
    monitor_recall: bool = recall_target is not None

    info_message = []
    info_message.append("Details on the training:")
    info_message.append(f"Model: {model}")
    info_message.append(f"Loss Function: {loss_fn}")
    info_message.append(f"Optimizer: {optimizer}")
    info_message.append(f"Scheduler: {scheduler}")
    info_message.append(f"Epochs: {epochs}")
    if monitor_f1:
        info_message.append(f"f1_target = {f1_target}")
    if monitor_precision:
        info_message.append(f"precision_target = {precision_target}")
    if monitor_recall:
        info_message.append(f"recall_target = {recall_target}")

    printer.debug("\n".join(info_message))
    try:
        for t in range(epochs):
            printer.info(f"\nEpoch {t+1}\n-------------------------------")
            model.train_loop(train_dataloader, loss_fn, optimizer)
            f1_score, precision, recall = model.test_loop(test_dataloader, loss_fn)
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(f1_score)
                elif isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
                    scheduler.step()
                else:
                    raise ValueError("scheduler must be ReduceLROnPlateau or StepLR.")
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


def build_and_train_model(
        df: pd.DataFrame,
        input_columns: list,
        output_columns: list
        ) -> CategoricNeuralNetwork:
    """
    Build and train the model. Will make use of values in ConfigRun.
    They can be set before hand for a custom configuration.

    Args:
        dataset (pd.DataFrame): The dataset to be used.
        input_columns (list): The names of the columns to be used as input.
        output_columns (list): The names of the columns to be used as output.

    Returns:
        model (CategoricNeuralNetwork): The trained model.
    """
    dataset = configure_dataset(df, input_columns, output_columns)
    train_dataloader, test_dataloader = get_dataloaders(dataset)
    model = create_model(dataset)
    loss_fn = get_loss_fn()
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    train(model, train_dataloader, test_dataloader, loss_fn, optimizer, scheduler)
    return model

if __name__ == '__main__':
    pass
