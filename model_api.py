"""
This module provides classes and functions for creating, training and configuring a categoric neural network model.

Classes:
    - ConfigRun: A dataclass to store the default configuration for the model training.

Functions:
    - configure_default_loggers(fil_name: str = None, fil_dir: str = None): Configure the default loggers for the module.
    - wipe_handlers(): Wipe the loggers from the root logger.
    - reconfigure_loggers(): Reconfigure the loggers.
    - configure_dataset(df: pd.DataFrame, input_columns: list, output_columns: list) -> CategoricDataset: Configure the dataset for training the model.
    - get_dataloaders(dataset: CategoricDataset, train_size: float = None, batch_size: int = None, balance_training_data: bool = True, aggregate_outputs: bool = True) -> tuple: Get the dataloaders for the training and testing of the model.
    - create_model(dataset: CategoricDataset, use_input_embedding: bool = None, use_output_embedding: bool = None, max_hidden_neurons: int = None, hidden_layers: int = None, train_nlp_embedding: bool = None, nlp_model_name: str = None) -> CategoricNeuralNetwork
"""

import logging
import json
import time
import dataclasses
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from dataset_define import CategoricDataset
from learning_model import CategoricNeuralNetwork, StopTraining
import set_logger
logger: logging.Logger = logging.getLogger("TFM")
printer: logging.Logger = logging.getLogger("printer")
matplotlib.use("Agg")


@dataclasses.dataclass
class ConfigRun:
    """
    A dataclass to store the default configuration for the model training.

    Attributes:
        balance_training_data (bool): Whether to balance the training data.
        aggregate_outputs (bool): Whether to aggregate the outputs.
        max_hidden_neurons (int): The maximum number of hidden neurons in the model.
        hidden_layers (int): The number of hidden layers in the model.
        model_uses_input_embedding (bool): Whether the model uses input embedding.
        model_uses_output_embedding (bool): Whether the model uses output embedding.
        model_trains_nlp_embedding (bool): Whether the model trains the NLP embedding.
        learning_rate (float): The learning rate for the model.
        batch_size (int): The batch size for the model.
        epochs (int): The number of epochs for the model.
        train_size (float): The proportion of the dataset to be used for training.
        train_targets (dict): The targets for the training.
        optimizer_name (str): The name of the optimizer to be used.
        nlp_model_name (str): The name of the NLP model to be used.
        lr_decay_factor (float): The factor by which the learning rate will be decayed.
        scheduler_type (str): The type of the scheduler.
        scheduler_plateau_mode (str): The mode for the scheduler.
        lr_decay_patience (int): The patience for the scheduler.
        lr_decay_target (str): The target for the learning rate decay.
        lr_decay_step (int): The steps for the scheduler.
        out_dir (Path): The directory to save the reports and outputs.
        case_name (str): The name of the case.
        report_dir (str): The directory to save the reports.
        report_filename (str): The name of the report file.

    Methods:
        None

    """
    # Preprocessing parameters
    balance_training_data: bool = True
    aggregate_outputs: bool = True
    # Model parameters
    max_hidden_neurons: int = 2058
    hidden_layers: int = 2
    model_uses_input_embedding: bool = True
    model_uses_output_embedding: bool = False
    model_trains_nlp_embedding: bool = False
    # Training parameters
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 15
    train_size: float = 0.8
    train_targets: dict = dataclasses.field(default_factory=lambda: {"f1": 0.75})
    # Optimizer
    optimizer_name: str = "AdamW"
    nlp_model_name: str = "huggingface/CodeBERTa-small-v1"
    # Scheduler: Learning rate multiplicative factor
    lr_decay_factor: float = 0.1
    scheduler_type: str = None
    # Scheduler: relevant only if Reduce on Plateau
    scheduler_plateau_mode: str = "max"
    lr_decay_patience: int = 10
    lr_decay_target: str = "f1"
    # Scheduler: relevant only if reduce on step
    lr_decay_step: int = 10
    # Reports and outputs
    out_dir: Path = Path("out").resolve()
    case_name: str = "default"
    report_dir_name: str = "reports"
    report_filename: str = "report"

    @property
    def out_case_dir(self):
        return self.out_dir.joinpath(self.case_name)

    @property
    def report_dir(self):
        return self.out_case_dir.joinpath(self.report_dir_name)

    def print(self):
        """
        Print all config values
        """
        out = []
        for key, value in self.__dict__.items():
            if not key.startswith("__") and key != "print" and key != "out_case_dir" and key != "report_dir":
                out.append(f"{key}: {value}")
        return "\n".join(out)

    def to_dict(self):
        """
        Convert the ConfigRun instance to a dictionary, converting Paths to strings.
        """
        return {field.name: (str(getattr(self, field.name)) if isinstance(getattr(self, field.name), Path) else getattr(self, field.name))
                for field in dataclasses.fields(self)}

    @classmethod
    def from_dict(cls, dict_):
        """
        Create a ConfigRun instance from a dictionary, converting strings to Paths where necessary.
        """
        dict_['out_dir'] = Path(dict_['out_dir'])
        return cls(**dict_)


class ModelApi:
    """
    A class to store the functions for creating, training and configuring a categoric neural network model.

    Attributes:
        None

    Methods:
        configure_default_loggers(fil_name: str = None, fil_dir: str = None): Configure the default loggers for the module.
        wipe_handlers(): Wipe the loggers from the root logger.
        reconfigure_loggers(): Reconfigure the loggers.
        configure_dataset(df: pd.DataFrame, input_columns: list, output_columns: list) -> CategoricDataset: Configure the dataset for training the model.
        get_dataloaders(dataset: CategoricDataset, train_size: float = None, batch_size: int = None, balance_training_data: bool = True, aggregate_outputs: bool = True) -> tuple: Get the dataloaders for the training and testing of the model.
        create_model(dataset: CategoricDataset, use_input_embedding: bool = None, use_output_embedding: bool = None, max_hidden_neurons: int = None, hidden_layers: int = None, train_nlp_embedding: bool = None, nlp_model_name: str = None) -> CategoricNeuralNetwork
        get_optimizer(model: CategoricNeuralNetwork, learning_rate: float = None, optimizer_name: str = "AdamW") -> torch.optim.Optimizer
        get_scheduler(optimizer: torch.optim.Optimizer, factor: float = 0.1, mode: str = None, patience: int = None, steps: int = None, scheduler_type: str = None) -> torch.optim.lr_scheduler._LRScheduler
        get_loss_fn(use_output_embedding: bool = None) -> torch.nn.Module
        train(model: CategoricNeuralNetwork, train_dataloader: DataLoader, test_dataloader: DataLoader, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, train_targets: dict = None, epochs: int = None, lr_decay_target: str = None, scheduler: torch.optim.lr_scheduler._LRScheduler | None = None, do_report: bool = True, live_report: bool = False) -> None
        build_and_train_model(df: pd.DataFrame, input_columns: list, output_columns: list) -> CategoricNeuralNetwork
        save_model(model: CategoricNeuralNetwork, path: str) -> None
        load_model(path: str) -> CategoricNeuralNetwork
        add_output_posssibilities(df: pd.DataFrame,  output_category: str, output_value: str) -> pd.DataFrame
        """

    def __init__(self):
        self.config = ConfigRun()

    def configure_default_loggers(self, fil_name: str = None) -> None:
        """
        Configure the default loggers for the module.

        Those are loggers named "TFM" and "printer".
        They will be handled by the root logger, as they emit messages
        from throughout the application. The user can use them to
        print messages to the console or to save them in a log file, and
        the logging hierarchy will be respected.

        Args:
            fil_name (str): The name of the file to save the logs.

        Returns:
            None
        """
        if fil_name is None:
            fil_name: str = self.config.report_filename
            fil_path: Path = self.config.report_dir.joinpath(fil_name)
        else:
            printer.info("Path for log files was given. Overriding default")
        if not fil_path.parent.exists():
            fil_path.parent.mkdir(parents=True)
        # attach handlers to root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.NOTSET)
        # create file handler which will log messages to screen.
        # in principle those are messages that come from logger "printer"
        screen_handler = set_logger.PrinterScreenHandler()
        screen_handler.addFilter(set_logger.Filter_name("printer"))
        screen_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(screen_handler)
        # Add a second screen handler to root logger to print out warning+ messages
        screen_handler_high = set_logger.ColoredDetailedScreenHandler()
        screen_handler_high.setLevel(logging.WARNING)
        root_logger.addHandler(screen_handler_high)
        # create file handler which will log messages to file.
        # in principle those are messages that come from logger "logger"
        file_handler = set_logger.DetailedFileHandler(f"{fil_path}.log", mode="w")
        # Put everything both logs in the file
        file_handler.addFilter(set_logger.Filter_name(["TFM", "printer"]))
        file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        # create file handler which will log all messages to file in json format.
        json_handler = set_logger.PersistentLogHandler(f"{fil_path}.jsonl", mode="w")
        json_handler.setLevel(logging.DEBUG)
        # No filter is needed because it will log all messages
        root_logger.addHandler(json_handler)
        logger.debug("Loggers configured.")
        logger.debug("Log and report files will be saved in %s", fil_path)

    def wipe_handlers(self) -> None:
        """
        Wipe the loggers from the root logger.

        This eliminates all handlers from the root logger.

        Returns:
            None
        """
        root_logger = logging.getLogger()
        logger.debug("Wiping all handlers.")
        handlers = list(root_logger.handlers)
        for handler in handlers:
            root_logger.removeHandler(handler)

    def reconfigure_loggers(self) -> None:
        """
        Reconfigure the loggers.

        This function will wipe the handlers from the root logger and then reconfigure them.
        This assures that the configuration is updated by the user, otherwise
        it will use the default configuration.

        Returns:
            None
        """
        self.wipe_handlers()
        self.configure_default_loggers()
        logger.debug("Loggers reconfigured.")
        logger.debug(
            "All handlers wiped and reconfigured."
            "This includes any handlers added by the user!"
            )

    def configure_dataset(
            self,
            df: pd.DataFrame,
            input_columns: list,
            output_columns: list
            ) -> CategoricDataset:
        """
        Configure the dataset for training the model.

        Args:
            df (pd.DataFrame): The DataFrame to be used.
            input_columns (list): The columns to be used as input.
            output_columns (list): The columns to be used as output.

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
        logger.info("Dataset configured.")
        return dataset

    def get_dataloaders(
            self,
            dataset: CategoricDataset,
            train_size: float = None,
            batch_size: int = None,
            balance_training_data: bool = None,
            aggregate_outputs: bool = None
            ) -> tuple:
        """
        Get the dataloaders for the training and testing of the model.

        Args:
            dataset (CategoricDataset): The dataset to be used.
            train_size (float): The proportion of the dataset to be used for training.
                                If None, it will use the configuration in ConfigRun.
            batch_size (int): The batch size to be used.
                            If None, it will use the configuration in ConfigRun.

            balance_training_data (bool): Whether to balance the training data.
                                        This is done by oversampling the minority classes.

            aggregate_outputs (bool): Whether to aggregate the outputs.
                                    This is useful if some input appears multiple times
                                        with different outputs.

        Returns:
            train_dataloader (DataLoader): The DataLoader for the training dataset.
            test_dataloader (DataLoader or None): The DataLoader for the testing dataset.
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        else:
            logger.info("Batch size was set to %d overriding the config", batch_size)
        if train_size is None:
            train_size = self.config.train_size
        else:
            logger.info("Train size was set to %.2f overriding the config", train_size)
        if balance_training_data is None:
            balance_training_data = self.config.balance_training_data
        else:
            logger.info("balance_training_data was set to %s overriding the config", balance_training_data)
        if aggregate_outputs is None:
            aggregate_outputs = self.config.aggregate_outputs
        else:
            logger.info("aggregate_outputs was set to %s overriding the config", aggregate_outputs)

        train_dataset: CategoricDataset
        test_dataset: CategoricDataset
        train_dataset, test_dataset = dataset.train_test_split(train_size=train_size)
        if balance_training_data:
            train_dataset.balance(train_dataset.output_columns[0])
        train_dataloader: DataLoader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
            )

        if aggregate_outputs:
            logger.debug("Aggregating outputs in the dataframe.")
            test_dataset.group_by()
        test_dataloader: DataLoader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
            )
        return train_dataloader, test_dataloader

    def create_model(
            self,
            dataset: CategoricDataset,
            use_input_embedding: bool = None,
            use_output_embedding: bool = None,
            max_hidden_neurons: int = None,
            hidden_layers: int = None,
            train_nlp_embedding: bool = None,
            nlp_model_name: str = None
            ) -> CategoricNeuralNetwork:
        """
        Create a categoric neural network model.

        Args:
            dataset (CategoricDataset): The dataset to be used for training the model.
            use_input_embedding (bool): Whether to use input embedding in the model.
            use_output_embedding (bool): Whether to use output embedding in the model.
            max_hidden_neurons (int): The maximum number of hidden neurons in the model.
            hidden_layers (int): The number of hidden layers in the model.
            train_nlp_embedding (bool): Whether to train the NLP embedding in the model.
            nlp_model_name (str): The name of the NLP model to be used.

        Returns:
            model (CategoricNeuralNetwork): The created categoric neural network model.
        """
        assert dataset.already_configured, "The dataset must be configured before creating the model."
        if use_input_embedding is None:
            use_input_embedding = self.config.model_uses_input_embedding
        else:
            logger.info("use_input_embedding was set to %s overriding the config", use_input_embedding)
        if use_output_embedding is None:
            use_output_embedding = self.config.model_uses_output_embedding
        else:
            logger.info("use_output_embedding was set to %s overriding the config", use_output_embedding)
        if train_nlp_embedding is None:
            train_nlp_embedding = self.config.model_trains_nlp_embedding
        else:
            logger.info("train_nlp_embedding was set to %s overriding the config", train_nlp_embedding)
        if nlp_model_name is None:
            nlp_model_name = self.config.nlp_model_name
        else:
            logger.info("nlp_model_name was set to %s overriding the config", nlp_model_name)

        if max_hidden_neurons is None:
            max_hidden_neurons = self.config.max_hidden_neurons
        else:
            logger.info("max_hidden_neurons was set to %d overriding the config", max_hidden_neurons)
        if hidden_layers is None:
            hidden_layers = self.config.hidden_layers
        else:
            logger.info("hidden_layers was set to %d overriding the config", hidden_layers)

        info_message = []
        info_message.append("Details on the model:")
        info_message.append(f"Dataset: Trying to relate{dataset.input_columns} to {dataset.output_columns}")
        info_message.append(f"use_input_embedding = {use_input_embedding}")
        info_message.append(f"use_output_embedding = {use_output_embedding}")
        info_message.append(f"max_hidden_neurons = {max_hidden_neurons}")
        info_message.append(f"hidden_layers = {hidden_layers}")
        info_message.append(f"train_nlp_embedding = {train_nlp_embedding}")
        info_message.append(f"nlp_model_name = {nlp_model_name}")

        printer.debug("%s", "\n".join(info_message))

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
            self,
            model: CategoricNeuralNetwork,
            learning_rate: float = None,
            optimizer_name: str = "AdamW"
            ) -> torch.optim.Optimizer:
        """
        Get the optimizer for the model.

        Args:
            model (CategoricNeuralNetwork): The model to be used.
            learning_rate (float): The learning rate for the optimizer.
            optimizer_name (str): The name of the optimizer to be used.
                                Must be one of "AdamW", "SGD", or "Adam".

        Returns:
            optimizer (torch.optim.Optimizer): The optimizer to be used.
        """
        # check the input
        if learning_rate is None:
            learning_rate = self.config.learning_rate
        else:
            logger.info("Learning rate was set to %.2e overriding the config", learning_rate)

        if optimizer_name is None:
            optimizer_name = self.config.optimizer_name
        else:
            logger.info("optimizer_name was set to %s overriding the config", optimizer_name)

        # create the optimizer
        if optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        elif optimizer_name == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        elif optimizer_name == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            raise ValueError("optimizer_name must be one of 'AdamW', 'SGD', or 'Adam'.")
        logger.debug("Optimizer created.")
        return optimizer

    def get_scheduler(
            self,
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
            factor (float): The factor by which the learning rate will be decayed.
            mode (str): The mode for the scheduler.
            patience (int): The patience for the scheduler.
            steps (int): The steps for the scheduler.
            scheduler_type (str): The type of the scheduler.
                                Must be one of "ReduceLROnPlateau" or "StepLR".

        Returns:
            scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler to be used.
        """
        # check the input
        if scheduler_type is None:
            if self.config.scheduler_type is not None:
                scheduler_type = self.config.scheduler_type
            else:
                scheduler_type = "ReduceLROnPlateau"
                logger.debug("scheduler_type was not set. Using ReduceLROnPlateau.")
        else:
            logger.info("scheduler_type was set to %s overriding the config", scheduler_type)
        if factor is None:
            factor = self.config.lr_decay_factor
        else:
            logger.info("learning rate decay factor was set to %.2f overriding the config", factor)
        # create the scheduler
        if scheduler_type == "ReduceLROnPlateau":
            if patience is None:
                patience = self.config.lr_decay_patience
            else:
                logger.info("learning rate decay patience was set to %d overriding the config", patience)
            if mode is None:
                mode = self.config.scheduler_plateau_mode
            else:
                logger.info("scheduler_plateau_mode was set to %s overriding the config", mode)
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
                steps = self.config.lr_decay_step
            else:
                logger.info("learning rate decay step was set to %d overriding the config", steps)
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
        logger.debug("Scheduler created: %s", scheduler.__class__.__name__)
        return scheduler

    def get_loss_fn(
            self,
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
            use_output_embedding = self.config.model_uses_output_embedding
        else:
            logger.info("use_output_embedding was set to %s overriding the config", use_output_embedding)

        if use_output_embedding:
            loss_fn = torch.nn.CosineEmbeddingLoss()
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss()

        logger.debug("Loss function created: %s", loss_fn.__class__.__name__)
        return loss_fn

    def train(
            self,
            model: CategoricNeuralNetwork,
            train_dataloader: DataLoader,
            test_dataloader: DataLoader,
            loss_fn: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            train_targets: dict = None,
            epochs: int = None,
            lr_decay_target: str = None,
            scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
            do_report: bool = True,
            live_report: bool = False
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
                epochs (int): The number of epochs to train the model.
                train_targets (dict): The targets for the training.
                lr_decay_target (str): The target for the learning rate decay.
                                    Only relevant if the scheduler is ReduceLROnPlateau.

            Optional (will default to the values in the function call):
                do_report (bool): Whether to generate a report.
                live_report (bool): Whether to generate a live report.


        Returns:
            None
        """

        if epochs is None:
            epochs = self.config.epochs
        else:
            logger.info("Epochs was set to %d overriding the config", epochs)

        if train_targets is None:
            train_targets = self.config.train_targets
        else:
            logger.info("train_targets was set to %s overriding the config", train_targets)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if lr_decay_target is None:
                lr_decay_target = self.config.lr_decay_target
            else:
                logger.info("lr_decay_target was set to %s overriding the config", lr_decay_target)

        # Define targets for training
        f1_target = train_targets.get("f1", None)
        precision_target = train_targets.get("precision", None)
        recall_target = train_targets.get("recall", None)

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
        info_message.append(f"Configuration: {self.config.print()}")
        if monitor_f1:
            info_message.append(f"f1_target = {f1_target}")
        if monitor_precision:
            info_message.append(f"precision_target = {precision_target}")
        if monitor_recall:
            info_message.append(f"recall_target = {recall_target}")
        info_message.append("\n")

        printer.debug("\n".join(info_message))
        try:
            if do_report:
                df = None
                start_time = time.time()
            for t in range(epochs):
                printer.info("Epoch %d\n-------------------------------", t+1)
                model.train_loop(train_dataloader, loss_fn, optimizer)
                f1_score, precision, recall = model.test_loop(test_dataloader, loss_fn)
                if scheduler is not None:
                    lr_1 = scheduler.get_last_lr()[0]
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        if lr_decay_target == "f1":
                            scheduler.step(f1_score)
                        elif lr_decay_target == "precision":
                            scheduler.step(precision)
                        elif lr_decay_target == "recall":
                            scheduler.step(recall)
                        else:
                            raise ValueError("lr_decay_target must be 'f1', 'precision', or 'recall'.")
                    elif isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
                        scheduler.step()
                    else:
                        raise ValueError("scheduler must be ReduceLROnPlateau or StepLR.")
                    lr_2 = scheduler.get_last_lr()[0]
                    if lr_1 != lr_2:
                        printer.info("\nLearning rate changed from %s to %s\n", lr_1, lr_2)
                        if lr_2 <= 1e-7:
                            logger.warning(
                                "Learning rate got too small."
                                "Interrupting training, meaningless to continue."
                                "Increase patience or decrease factor."
                                )
                            raise StopTraining("Training Stopped. Learning rate too small.")
                if do_report:
                    new_row = pd.DataFrame({
                        "Epoch": [t+1],
                        "F1": [f1_score],
                        "Precision": [precision],
                        "Recall": [recall],
                        "Time": [(time.time() - start_time) / 60]}
                        )
                    df = pd.concat([df, new_row], ignore_index=True)
                    if live_report:
                        # Real-time plotting with Seaborn
                        logger.debug("Updating Plotted report live.")
                        plt.ion()
                        plt.clf()  # Clear the current figure
                        if "f1" in train_targets.keys():
                            color = "red"
                            sns.lineplot(x='Epoch', y='F1', data=df, label='F1 Score', color=color)
                            sns.lineplot(x='Epoch', y=train_targets["f1"], data=df, label='F1 Target', color=color)
                        if "recall" in train_targets:
                            color = "green"
                            sns.lineplot(x='Epoch', y='Recall', data=df, label='Recall', color=color)
                            sns.lineplot(x='Epoch', y=train_targets["recall"], data=df, label='Recall Target', color=color)
                        if "precision" in train_targets:
                            color = "blue"
                            sns.lineplot(x='Epoch', y='Precision', data=df, label='Precision', color=color)
                            sns.lineplot(x='Epoch', y=train_targets["precision"], data=df, label='Precision Target', color=color)
                        plt.legend()
                        plt.pause(0.01)  # Pause to update the plot
                        plt.draw()  # Ensure the plot is updated

                if all([
                    f1_score > f1_target if monitor_f1 else True,
                    precision > precision_target if monitor_precision else True,
                    recall > recall_target if monitor_recall else True
                ]):
                    raise StopTraining("Target reached.")
        except StopTraining as e:
            printer.info(e)
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user.")
        printer.info("Time taken: %.2f minutes.\n", (time.time() - start_time) / 60)
        if do_report:
            self._write_report(df, train_targets=train_targets)

    def _write_report(
            self,
            df: pd.DataFrame,
            train_targets: dict[str, float] = None
            ) -> None:
        """
        Write a report to a file.

        Args:
            df (pd.DataFrame): The DataFrame to be written.
            train_targets (dict): The targets for the training.
                                If there are targets, they will be plotted.

        Returns:
            None
        """
        filename = self.config.report_dir.joinpath(self.config.report_filename)
        if not filename.parent.exists():
            filename.parent.mkdir(parents=True)
        report = []
        report.append("Report on the training")
        report.append("-------------------------------")
        report.append(df.to_string())
        report.append(f"Ran for {df.shape[0]} epochs.")
        report.append("Final values:")
        report.append(f"F1: {df['F1'].iloc[-1]}")
        report.append(f"Precision: {df['Precision'].iloc[-1]}")
        report.append(f"Recall: {df['Recall'].iloc[-1]}")
        report.append(f"Time: {df['Time'].iloc[-1]}")
        printer.info("%s", "\n".join(report))
        sns.set_style("darkgrid")
        sns.set_context("paper")  # Options: paper, notebook, talk, poster
        # save a plot of the report
        fig, ax = plt.subplots()
        f1_color = 'red'
        precision_color = 'blue'
        recall_color = 'green'
        sns.lineplot(x='Epoch', y='F1', data=df, label='F1 Score', ax=ax, color=f1_color)
        sns.lineplot(x='Epoch', y='Recall', data=df, label='Recall', ax=ax, color=recall_color)
        sns.lineplot(x='Epoch', y='Precision', data=df, label='Precision', ax=ax, color=precision_color)
        # print the training targets present
        if train_targets:
            for key, value in train_targets.items():
                if key == "f1":
                    ax.axhline(value, color=f1_color, linestyle='--', label=f"{key} target")
                elif key == "precision":
                    ax.axhline(value, color=precision_color, linestyle='--', label=f"{key} target")
                elif key == "recall":
                    ax.axhline(value, color=recall_color, linestyle='--', label=f"{key} target")
                else:
                    ax.axhline(value, color='black', linestyle='--', label=f"{key} target")
        ax.set_ylabel("Score", rotation="horizontal")
        if self.config.case_name != "default":
            ax.set_title(f"Training {self.config.case_name}")
        # save the
        fig.savefig(filename)
        logger.debug("Report saved as %s", filename)

    def build_and_train_model(
            self,
            df: pd.DataFrame,
            input_columns: list,
            output_columns: list
            ) -> CategoricNeuralNetwork:
        """
        Build and train the model. Will make use of values in ConfigRun.
        They can be set before hand for a custom configuration.

        Args:
            df (pd.DataFrame): The dataset to be used.
            input_columns (list): The names of the columns to be used as input.
            output_columns (list): The names of the columns to be used as output.

        Returns:
            model (CategoricNeuralNetwork): The trained model.
        """
        logger.debug("Building and training the model automatically.")
        dataset = self.configure_dataset(df, input_columns, output_columns)
        logger.debug("Dataset configuration executed.")
        train_dataloader, test_dataloader = self.get_dataloaders(dataset)
        logger.debug("Dataloaders creation executed.")
        model = self.create_model(dataset)
        logger.debug("Model creation executed.")
        loss_fn = self.get_loss_fn()
        logger.debug("Loss function creation executed.")
        optimizer = self.get_optimizer(model)
        logger.debug("Optimizer creation executed.")
        scheduler = self.get_scheduler(optimizer)
        logger.debug("Scheduler creation executed.")
        self.train(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler
            )
        logger.debug("Training executed. Returning the trained model.")
        return model

    def save_model(self, model: CategoricNeuralNetwork, filename: str | Path = None) -> None:
        """
        Save the model to a file.

        This will save the model and its initialization parameters to a file.
        Note that for a given model, 2 files are needed, the initialization parameters
        and the weights of the model parameters themselves.

        Args:
            model (CategoricalNeuralNetwork): The model to be saved.
            filename (str): The name of the file to save the model.

        Returns:
            None
        """
        if filename is None:
            filename = self.config.out_case_dir.joinpath(self.config.case_name)
        else:
            if not isinstance(filename, Path):
                filename = Path(filename).resolve()
            logger.info("Filename to save model was set to %s overriding the config", filename)
        if not filename.parent.exists():
            filename.parent.mkdir(parents=True)

        model_file = filename.with_suffix(".pth")
        ini_file = filename.with_suffix(".ini")
        torch.save(model.state_dict(), model_file)
        with open(ini_file, "w", encoding="utf-8") as fil:
            fil.write(json.dumps(self.config.to_dict()))
            fil.write("\n")
            fil.write(json.dumps(model._config))

        logger.info("Model saved as %s", filename)

    def load_model(self, filename: str) -> CategoricNeuralNetwork:
        """
        Load a model from a file.

        Args:
            filename (str): The name of the file to load the model from.

        Returns:
            model (CategoricNeuralNetwork): The loaded model.
        """

        ini_path = Path(filename).with_suffix(".ini")
        if not ini_path.exists():
            logger.error("File %s does not exist.", ini_path)
            raise FileNotFoundError(f" {ini_path} does not exist.")
        ini_params = ini_path.read_text(encoding="utf-8").splitlines()
        api_params = json.loads(ini_params[0])
        model_params = json.loads(ini_params[1])
        self.config = ConfigRun.from_dict(api_params)
        model = CategoricNeuralNetwork(**model_params)
        model_path = Path(filename).with_suffix(".pth")
        if not model_path.exists():
            logger.error("File %s does not exist.", model_path)
            raise FileNotFoundError(f" {model_path} does not exist.")
        model.load_state_dict(torch.load(model_path))
        logger.info("Model loaded from %s", filename)
        logger.debug("Changing the model to evaluation mode mode.")
        model.to(model.device)
        model.eval()
        return model

    def add_output_possibility(self, model: CategoricNeuralNetwork, output_category: str, output_value: str) -> None:
        """
        Add a possibility for the output to the dataset.

        Args:
            model (CategoricNeuralNetwork): The model to be used.
            output_category (str): The name of the output category.
            output_value (str): The value of the output category.
        Returns:
            None
        """
        model.add_output_possibility(output_category, output_value)
        logger.info("Output possibility added to the model.")

if __name__ == '__main__':
    pass
