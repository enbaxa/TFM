"""
This module contains the definition of a custom dataset class.
"""
import logging
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from typing import List


logger = logging.getLogger("TFM")
printer = logging.getLogger("printer")

class CategoricDataset(TorchDataset):
    """

    A custom dataset class for handling categoric data.

    This class is a subclass of the PyTorch Dataset class. It is designed to handle
    datasets with categoric data. It is used to prepare the data for training and testing
    machine learning models.

    Args:
        data (list): The data to be used by the dataset.

    Attributes:
        data (list): The data used by the dataset.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the item at the given index.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initializes a new instance of the MyDataset class.

        Args:
            data (pandas.DataFrame): The data to be used by the dataset.
                                     It should have labelled columns.
        """
        printer.debug("Creating a new instance of CategoricDataset.")
        # "Public" Variables
        self.data = data
        self.input_columns = None
        self.output_columns = None
        self.category_mappings: dict = dict()

        # "Private" Variables
        self._device: str = assess_device()
        self._already_configured: bool = False

    def define_input_output(self, input_columns: List[str], output_columns: List[str]) -> None:
        """
        Defines the input and output columns to be used by the dataset.
        Relies on the data being a pandas DataFrame with labelled columns.

        Args:
            input_columns (List[str]): The names of the columns to be used as input.
            output_columns (List[str]): The name of the column to be used as output.
        """
        if self._already_configured:
            raise ValueError(
                "The dataset has already been configured.\n"
                "Please create a new instance of the dataset to configure it again.\n"
            )
        else:
            self._already_configured = True
        assert isinstance(input_columns, list) and all(isinstance(col, str) for col in input_columns), \
            "input_columns should be a list of strings."
        assert isinstance(output_columns, list) and len(output_columns) == 1 and all(isinstance(col, str) for col in output_columns), \
            "output_column should be a list containing a single string."

        self.input_columns = input_columns
        self.output_columns = output_columns

        self._create_identifiers(self.output_columns)
        self._create_identifiers(self.input_columns)

    def _create_identifiers(self, columns=None):
        """
        Creates unique identifiers for each category in the output columns.

        This method iterates overthe output column to create a mapping
        of categories to unique identifiers. It uses the unique values in each column to
        generate the identifiers.

        Args:
            None

        Returns:
            None
        """
        if columns is None:
            columns = self.output_columns
        # Iterate over each desired column
        for idx, column in enumerate(columns):
            # Create an empty dictionary for the current column
            self.category_mappings[column] = {}
            # Get the entries for the current column
            entries = self.data[column]
            # Iterate over each entry
            for values in entries:
                if not isinstance(values, list):
                    # Check if the value is already in the category mappings
                    if values not in self.category_mappings[column]:
                        self.category_mappings[column][values] = len(self.category_mappings[column])
                else:
                    # Iterate over each value in the entry
                    for value in values:
                        # Check if the value is already in the category mappings
                        if value not in self.category_mappings[column]:
                            # If not, assign a unique identifier to the value
                            self.category_mappings[column][value] = len(self.category_mappings[column])

    def reverse_mapping(self, column: str, code: int):
        """
        Given a column name and a code, returns the original category value.

        Args:
            column (str): The name of the column.
            code (int): The numerical code to be reversed.

        Returns:
            The original category corresponding to the given code.
        """
        # Reverse lookup in the stored category_mappings
        return next(
            key for key, value in self.category_mappings[column].items() if value == code
        )

    @property
    def number_input_categories(self):
        """
        Returns the number of categories of the input column.
        This is necessary for the input layer of the neural network.

        Returns:
            int: The number of input fields
        """
        return len(self.input_columns)

    @property
    def number_output_categories(self):
        """
        Returns the number of categories of the output column.
        This is necessary for the output layer of the neural network.
        (e.g. for a classification problem,
        the number of categories is the number of classes)

        Returns:
            int: The number categories of the output column.
        """
        assert len(self.output_columns) == 1
        return len(self.category_mappings[self.output_columns[0]])

    def train_test_split(self, train_size: float = 0.8):
        """
        Split the dataset into training and testing sets.

        Args:
            dataset (SingleCategoricDataset): The dataset to split.
            train_size (float): The proportion of the dataset to include in the training set.
            output_columns (str):The name of the column to be used as output.
                                 This is optional, and it is only to check if
                                 all output possibilities are reprsented in the training set.

        Returns:
            tuple: A tuple containing the training and testing sets.
        """
        train = CategoricDataset(self.data.sample(frac=train_size, random_state=0))
        test = CategoricDataset(self.data.drop(train.data.index))
        # reset the index of each to avoid crashing the DataLoader
        train.data.reset_index(drop=True, inplace=True)
        test.data.reset_index(drop=True, inplace=True)
        # make test and train datasets be equal to the original dataset except for the data
        self._copy_specs(train)
        self._copy_specs(test)
        printer.debug(f"Training set size: {len(train)}")
        printer.debug(f"Testing set size: {len(test)}")
        if self.output_columns is not None:
            # Check if all output possibilities are represented in the training set
            train_output = set(train.data[self.output_columns].nunique())
            all_output = set(self.data[self.output_columns].nunique())
            if all_output.difference(train_output):
                printer.warning(
                    "Not all output possibilities are represented in the training set."
                    )
        return train, test

    def standarize(self):
        """
        Standarize the dataset by normalizing the data.

        Args:
            None

        Returns:
            None
        """
        # Make all string columns lowercase
        self.data = self.data.map(lambda x: x.lower() if isinstance(x, str) else x)

    @property
    def device(self) -> str:
        """
        Returns the device to be used for computation.

        Returns:
            str: The device to be used for computation.
        """
        return self._device

    @device.setter
    def device(self, value: str) -> None:
        """
        Sets the device to be used for computation.

        Args:
            value (str): The device to be used for computation.
        """
        self._device = value

    def _copy_specs(self, other):
        """
        Copies the specifications of the current dataset to another dataset.

        Args:
            other (CategoricDataset): The dataset to copy the specifications to.
        """
        other.input_columns = self.input_columns
        other.output_columns = self.output_columns
        other.category_mappings = self.category_mappings
        return None

    def _group_by(self):
        """
        Group the data by the specified input columns and aggregate the outputs.

        Args:
           None

        Returns:
            None
        """
        # Aggregate the outputs for each unique input
        self.data = self.data.groupby(self.input_columns)[self.output_columns]\
            .agg(list)\
            .reset_index()

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns the item at the given index.

        This method is called by the DataLoader to retrieve the data elements.
        It returns the input and output tensors for the given index.
        The inputs are the raw text data, and the outputs are multi-hot encoded tensors.

        This allows the inputs to be processed if desired (i.e. by embeddings)
        while the outputs are ready for use in the neural network.

        Args:
            idx (int): The index of the data element to retrieve.

        Returns:
            tuple: A tuple containing the input and output tensors.


        """
        element = self.data.iloc[idx]
        # Prepare raw text inputs; placeholders could be used for missing data
        input_texts = [str(element[col]) for col in self.input_columns]  # Dictionary of column: text
        output_texts = [str(element[col]) for col in self.output_columns]  # Dictionary of column: text
        # Prepare the output tensor - one-hot encoding
        output_encoded = self.create_hot_vector(idx)
        return input_texts, output_encoded

    def create_hot_vector(self, idx, column=None):
        """
        Creates a multi-hot vector for the specified column at the given index.

        Args:
            idx (int): The index of the data element to encode.
            column (str, optional): The column to encode. If None, defaults to the first output column.

        Returns:
            torch.FloatTensor: The multi-hot encoded vector.
        """
        if column is None:
            column = self.output_columns[0]

        element = self.data.iloc[idx]
        values = element[column]

        if not isinstance(values, list):
            values = [values]

        desired_outputs = torch.tensor(
            [self.category_mappings[column][value] for value in values],
            dtype=torch.long
        ).unique()

        output_encoded = torch.nn.functional.one_hot(
            desired_outputs,
            num_classes=len(self.category_mappings[column])
        ).sum(dim=0).to(self.device)

        return output_encoded


def assess_device():
    """
    Determines the device to be used for computation.

    Returns:
        device (str): The device to be used for computation.
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.debug(f"Using {device} device")
    return device
