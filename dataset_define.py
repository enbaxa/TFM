"""
This module contains the definition of a custom dataset class.
"""
import logging
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset


logger = logging.getLogger("TFM")


class CategoricDataset(TorchDataset):
    """
    A custom dataset class for handling data in a specific format.
    It also serves as interface to preparing the relevant fields
    for the training model. It is meant to be a dataset that
    can use an arbitrary number of its columns as input, but
    one single categorical column as output.

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
        self.data = data
        logger.info("Creating a new instance of MyDataset.")
        self.input_columns = []
        self.output_column = []
        self.category_mappings = {}
        self._device = assess_device()

    def define_input_output(self, /, input_columns, output_column):
        """
        Defines the input and output columns to be used by the dataset.
        Relies on the data being a pandas DataFrame with labelled cols.

        Args:
            input_columns: The names of the columns to be used as input.
            output_column: The names of the columns to be used as output.
        """
        self.input_columns = input_columns
        assert len(output_column) == 1, "Only one output column is allowed."
        self.output_column = output_column
        self._group_by()
        self._create_identifiers()

    def _create_identifiers(self):
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
        self.category_mappings = {}
        for column in self.output_column:
            self.category_mappings[column] = {}
            entries = self.data[column]
            for values in entries:
                for value in values:
                    if value not in self.category_mappings[column]:
                        self.category_mappings[column][value] = len(self.category_mappings[column])

    def reverse_mapping(self, column, code):
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
        assert len(self.output_column) == 1
        return len(self.category_mappings[self.output_column[0]])

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

    def _group_by(self):
        """
        Group the data by the specified input columns and aggregate the outputs.

        Args:
           None

        Returns:
            None
        """
        # Aggregate the outputs for each unique input
        self.data = self.data.groupby(self.input_columns)[self.output_column]\
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
        Get the input and output tensors for the given index.

        Args:
            idx (int): The index of the data element to retrieve.

        Returns:
            tuple: A tuple containing the input and output tensors.
        """
        element = self.data.iloc[idx]

        # Prepare raw text inputs; placeholders could be used for missing data
        input_texts = {col: str(element[col]) for col in self.input_columns}  # Dictionary of column: text

        # Prepare the output tensor - one-hot encoding
        desired_outputs = torch.tensor(
            list(
                map(
                    lambda x: self.category_mappings[self.output_column[0]][x],
                    element[self.output_column[0]]
                    )
                ),
            dtype=torch.long
            )

        output_encoded = torch.zeros(
            self.number_output_categories,
            dtype=torch.float
            ).scatter_(
                dim=0,
                index=desired_outputs,
                value=1
                )

        # Assuming the output is categorical and directly usable as a label
        # label = torch.tensor(self.category_mappings[element[self.output_column]], dtype=torch.long)

        return input_texts, output_encoded

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
    logging.info(f"Using {device} device")
    return device
