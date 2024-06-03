"""
This module contains the definition of a custom dataset class.
"""
import logging
import pandas as pd
from torch.utils.data import Dataset as TorchDataset


logger = logging.getLogger("TFM")
printer = logging.getLogger("printer")


class CategoricDataset(TorchDataset):
    """

    A custom dataset class for handling categoric data.

    This class is used to handle categoric data in a dataset.
    It is used to prepare the data for training a neural network model.
    The dataset should be a pandas DataFrame with labelled columns.

    Attributes:
        data (pd.DataFrame): The data to be used by the dataset.
        input_columns (list[str]): The names of the columns to be used as input.
        output_columns (list[str]): The name of the column to be used as output.
        input_categories (dict): The categories of the input columns.
        output_categories (dict): The categories of the output columns.
        category_mappings (dict): A dictionary mapping categories to unique identifiers.

    Methods:
        define_input_output: Defines the input and output columns to be used by the dataset.
        _create_identifiers: Creates unique identifiers for each category in the output columns.
        reverse_mapping: Given a column name and a numeric id, returns the original category value.
        train_test_split: Split the dataset into training and testing sets.
        standarize: Standarize the dataset by normalizing the data.
        _copy_specs: Copies the specifications of the current dataset to another dataset.
        _group_by: Group the data by the specified input columns and aggregate the outputs.
        __len__: Returns the length of the dataset.
        __getitem__: Returns the item at the given index.
    """

    def __init__(self, data: pd.DataFrame, standarize: bool = False):
        """
        Initializes a new instance of the MyDataset class.

        Args:
            data (pandas.DataFrame): The data to be used by the dataset.
                                     It should have labelled columns.

        Returns:
            None
        """
        printer.debug("Creating a new instance of CategoricDataset.")
        # "Public" Variables
        self.data: pd.DataFrame = data
        if standarize:
            self.standarize()  # Standarize the data
        self.input_columns: None = None
        self.output_columns: None = None
        self.category_mappings: dict[dict] = dict()

        # "Private" Variables
        self._already_configured: bool = False

    def define_input_output(
            self,
            input_columns: list[str],
            output_columns: list[str],
            store_input_features: bool = False
            ) -> None:
        """
        Defines the input and output columns to be used by the dataset.
        Relies on the data being a pandas DataFrame with labelled columns.

        Args:
            input_columns (list[str]): The names of the columns to be used as input.
            output_columns (list[str]): The name of the column to be used as output.
            store_input_features (bool): Whether to store the input features in the dataset.
                                        In many cases this can be irrelevant
                                        and storing them can be memory inefficient.
                                        Especially when the input features are large.

        Returns:
            None
        """
        if self._already_configured:
            raise ValueError(
                "The dataset has already been configured.\n"
                "Please create a new instance of the dataset to configure it again.\n"
            )
        else:
            self._already_configured: bool = True
        assert isinstance(input_columns, list) \
            and all(isinstance(col, str) for col in input_columns), \
            "input_columns should be a list of strings."
        assert isinstance(output_columns, list) and len(output_columns) == 1 \
            and all(isinstance(col, str) for col in output_columns), \
            "output_column should be a list containing a single string."

        # Store the input and output columns labels
        self.input_columns: list[str] = input_columns
        self.output_columns: list[str] = output_columns

        # Group the data by the input columns and aggregate the outputs
        if store_input_features:
            self._create_identifiers(self.input_columns)
        self._create_identifiers(self.output_columns)

        # Create a dictionary mapping the input and output categories to unique identifiers
        input_categories: dict[int, str] = {i: x for i, x in enumerate(input_columns)}
        output_categories: dict[int, str] = {i: x for i, x in enumerate(output_columns)}

        # add them to the category_mappings
        self.category_mappings["input_categories"] = input_categories
        self.category_mappings["output_categories"] = output_categories

    def _create_identifiers(self, columns=None) -> None:
        """
        Creates unique identifiers for each value in the input and output columns.
        They are stored in the category_mappings dictionary.
        They keys are the column names, and the values are dictionaries.
        The dictionaries map the category values to unique identifiers.

        Args:
            columns (list): The names of the columns to be used.

        Returns:
            None

        """
        # Iterate over each desired column
        for column in columns:
            # Create an empty dictionary for the current column
            self.category_mappings[column] = {}
            # Get the entries for the current column
            entries: pd.Series = self.data[column]
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

    def reverse_mapping(self, column: str, code: int) -> str:
        """
        Given a column name and a numeric id, returns the original category value.

        Args:
            column (str): The name of the column to be used.
            code (int): The unique identifier of the value.

        Returns:
            str: The original category value.
        """
        # Reverse lookup in the stored category_mappings
        return next(
            key for key, value in self.category_mappings[column].items() if value == code
        )

    @property
    def number_input_categories(self) -> int:
        """
        Returns the number of categories of the input column.
        This is necessary for the input layer of the neural network.

        Returns:
            int: The number of input fields
        """
        return len(self.input_columns)

    @property
    def number_output_categories(self) -> int:
        """
        Returns the number of categories of the outputs.

        Returns:
            int: The number of output categories

        """
        return len(self.output_columns)

    def train_test_split(self, train_size: float = 0.8) -> tuple:
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
        # Split the data into training and testing sets
        train_df: pd.DataFrame = self.data.sample(frac=train_size, random_state=0)
        train: CategoricDataset = CategoricDataset(train_df)
        test_df: pd.DataFrame = self.data.drop(train.data.index)
        test: CategoricDataset = CategoricDataset(test_df)
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
            train_output: set = set(train.data[self.output_columns].nunique())
            all_output: set = set(self.data[self.output_columns].nunique())
            if all_output.difference(train_output):
                printer.warning(
                    "Not all output possibilities are represented in the training set."
                    )
        return train, test

    def standarize(self) -> None:
        """
        Standarize the dataset by normalizing the data.

        Args:
            None

        Returns:
            None
        """
        # Make all string columns lowercase
        self.data: pd.DataFrame = self.data.map(lambda x: x.lower() if isinstance(x, str) else x)

    @property
    def already_configured(self) -> bool:
        """
        Returns whether the dataset has already been configured.

        Returns:
            bool: Whether the dataset has already been configured.
        """
        return self._already_configured

    def _copy_specs(self, other) -> None:
        """
        Copies the specifications of the current dataset to another dataset.

        Args:
            other (CategoricDataset): The dataset to copy the specifications to.
        """
        other.input_columns = self.input_columns
        other.output_columns = self.output_columns
        other.category_mappings = self.category_mappings
        return None

    def _group_by(self) -> None:
        """
        Group the data by the specified input columns and aggregate the outputs.
        The aggregation is done by concatenating the outputs into a list.

        Args:
           None

        Returns:
            None
        """
        # Aggregate the outputs for each unique input
        self.data = self.data.groupby(self.input_columns)[self.output_columns]\
            .agg(list)\
            .reset_index()

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx) -> tuple[list, list]:
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
        # element is a series of a single row of the dataframe
        element: pd.Series = self.data.iloc[idx]
        # Prepare raw text inputs; placeholders could be used for missing data (not implemented)
        input_texts: list[str] = [str(element[col]) for col in self.input_columns]
        output_texts: list[str] = [str(element[col]) for col in self.output_columns]
        # Prepare the output tensor - one-hot encoding
        return input_texts, output_texts
