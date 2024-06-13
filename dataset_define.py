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
    The input and output columns should be specified before training the model.
    The input and output columns are used to group the data and aggregate the outputs.
    The outputs are aggregated by concatenating them into a list.
    The input and output categories are mapped to unique identifiers.
    The input and output tensors are returned by the __getitem__ method.
    The inputs are the raw text data, and the outputs are multi-hot encoded tensors.
    The dataset can be split into training and testing sets using the train_test_split method.

    Methods:
        define_input_output: Defines the input and output columns to be used by the dataset.
        reverse_mapping: Given a column name and a numeric id, returns the original category value.
        train_test_split: Split the dataset into training and testing sets.
        balance: Balance the dataset by duplicating rows to match the maximum count.
        standarize: Standarize the dataset by normalizing the data.
        group_by: Group the data by the specified input columns and aggregate the outputs.
        __len__: Returns the length of the dataset.
        __getitem__: Returns the item at the given index.

    Attributes:
        number_input_categories: Returns the number of categories of the input column.
        number_output_categories: Returns the number of categories of the outputs.
        already_configured: Returns whether the dataset has already been configured.
    """

    def __init__(self, data: pd.DataFrame, standarize: bool = False):
        """
        Initializes a new instance of the CategoricDataset class.

        Args:
            data (pandas.DataFrame): The data to be used by the dataset.
                                     It should have labelled columns.
            standarize (bool): Whether to standarize the data.

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
            train_size (float): The size of the training set.
                                The testing set will be the remaining data.
                                The default value is 0.8.   # 80% training, 20% testing

        Returns:
            tuple: A tuple containing the training and testing datasets.
        """
        # Split the data into training and testing sets
        train_df: pd.DataFrame = self.data.sample(frac=train_size, random_state=0)
        train: CategoricDataset = CategoricDataset(train_df)
        if train_size == 1:
            # If the training set is the whole dataset, return it as the training and testing sets
            logger.warning(
                "The training set is the whole dataset."
                " The test will also be the whole dataset."
                )
            test_df = train_df.copy()
        else:
            test_df: pd.DataFrame = self.data.drop(train.data.index)
        test: CategoricDataset = CategoricDataset(test_df)
        # reset the index of each to avoid crashing the DataLoader
        train.data.reset_index(drop=True, inplace=True)
        test.data.reset_index(drop=True, inplace=True)
        # make test and train datasets be equal to the original dataset except for the data
        self._copy_specs(train)
        self._copy_specs(test)
        printer.debug("Training set size: %d", len(train))
        printer.debug("Testing set size: %d", len(test))

        # Check if all output possibilities are represented in the training set
        train_output: set = set(train.data[self.output_columns].nunique())
        all_output: set = set(self.data[self.output_columns].nunique())
        if all_output != train_output:
            logger.warning(
                "Not all output possibilities are represented in the training set."
                " It could be bad luck or that the dataset has very sparse outputs."
                )
            # check if the original dataset has very sparse outputs in general
            output_counts_median = self.data[self.output_columns].value_counts().median()
            if output_counts_median < 10:
                logger.warning(
                    "The dataset has very sparse outputs."
                    " The median of the output counts is %d"
                    " It is recommended to use a different dataset."
                    " \nThe program will continue, but the results"
                    " might be meaningless as the model will most likely"
                    " overfit to the training set."
                    "\nThe program will now do its best to make a meaningful"
                    " split, but it is not guaranteed.",
                    output_counts_median
                    )
            # create a new training set equal to the original dataset
            train = CategoricDataset(self.data.copy())
            self._copy_specs(train)
            # create an empty testing set with the same columns as the training set
            test = CategoricDataset(pd.DataFrame(columns=train.data.columns))
            self._copy_specs(test)
            train, test = self.cherry_pick_split(train, test, train_size=train_size)
            logger.warning(
                "Cherry picking split: Training with %d%% of data",
                len(train) / len(self) * 100
                )
            if False:  # deprecated
                logger.warning(
                    "Forcefully copying missing rows to the training."
                    " This can lead to overfitting, and to a biased analysis."
                    " But if there are missing values in the training set,"
                    " the model will not be able to predict them."
                    " There is no way around it if the training set is not representative."
                    )
                self.add_missing_rows(train, test, self.output_columns[0])
        return train, test

    def cherry_pick_split(self, train, test, train_size=0.8):
        """
        Cherry pick a row from the training set and add it to the testing set.
        This is useful when the training dataset is sparse.
        To avoid having unseen outputs in the testing set, we cherry pick a row
        with the most common output value and add it to the testing set.
        That way, the model will be able to at least guarantee seeing the
        least common output value. This is sub-optimal, but not having seen
        outputs in the testing set is worse.

        Args:
            train (CategoricDataset): The training dataset.
            test (CategoricDataset): The testing dataset.

        Returns:
            tuple: A tuple containing the training and testing datasets.
        """
        # Take the most common output value.
        # Remove 1 row from the training set with that value in the column.
        # Add it to the testing set
        # do it in a loop until the split size is as request or until
        # no more rows with the value count of some output dropping to 0
        train: CategoricDataset
        test: CategoricDataset
        train.data: pd.DataFrame
        test.data: pd.DataFrame
        while len(train) / len(self) >= train_size:
            output_count = train.data[train.output_columns[0]].value_counts()
            if output_count.max() == 1:
                logger.warning(
                    "The output column has no repeated values."
                    " It is not possible to split the dataset further."
                    " final size ratio of training set: %d%%"
                    " Test size is likely to be tiny, this can cause validation errors.",
                    len(train) / len(self) * 100
                    )
                logger.warning(
                    "Populating the test with random columns from training set."
                    " This is not ideal, but some statistics are needed to assess the model"
                    )
                # check how many entries we need to add to the test set
                # to reach the desired split size
                needed = int(len(self) * (1 - train_size) - len(test))
                rows = train.data.sample(needed)
                test.data = pd.concat([test.data, rows], ignore_index=True)
                logger.warning("Added %d rows to the test set that are also in training set",
                               needed)
                return train, test
            output_count = train.data[train.output_columns[0]].value_counts()
            value = output_count.idxmax()
            row = train.data[train.data[train.output_columns[0]] == value].sample()
            test.data = pd.concat([test.data, row], ignore_index=True)
            train.data.drop(row.index, inplace=True)
            train.data.reset_index(drop=True, inplace=True)
        return train, test

    def balance(self, column):
        """
        Balance the dataset by duplicating rows to come closer to the mean count.
        (Oversampling minority classes).

        Args:
            column (str): The column to balance.

        Returns:
            None
        """
        df: pd.DataFrame = self.data
        # Count the occurrences of each value in the specified column
        value_counts: pd.Series = df[column].value_counts()
        # Determine the maximum count
        med_count: float = value_counts.median()
        std_count: float = value_counts.std()
        logger.debug(
            "Attempting to balance the dataset by duplicating rows to overcome class imbalance."
            "median= %d, std= %.2f",
            med_count, std_count)
        # Iterate over each unique value in the column
        # Create a list to hold the balanced rows
        df_oversampled_rows = None
        for value, count in value_counts.items():
            # Select all rows with the current value
            rows: pd.DataFrame = df[df[column] == value]
            # Calculate the number of times we need to duplicate the rows
            separation_from_med: int = med_count - count
            # if lower than mean by more than 3 times sigma, oversample
            # until it reaches within 3 times sigma
            if separation_from_med > 3 * std_count:
                num_to_duplicate = int(med_count - std_count - count)
                # Append the original rows and the duplicated rows to the list
                if num_to_duplicate > 0:
                    df_oversampled_rows = pd.concat([df_oversampled_rows, rows.sample(n=num_to_duplicate, replace=True)])
        # Concatenate all balanced rows into a single DataFrame
        balanced_df = pd.concat([df, df_oversampled_rows])
        # Shuffle the balanced DataFrame to mix the rows
        balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)

        # Update the data attribute with the balanced DataFrame
        self.data = balanced_df
        if len(df) != len(balanced_df):
            logger.info(
                "Balanced dataset by duplicating rows to overcome class imbalance."
                " The df size changed from %d to %d",
                len(df), len(balanced_df)
            )
        else:
            logger.debug("No Balancing done. The df size remained the same: %d",
                         len(df)
                         )

    def standarize(self) -> None:
        """
        Standarize the dataset by normalizing the data.
        This is done by making all string columns lowercase.

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
        This is useful when splitting the dataset into training and testing sets.

        Args:
            other (CategoricDataset): The dataset to copy the specifications to.
        """
        other.input_columns = self.input_columns
        other.output_columns = self.output_columns
        other.category_mappings = self.category_mappings
        return None

    def group_by(self) -> None:
        """
        Group the data by the specified input columns and aggregate the outputs.
        The aggregation is done by concatenating the outputs into a list.

        This is useful when the dataset has multiple outputs for a single input,
        and each is split into a separate rows. When training the model,
        this is fine, but when checking the results, we need to group the outputs
        for each input to compare them with all the expected outputs at once.

        Args:
           None

        Returns:
            None
        """
        # Aggregate the outputs for each unique input
        self.data = self.data.groupby(self.input_columns)[self.output_columns]\
            .agg(lambda x: list(set(x)))\
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

    @staticmethod
    def add_missing_rows(dataset1, dataset2, column):
        """
        Add missing rows from dataset2 to dataset1.
        This is useful when the training dataset has missing values that are present in the test dataset.
        This function adds the missing rows to the training dataset.

        Args:
            dataset1 (CategoricDataset): The training dataset.
            dataset2 (CategoricDataset): The test dataset.
            column (str): The column to compare between the two datasets.

        Returns:
            None
        """
        logger.error("This function is deprecated and should not be used.")
        df1 = dataset1.data
        df2 = dataset2.data
        # Find the values in the 'output' column of test that are not in train
        missing_values = df2[~df2[column].isin(df1[column])]
        # Append the missing rows to the train DataFrame
        updated_df1 = pd.concat([df1, missing_values]).reset_index(drop=True)
        dataset1.data = updated_df1

