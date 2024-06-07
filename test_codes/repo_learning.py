
"""
This module contains code for training and evaluating a model recognizing relations
between function names and their tests.

The module includes functions for reading the dataset, configuring the dataset for training, creating a model,
defining the loss function and optimizer, training the model, and evaluating the model on test instances.

It can be used as an example of how to train a model using the model API.
"""
import logging
import time
from pathlib import Path

import pandas as pd
from set_logger import DetailedScreenHandler, DetailedFileHandler
import model_api

logger = logging.getLogger()
printer = logging.getLogger("printer")


def get_data(dataset_name):
    """
    Reads the sentiment dataset and returns it as a pandas DataFrame.

    Returns:
        df (pd.DataFrame): The sentiment dataset as a pandas DataFrame.
    """
    # Read the sentiment dataset
    dataset_location = Path("test_datasets").joinpath(dataset_name)
    dataset_path = Path(__file__).resolve().parent.joinpath(dataset_location)
    df = pd.read_csv(dataset_path).drop("Unnamed: 0", axis=1)
    # put both entries of each match as columns text and label of a df
    return df


def main(dataset_name):
    """
    Main function for training and evaluating a model recognizing relations
    between function names and their tests.
    """
    # Configure the dataset using the input and output columns
    # Give them as lists of strings
    df = get_data(dataset_name=dataset_name)
    input_columns = ["method_name"]
    output_columns = ["test_case_name"]

    # configure the dataset into a model_api.Dataset object
    dataset = model_api.configure_dataset(df, input_columns=input_columns, output_columns=output_columns)
    # Define the configuration for the model
    config = model_api.ConfigRun
    config.epochs = 30
    config.hidden_layers = 2
    config.model_uses_output_embedding = True
    config.nlp_model_name = "huggingface/CodeBERTa-small-v1"
    config.train_size = 1.0
    config.train_targets = {"f1": 0.75, "recall": 0.75}
    # Define the model and training setup
    model = model_api.create_model(dataset=dataset)
    loss_fn = model_api.get_loss_fn()
    optimizer = model_api.get_optimizer(model)
    scheduler = model_api.get_scheduler(optimizer)
    train_dataloader, test_dataloader = model_api.get_dataloaders(dataset)
    start = time.time()
    model_api.train(model=model,
                    train_dataloader=train_dataloader,
                    test_dataloader=test_dataloader,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    scheduler=scheduler
                    )
    printer.debug("Training finished.")
    printer.debug("Time taken: %.2f minutes.", (time.time() - start) / 60)
    # Get a set of random instances from the dataset and test the model
    model.eval()
    dataset.group_by()
    outp = []
    for i in range(10):
        # take a random entry of df and use the input to get some output
        # and compare it with the real output
        inp_name = dataset.data.iloc[i]["method_name"]
        expected_output = dataset.data.iloc[i]["test_case_name"]
        guess = model.execute(inp_name, mode="multilabel")
        inp = f"Input: {inp_name}\nGuess: {guess}"
        out = f"Real: {expected_output}"
        outp.append(inp)
        outp.append(out)
        outp.append("\n")
    printer.info("%s", "\n".join(outp))


if __name__ == '__main__':
    logger.addHandler(DetailedScreenHandler())
    logger.setLevel(logging.INFO)
    printer.addHandler(DetailedFileHandler("execution.log", mode="w"))
    printer.setLevel(logging.DEBUG)
    main(dataset_name="medium_36.csv")  # 36 is hard to learn
