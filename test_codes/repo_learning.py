
"""
This module contains code for training and evaluating a model recognizing relations
between function names and their tests.

The module includes functions for reading the dataset, configuring the dataset for training, creating a model,
defining the loss function and optimizer, training the model, and evaluating the model on test instances.

It can be used as an example of how to train a model using the model API.
"""
import logging
import time
import argparse
from pathlib import Path
import matplotlib
import pandas as pd
import model_api
matplotlib.use("Agg")

logger = logging.getLogger("TFM")
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


def main(
    *,
    dataset_name,
    neurons,
    layers,
    epochs: int = 30,
    train_size: float = 1.0,
    f1: float = 0.75,
    recall: float = 0.75,
    report_dir: str = "reports"
        ):
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
    config.epochs = epochs
    config.max_hidden_neurons = neurons
    config.hidden_layers = layers
    config.model_uses_output_embedding = True
    config.nlp_model_name = "huggingface/CodeBERTa-small-v1"
    config.train_size = train_size
    config.train_targets = {}
    if f1 is not None:
        config.train_targets["f1"] = f1
    if recall is not None:
        config.train_targets["recall"] = recall
    config.report_dir = Path(report_dir)
    model_api.reconfigure_loggers()
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
    printer.info("Training finished.")
    printer.info("Time taken: %.2f minutes.\n", (time.time() - start) / 60)
    # Get a set of random instances from the dataset and test the model
    model.eval()
    dataset.group_by()
    outp = []
    for i in range(dataset.data.shape[0]):
        # take a random entry of df and use the input to get some output
        # and compare it with the real output
        inp_name = dataset.data.iloc[i]["method_name"]
        expected_output = dataset.data.iloc[i]["test_case_name"]
        guess = model.execute(inp_name, mode="multilabel")
        inp = f"Input: {inp_name}\nGuess: {guess}"
        out = f"Real: {expected_output}"
        #outp.append(inp)
        #outp.append(out)
        outp.append("\n".join((inp, out, "\n")))
    printer.info("%s", "\n".join(outp[:5]))
    logger.info("%s", "\n".join(outp))


if __name__ == '__main__':
    description = (
        "Train and evaluate a model recognizing relations between function names and their tests."
        "\nThe datasets are automatially available by name, and different architectures can be used."
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="medium_36",
        help="The name of the dataset to use for training and evaluation.",
    )
    parser.add_argument(
        "--neurons",
        type=int,
        default=2058,
        help="The maximum number of neurons to use in the hidden layers of the model."
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=2,
        help="The number of hidden layers to use in the model."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="The number of epochs to train the model."
    )
    parser.add_argument(
        "--train_size",
        type=float,
        default=1.0,
        help="The proportion of the dataset to use for training."
    )
    parser.add_argument(
        "--f1",
        type=float,
        default=0.75,
        help="The target F1 score for training the model."
    )
    parser.add_argument(
        "--recall",
        type=float,
        default=0.75,
        help="The target recall score for training the model."
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        default="reports",
        help="The directory to save the reports of the model."
    )
    args = parser.parse_args()
    dataset_name = args.dataset_name  # default "medium_36.csv" is hard to learn
    neurons = args.neurons
    layers = args.layers
    epochs = args.epochs
    train_size = args.train_size
    f1 = args.f1
    recall = args.recall
    report_dir = args.report_dir
    main(
        dataset_name=dataset_name + ".csv",
        neurons=neurons,
        layers=layers,
        epochs=epochs,
        train_size=train_size,
        f1=f1,
        recall=recall,
        report_dir=report_dir
        )
