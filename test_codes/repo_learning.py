
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
import pandas as pd
import model_api

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
    case_name: str = "default"
        ):
    """
    Main function for training and evaluating a model recognizing relations
    between function names and their tests.
    """

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
    config.case_name = case_name
    model_api.reconfigure_loggers()
    # Configure the dataset using the input and output columns
    # Give them as lists of strings
    df = get_data(dataset_name=dataset_name)
    input_columns = ["method_name"]
    output_columns = ["test_case_name"]
    # configure the dataset into a model_api.Dataset object
    dataset = model_api.configure_dataset(df, input_columns=input_columns, output_columns=output_columns)
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
    # initialize counters
    correct_guesses = 0
    incorrect_guesses = 0
    missing_guesses = 0
    perfect_guesses = 0
    correct_guesses_priority = 0
    total_guesses_made = 0
    total_guesses_expected = 0
    total_classifications_made = 0
    for i in range(dataset.data.shape[0]):
        # take a random entry of df and use the input to get some output
        # and compare it with the real output
        inp_name = dataset.data.iloc[i]["method_name"]
        expected_output = dataset.data.iloc[i]["test_case_name"]
        guess = model.execute(inp_name, mode="multilabel")
        inp = f"Input: {inp_name}\nGuess: {guess}"
        out = f"Real: {expected_output}"
        outp.append("\n".join((inp, out, "\n")))
        correct_guesses += len([x for x in guess if x in expected_output])
        incorrect_guesses += len([x for x in guess if x not in expected_output])
        missing_guesses += len([x for x in expected_output if x not in guess])
        if set(guess) == set(expected_output):
            perfect_guesses += 1
        # priority mode
        guess_priority = model.execute(inp_name, mode="priority")
        # count correct guesses in the first 20% of the list
        if all([x in guess_priority[:int(len(guess_priority) * 0.2)] for x in expected_output]):
            correct_guesses_priority += 1
        total_guesses_made += len(guess)
        total_guesses_expected += len(expected_output)
        total_classifications_made += 1

    printer.info("%s", "\n".join(outp[:5]))
    logger.info("%s", "\n".join(outp))

    results_percentual_dict ={
        "Correct guesses": correct_guesses / total_guesses_made * 100,
        "Incorrect guesses": incorrect_guesses / total_guesses_made * 100,
        "Missing guesses": missing_guesses / total_guesses_expected * 100,
        "Perfect guesses": perfect_guesses / total_classifications_made * 100,
        "correctintop20": perfect_guesses / total_classifications_made * 100
    }
    logger.info(
        "Correct guesses: %d. This is %.3f%% of the total\n"
        "Incorrect guesses: %d. This is %.3f%% of the total\n\n"
        "Missing guesses: %d. This is %.3f%% of the total\n\n"
        "Perfect guesses: %d\n. This is %.3f%% of the total\n\n"
        "All correct guesses in first 20%% of priority mode: %.4f%% of the total\n",
        correct_guesses, results_percentual_dict["Correct guesses"],
        incorrect_guesses, results_percentual_dict["Incorrect guesses"],
        missing_guesses, results_percentual_dict["Missing guesses"],
        perfect_guesses, results_percentual_dict["Perfect guesses"],
        results_percentual_dict["correctintop20"]
        )
    return results_percentual_dict


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
    args = parser.parse_args()
    dataset_name = args.dataset_name  # default "medium_36.csv" is hard to learn
    neurons = args.neurons
    layers = args.layers
    epochs = args.epochs
    train_size = args.train_size
    f1 = args.f1
    recall = args.recall
    main(
        dataset_name=dataset_name + ".csv",
        neurons=neurons,
        layers=layers,
        epochs=epochs,
        train_size=train_size,
        f1=f1,
        recall=recall,
        case_name=dataset_name
        )
