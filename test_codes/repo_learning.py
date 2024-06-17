
"""
This module contains code for training and evaluating a model recognizing relations
between function names and their tests.

The module includes functions for reading the dataset, configuring the dataset for training, creating a model,
defining the loss function and optimizer, training the model, and evaluating the model on test instances.

It can be used as an example of how to train a model using the model API.
"""
import logging
import argparse
import json
import re
from pathlib import Path
import pandas as pd
import model_api
from dataset_define import CategoricDataset
from learning_model import CategoricNeuralNetwork

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


def evaluate_instances(dataset: CategoricDataset, model: CategoricNeuralNetwork, api: model_api.ModelApi):
    """
    Evaluates the model on the instances of the dataset.

    Args:
        dataset (CategoricDataset): The dataset to evaluate the model on.
        model (CategoricNeuralNetwork): The model to evaluate.
        api (model_api.ModelApi): The model API instance.

    Returns:
        results_percentual_dict (dict): A dictionary with the results of the evaluation.
    """
    correct_guesses = 0
    incorrect_guesses = 0
    missing_guesses = 0
    perfect_guesses = 0
    correct_guesses_priority = 0
    total_guesses_made = 0
    total_guesses_expected = 0
    total_classifications_made = 0
    outp = []
    model.eval()
    dataset.group_by()
    dataset.data.reset_index(drop=True, inplace=True)  # reset index to avoid problems
    for i, row in dataset.data.iterrows():
        if i % 10 == 0:
            printer.info("Evaluating instance %d of %d", i, dataset.data.shape[0])
        # take a random entry of df and use the input to get some output
        # and compare it with the real output
        inp_name = row["method_name"]
        expected_output = row["test_class_name"]
        guess = model.execute(inp_name, mode="multilabel")
        inp = f"Input: {inp_name}\nGuess: {guess}"
        out = f"Real: {expected_output}"
        if i < 5:
            printer.info("\n%s", "\n".join((inp, out, "\n")))
        correct_guesses += len([x for x in guess if x in expected_output])
        incorrect_guesses += len([x for x in guess if x not in expected_output])
        missing_guesses += len([x for x in expected_output if x not in guess])
        if set(guess) == set(expected_output):
            perfect_guesses += 1
        # priority mode
        guess_priority = model.execute(inp_name, mode="priority")
        # count correct guesses in the first 33% of the list
        top = guess_priority[:int(len(guess_priority) * 0.33)]
        prio_out = f"Priority: {top}"
        outp.append("\n".join((inp, out, prio_out, "\n")))
        if all([x in top for x in expected_output]):
            correct_guesses_priority += 1
        total_guesses_made += len(guess)
        total_guesses_expected += len(expected_output)
        total_classifications_made += 1
    results_percentual_dict = {
        "Correct guesses": correct_guesses / total_guesses_made * 100 if total_guesses_made else 0,
        "Incorrect guesses": incorrect_guesses / total_guesses_made * 100 if total_guesses_made else 0,
        "Missing guesses": missing_guesses / total_guesses_expected * 100,
        "Perfect guesses": perfect_guesses / total_classifications_made * 100,
        "correctintop": correct_guesses_priority / total_classifications_made * 100
    }
    logger.info("%s", "\n".join(outp))
    logger.info(
        "Correct guesses: %d. This is %.3f%% of the total\n"
        "Incorrect guesses: %d. This is %.3f%% of the total\n"
        "Missing guesses: %d. This is %.3f%% of the total\n"
        "Perfect guesses: %d. This is %.3f%% of the total\n"
        "All correct guesses in first 33%% of priority mode: %.4f%% of the total\n",
        correct_guesses, results_percentual_dict["Correct guesses"],
        incorrect_guesses, results_percentual_dict["Incorrect guesses"],
        missing_guesses, results_percentual_dict["Missing guesses"],
        perfect_guesses, results_percentual_dict["Perfect guesses"],
        results_percentual_dict["correctintop"]
        )
    # Addtional. Parse the log file to get the results
    json_log = api.config.out_dir.joinpath(f"{api.config.case_name}/reports").joinpath("report.jsonl")
    with open(json_log, "r") as f:
        for line in f:
            log_line = json.loads(line)
            if "Report on the training" in log_line["message"]:
                train_report = line
                break
    mat = re.search(
        r"Ran for (?P<epochs>\d+) epochs"
        r".*F1: (?P<F1>\d\.\d+)"
        r".*Precision: (?P<Precision>\d\.\d+)"
        r".*Recall: (?P<Recall>\d\.\d+)"
        r".*Time: (?P<Time>\d+\.\d+)",
        train_report)
    results_percentual_dict.update({k: float(v) for k, v in mat.groupdict().items()})
    return results_percentual_dict

def main(
    *,
    dataset_name,
    neurons,
    layers,
    epochs: int = 30,
    learning_rate: float = 0.001,
    train_size: float = 1.0,
    f1: float = 0.75,
    recall: float = 0.75,
    case_name: str = "default"
        ):

    # Define the configuration for the model
    api = model_api.ModelApi()
    api.config.epochs = epochs
    api.config.max_hidden_neurons = neurons
    api.config.hidden_layers = layers
    api.config.model_uses_output_embedding = True
    api.config.nlp_model_name = "huggingface/CodeBERTa-small-v1"
    api.config.train_size = train_size
    api.config.learning_rate = learning_rate
    api.config.train_targets = {}
    if f1 is not None:
        api.config.train_targets["f1"] = f1
    if recall is not None:
        api.config.train_targets["recall"] = recall
    api.config.lr_decay_patience = 3
    api.config.case_name = case_name + f"_l{layers}"
    api.config.out_dir = Path("out/").resolve()
    api.config.model_trains_nlp_embedding = False
    api.config.model_uses_output_embedding = True
    api.reconfigure_loggers()
    # Configure the dataset using the input and output columns
    # Give them as lists of strings
    df = get_data(dataset_name=dataset_name)
    input_columns = ["method_name"]
    output_columns = ["test_class_name"]
    model = api.build_and_train_model(df, input_columns, output_columns)
    printer.info("Training finished.")
    api.save_model(model, api.config.out_dir.joinpath(api.config.case_name).joinpath("model"))
    # initialize counters
    dataset = api.configure_dataset(df, input_columns=input_columns, output_columns=output_columns)
    return evaluate_instances(dataset, model, api)


if __name__ == '__main__':
    description = (
        "Train and evaluate a model recognizing relations between function names and their tests."
        "\nThe datasets are automatially available by name, and different architectures can be used."
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="medium_3",
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
        default=1,
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
        default=0.70,
        help="The target F1 score for training the model."
    )
    parser.add_argument(
        "--recall",
        type=float,
        default=0.75,
        help="The target recall score for training the model."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="The initial learning rate to use for training the model."
    )
    args = parser.parse_args()
    dataset_name = args.dataset_name  # default "medium_3.csv" learns nicely
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
