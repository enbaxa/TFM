"""
This script trains a model to classify sentiment and evaluates it with some sentences.
The dataset used for training is a sentiment dataset with positive and negative sentences.
The model is trained with different configurations of hidden layers and neurons in the hidden layers.
The accuracy of the model is evaluated with some test sentences, which are not in the training dataset.
The test sentences are a mix of positive and negative sentences.
"""
import logging
from pathlib import Path

import pandas as pd
import model_api
from test_codes.sentences import positive_sentences, negative_sentences

logger = logging.getLogger("TFM")
printer = logging.getLogger("printer")


def get_data():
    """
    Reads the sentiment dataset and returns it as a pandas DataFrame.

    Returns:
        df (pd.DataFrame): The sentiment dataset as a pandas DataFrame.
    """
    # Read the sentiment dataset
    dataset_location = Path("test_datasets/sentiment_dataset.txt")
    dataset_path = Path(__file__).resolve().parent.joinpath(dataset_location)
    df = pd.read_csv(dataset_path, sep="\t", header=None, names=["text", "label"])
    df["label"] = ["positive" if x == 1 else "negative" for x in df["label"]]
    print(df)
    return df


def main(neurons: int, layers: int):
    """
    Trains a model to classify sentiment and evaluates it with some sentences.

    Args:
        - neurons (int): The number of neurons in the hidden layers.
        - layers (int): The number of hidden layers.

    Returns:
        - accuracy (float): The accuracy of the model on the test sentences.
        - model (model_api.Model): The trained model.
    """
    # Create an instance of the ConfigRun class
    api = model_api.ModelApi()
    api.config.model_uses_input_embedding = True
    api.config.model_uses_output_embedding = False
    api.config.batch_size = 32
    api.config.max_hidden_neurons = neurons
    api.config.hidden_layers = layers
    api.config.train_targets = {"f1": 0.75}
    api.config.nlp_model_name = "distilbert-base-uncased"
    api.config.case_name = f"sentiment_n{neurons}_l{layers}_faster"
    api.reconfigure_loggers()

    printer.info(f"Running test with {neurons} neurons and {layers} layers")
    # Define the input and output columns
    df: pd.DataFrame = get_data()
    input_columns = ["text"]
    output_columns = ["label"]
    # Configure the dataset using the input and output columns
    dataset = api.configure_dataset(df, input_columns=input_columns, output_columns=output_columns)
    # Create an instance of the model
    model = api.create_model(dataset=dataset)
    # Get loss function, optimizer, and scheduler
    loss_fn = api.get_loss_fn()
    optimizer = api.get_optimizer(model)
    scheduler = api.get_scheduler(optimizer)
    # Get the train and test dataloaders
    train_dataloader, test_dataloader = api.get_dataloaders(dataset)
    # Train the model
    api.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        )
    # the model is now trained, let's evaluate it with some sentences
    # These are not in the training or testing dataset, they are completely new
    model.eval()
    output_possibilites: dict[int, str] = {i: x for x, i in model.category_mappings["label"].items()}
    logger.info(output_possibilites)

    # Count the number of correct guesses
    correct, total = 0, 0
    printer.info("Testing the model with some sentences (not in the dataset)")
    for sentence in positive_sentences:
        guessed_category = model.execute(sentence, mode="monolabel")[0]
        logger.info(
            "Input Sentence: '%s'\n"
            "Guessed Category: %s\n"
            "Expected Category: 'positive'\n"
            "-----------------------------",
            sentence, guessed_category
        )
        correct += 1 if guessed_category == "positive" else 0
        total += 1

    for sentence in negative_sentences:
        guessed_category = model.execute(sentence, mode="monolabel")[0]
        logger.info(
            "Input Sentence: '%s'\n"
            "Guessed Category: %s\n"
            "Expected Category: 'negative'\n"
            "-----------------------------",
            sentence, guessed_category
        )
        correct += 1 if guessed_category == "negative" else 0
        total += 1

    printer.info("correct count: %d/%d", correct, total)
    printer.info("Accuracy: %.2f%%\n\n", correct/total*100)
    return correct / total*100, model


if __name__ == "__main__":
    # Run the main function with different configurations
    msg = []
    neurons_attempt = (16, 32, 64)
    layers_attempt = (1, 2, 3)
    for neurons in neurons_attempt:
        for layers in layers_attempt:
            accuracy, model = main(neurons, layers)
            msg.append(f"Accuracy with {neurons} neurons and {layers} layers: {accuracy:.2f}%\n")
    printer.info("%s", "".join(msg))
