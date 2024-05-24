
"""
NLP module to handle the NLP embeddings. Can process both text
or tuples of text. If tuples of text, their outcome will be concatenated.

This module contains the following classes:
    * NlpEmbedding

This module contains the following functions:
    * None
"""

from typing import Union

import torch
from transformers import AutoTokenizer, AutoModel


class NlpEmbedding():
    """
    Class to handle the NLP embeddings.

    Args:
        model_name (str): The name of the model to be used for embeddings.

    Attributes:
        tokenizer (AutoTokenizer): The tokenizer used for tokenizing the input data.
        embedding_model (AutoModel): The model used for embeddings.
        device (torch.device): The device used for training the model.

    Methods:
        get_embedding(text: str): Gets the embedding for a given text.
    """

    def __init__(self, model_name: str = 'distilbert-base-uncased'):
        self._model_name: str = model_name
        self._device: str = None
        self._configure()
        # maybe use distilbert-base-uncased for faster training
        # self._model_name = 'microsoft/codebert-base'

    def _configure(self):
        """
        Initializes the tokenizer and embedding model for NLP embeddings.

        Returns:
            None
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self.model = AutoModel.from_pretrained(self._model_name).to(self.device)

    @property
    def device(self) -> str:
        """
        The device used for training the model.

        Returns:
            device (torch.device): The device used for training the model.
        """
        return self._device

    @device.setter
    def device(self, value: str) -> None:
        """
        Sets the device used for training the model.

        Args:
            value (str): The device used for training the model.

        Returns:
            None
        """
        if self._device is None:
            self._device = value
        else:
            raise AttributeError("Device has already been set.")
        self.model.to(self.device)

    def unset_device(self) -> None:
        """
        Unsets the device used for training the model.

        This is done so that the device can be set again.
        It has to be done manually so it is hard to do it by accident.

        Returns:
            None
        """
        self._device = None

    def get_embedding(self, inp: Union[str, tuple], pooling_strategy: str = "mean") -> torch.Tensor:
        """
        Gets the embedding for a given text.

        Args:
            text (str or tuple): The text to be embedded.
                                 Note: If tuple, the embeddings will be concatenated.
                                 Thus, the output will have a shape of (n, m)
                                 where n is the number of elements in the tuple
                                 and m is the embedding size.
            pooling_strategy (str): The pooling strategy to be used.

        Returns:
            embedding (torch.Tensor): The embedding for the text.
        """
        # tokenize each of the elements
        if isinstance(inp, str):
            elements: tuple = (inp,)
        else:
            elements: tuple = inp
        tokenized: torch.Tensor = self.tokenizer(
            elements,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=16
            )
        # Get the input_ids and attention_mask from tokenizer
        input_ids, attention_mask = (
            tokenized['input_ids'].to(self.device),
            tokenized['attention_mask'].to(self.device)
            )
        # apply the embedding model on the input ids and attention mask
        output: torch.Tensor = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
            )
        if pooling_strategy == "mean":
            pooled = torch.mean(output.last_hidden_state, dim=1)
        elif pooling_strategy == "max":
            pooled = torch.max(output.last_hidden_state, dim=1).values
        elif pooling_strategy == "sum":
            pooled = torch.sum(output.last_hidden_state, dim=1)
        elif pooling_strategy == "cls":
            pooled = output.last_hidden_state[:, 0, :]
        else:
            raise ValueError("Invalid pooling strategy. Please choose from 'mean', 'max', 'sum', 'cls'.")
        return pooled.reshape(-1)

