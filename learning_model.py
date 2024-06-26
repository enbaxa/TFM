"""
This module contains the definition models for the neural network.
"""
import logging
import ast
from typing import Union
import torch
from torch import nn
from torch.nn import functional
from nlp_embedding import NlpEmbedding


logger = logging.getLogger("TFM")
printer = logging.getLogger("printer")


class StopTraining(Exception):
    """
    Custom exception to stop training the model.
    """


class CategoricNeuralNetwork(nn.Module):
    """
    A neural network model for categoric data.

    The model can be trained using a dataset containing input and output data. The model
    can be trained to predict the output data based on the input data.

    The model is designed to work with categoric data, such as the names of functions
    and their test cases. It can be used to predict categories based on the input data.

    The model can use embeddings for the input and output data. The input data can be
    embedded using a pretrained NLP model, and the output data can be embedded using
    the same model. Embedding either is applied independtly of each other.

    The model can be trained to update the embeddings of the NLP model. This can be
    useful when the model is used in a domain-specific context where the pretrained
    embeddings are not sufficient.

    The model can be configured to use a different number of hidden layers and neurons
    in the hidden layers. The number of hidden layers and neurons can be specified when
    creating the model.

    Methods:
        forward: The forward pass of the model.
        train_loop: The training loop of the model.
        test_loop: The testing loop of the model.
        evaluate: Evaluates the model on a test instance (direct).
        execute: Executes the model on a test instance (wrapper of evaluate).
        build_neural_network: Builds the neural network.
        add_layer: Adds a layer to the model.
        _process_batch_to_embedding: Preprocesses the batch to embeddings.
        _process_batch_to_one_hot: Preprocesses the batch to one-hot vectors.
        _get_batch_loss: Computes the loss for a batch of data.
        _get_multi_one_hot: Preprocesses the field value to a multi-hot vector.
        _configure_input: Initializes the input size.
        _configure_output: Initializes the output size.
        _initialize_weights: Initializes the weights and biases of the model.
        _get_performance: Calculates the performance of the model.
        _compute_total_metrics: Computes the total metrics of the model.

    Attributes:
        category_mappings: A dictionary containing the mappings of the categories.
        max_hidden_neurons: The maximum number of hidden neurons in the hidden layers.
        hidden_layers: The number of hidden layers in the model.
        use_input_embedding: Whether to use input embeddings or not.
        use_output_embedding: Whether to use output embeddings or not.
        train_nlp_embedding: Whether to train the NLP embedding model or not.
        nlp_model_name: The name of the NLP model to use for embeddings.
        device: The device to use for the model.
        _use_input_embedding: Whether to use input embeddings or not.
        _use_output_embedding: Whether to use output embeddings or not.
        _nlp_model_name: The name of the NLP model to use for embeddings.
        _device: The device to use for the model.
        _nlp_embedding_model: The NLP embedding model to use for embeddings.
        _output_category_embeddings_nlp: The output category embeddings for the NLP model.
        _similarity_threshold: The similarity threshold to use for the model (Cosine Embeddings).
        _train_nlp_embedding: Whether to train the NLP embedding model or not.
        _input_size: The size of the input data.
        _output_size: The size of the output data.
        linear_relu_stack: The neural network model.
        softmax: The softmax function.
        cos: The cosine similarity function.
    """

    def __init__(
            self,
            /,
            category_mappings: dict[str, dict[int, str]],
            max_hidden_neurons: int = 2**13,
            hidden_layers: int = 2,
            use_input_embedding: bool = True,
            use_output_embedding: bool = False,
            train_nlp_embedding: bool = False,
            nlp_model_name='distilbert-base-uncased',
            similarity_threshold: float = 0.5
            ):
        super().__init__()
        # Save all input parameters in a dictionary for easy access
        # later when we save it to a file.
        # This is only to create the new instance when loading the model
        self._config = {
            "category_mappings": category_mappings,
            "max_hidden_neurons": max_hidden_neurons,
            "hidden_layers": hidden_layers,
            "use_input_embedding": use_input_embedding,
            "use_output_embedding": use_output_embedding,
            "train_nlp_embedding": train_nlp_embedding,
            "nlp_model_name": nlp_model_name,
            "similarity_threshold": similarity_threshold
        }
        self._device: str = self.assess_device()
        self._use_input_embedding: bool = use_input_embedding
        self._use_output_embedding: bool = use_output_embedding
        self._nlp_model_name = nlp_model_name
        # Initialize the NLP embedding model if needed
        if self._use_input_embedding is False and self._use_output_embedding is False:
            logger.warning("Using no embeddings.")
            self._nlp_embedding_model: NlpEmbedding | None = None
        else:
            self._nlp_embedding_model: NlpEmbedding | None = NlpEmbedding(model_name=self._nlp_model_name)
            self._nlp_embedding_model.device = self.device
            # # We will also need an RNN layer to process the embeddings
            # logger.info("Because embeddings are being used, an RNN layer will be added to the model.")
            # self.rnn = nn.RNN(
            #     input_size=self._nlp_embedding_model.model.config.hidden_size,
            #     hidden_size=self._nlp_embedding_model.model.config.hidden_size,
            #     num_layers=1,
            #     batch_first=True,
            #     nonlinearity='relu',
            #     dropout=0.3,
            #     )
        # make a dict of each element of dataset.input_columns and their corresponding index in the list
        self.category_mappings: dict[str, dict[int, str]] = category_mappings
        # "Private" variables
        self._output_category_embeddings_nlp: torch.Tensor | None = None
        self._similarity_threshold: float = similarity_threshold
        self._train_nlp_embedding: bool = train_nlp_embedding
        self._input_size: int = 0
        self._output_size: int = 0

        # Initialize the input size
        self._configure_input()
        # Initialize the output size
        self._configure_output()

        if self._train_nlp_embedding and self._nlp_embedding_model is not None:
            logger.warning(
                "Training the NLP embedding model can be very slow and resource-intensive.\n"
                "It is recommended to not train pretrained models further."
                "It can also be very unstable and might not converge.\n"
                )
            for param in self._nlp_embedding_model.model.parameters():
                param.requires_grad = True
        elif self._train_nlp_embedding is False and self._nlp_embedding_model is not None:
            for param in self._nlp_embedding_model.model.parameters():
                param.requires_grad = False
        printer.info("Input size: %s, Mapping to Output size: %s", self._input_size, self._output_size)

        # Initialize the hidden layers
        # Define the neural network
        self.linear_relu_stack = nn.Sequential()
        printer.info("Estimating the number of neurons needed in the hidden layers.")
        neurons: int = 0
        # dummy counter variable
        i: int = 1
        while neurons < (self._input_size + self._output_size) // 2:
            neurons = 2**i
            i += 1
        # limit to avoid memory issues
        neurons = min(neurons, max_hidden_neurons, self._input_size)
        self.build_neural_network(neurons, number_layers=hidden_layers)

        # Proceed to initialize the weights and biases
        self._initialize_weights()

        # Define the softmax and cosine similarity functions for easy access
        self.softmax: nn.Softmax = nn.Softmax(dim=-1)
        self.cos: nn.CosineSimilarity = nn.CosineSimilarity(dim=1)

        # Print the parameters that will be trained
        training_message = ["The following parameters will be trained:"]
        not_training_message = ["The following parameters will not be trained:"]
        for name, param in self.named_parameters():
            if param.requires_grad:
                training_message.append(f"{name}")
            else:
                not_training_message.append(f"{name} will not be part of the learning layer")
        printer.debug("%s", "\n".join(training_message))
        printer.debug("%s", "\n".join(not_training_message))

    def build_neural_network(self, neurons, number_layers):
        """
        Builds the neural network.

        The neural network is built with a number of hidden layers
        and a number of neurons per layer that decreases gradually
        from the input size to the output size.

        Args:
            neurons (int): The number of neurons in the hidden layers.
            number_layers (int): The number of hidden layers.

        Returns:
            None
        """
        neurons_per_layer = list()
        # decrease the number of neurons per layer gradually into the output size
        for i in range(0, number_layers):
            neurons_per_layer.append(int(neurons - (neurons - self._output_size) * i / number_layers))
        neurons_per_layer.append(self._output_size)
        self.add_layer("Layer0", nn.Linear, self._input_size, neurons_per_layer[0])
        for i in range(number_layers):
            self.add_layer(f"Layer{1+i}", nn.Linear, neurons_per_layer[i], neurons_per_layer[i+1])
        # add the output layer
        self.add_layer(
            f"Layer{1+number_layers}",
            nn.Linear,
            neurons_per_layer[-1],
            self._output_size,
            add_activation=False,
            add_drop=False,
            add_normalization=False
            )
        logger.info("Using a model with the following setup:\n%s", str(self.linear_relu_stack))

    def add_layer(
            self,
            name: str,
            layer: nn.Module,
            neurons_in: int,
            neurons_out: int,
            add_normalization: bool = True,
            add_activation: bool = True,
            add_drop: bool = True
            ) -> None:
        """
        Adds a layer to the model.

        The layer can be a linear layer, and can have normalization,
        activation, and dropout layers after it. The layer is added
        to the linear_relu_stack attribute of the model.

        Args:
            layer (nn.Module): The layer to be added.
            neurons (int): The number of neurons in the layer.
            add_activation (bool): Whether to add an activation function after the layer.
            add_normalization (bool): Whether to add a normalization layer after the layer.
            add_drop (bool): Whether to add a dropout layer after the layer.

        Returns:
            None
        """
        self.linear_relu_stack.add_module(name, layer(neurons_in, neurons_out))
        if add_normalization:
            self.linear_relu_stack.add_module(name+"_1", nn.BatchNorm1d(neurons_out))
        if add_activation:
            self.linear_relu_stack.add_module(name+"_2", nn.LeakyReLU())
        if add_drop:
            self.linear_relu_stack.add_module(name+"_3", nn.Dropout(0.3))

    def _configure_input(self):
        """
        Initializes the input size.

        Returns:
            None

        """
        if self._use_input_embedding:
            self._input_size: int = self._nlp_embedding_model.model.config.hidden_size * len(self.input_categories)
        else:
            self._input_size: int = sum([len(self.category_mappings[col]) for col in self.input_categories.values()])

    def _configure_output(self):
        """
        Initializes the output size.

        Returns:
            None
        """
        total_output_fields: int = sum([len(self.category_mappings[col]) for col in self.output_categories.values()])
        if self._use_output_embedding:
            # Use a pretrained model for the output embeddings
            self._output_size: int = self._nlp_embedding_model.model.config.hidden_size * len(self.output_categories)
            logger.info(
                "Using output embeddings. "
                "Using pretrained model "
                "of %s embedded dimensions.",
                self._output_size
                )
        else:
            # If no embeddings are used, the output size is the number of categories
            self._output_size: int = total_output_fields

    def _initialize_weights(self):
        """
        Initializes the weights and biases of the model.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputs) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            inputs *list): The input data.

        Returns:
            logits (torch.Tensor): The logits of the model.
        """
        if self._use_input_embedding:
            pre_processed_inputs: torch.Tensor = self._process_batch_to_embedding(
                batch=inputs
                ).to(self.device)
            # Failed approach to use RNN layer
            # pre_processed_inputs, _ = self.rnn(pre_processed_inputs)

        else:
            # No embeddings
            try:
                pre_processed_inputs: torch.Tensor = self._process_batch_to_one_hot(
                    batch=inputs,
                    fields_type="inputs"
                    ).to(self.device)
            except KeyError:
                printer.error(
                    "The input fields are not correctly defined."
                    " Please make sure that you set the initial dataset"
                    " To keep identifiers for the input fields.\n"
                    " This is the 'store_input_features' of the dataset class "
                    " method 'define_input_output'. (Should be set to True)"
                    )
                raise
        logits: torch.Tensor = self.linear_relu_stack(pre_processed_inputs)
        return logits

    def _process_batch_to_embedding(
            self,
            batch: torch.Tensor,
            ) -> torch.Tensor:
        """
        Preprocesses the batch of elements to embeddings.

        Args:
            batch (list): The list of elements to preprocess.

        Returns:
            embeddings (torch.Tensor): The embeddings of the elements.
        """
        # Get the length of the lists in the input dictionary
        lengths: set = set([len(v) for v in batch])
        try:
            assert len(lengths) == 1, "Not all input fields have the same length in this batch"
        except AssertionError:
            printer.debug("%s", str(batch))
            raise

        # Initialize an empty list to hold the embeddings
        embeddings: list = []
        # With the following code use the embeddings
        for batch_element in zip(*batch):
            # For each index, get the text from each list in the input dictionary
            embedded_element: torch.Tensor = self._nlp_embedding_model.get_embedding(
                batch_element,
                pooling_strategy="mean"
                )
            embeddings.append(embedded_element)
        # Stack the embeddings into a tensor. Represents the whole batch
        stacked_embeddings: torch.Tensor = torch.stack(embeddings).to(self.device)
        return stacked_embeddings

    def _process_batch_to_one_hot(
            self,
            batch: list,
            fields_type: str
            ) -> torch.Tensor:
        """
        Preprocesses the batch of elements (either input or output) to one-hot vectors.

        Args:
            batch (list): The list of elements to preprocess.
            fields_type (str): The fields to preprocess. (inputs or outputs)

        Returns:
            one_hot_vectors (torch.Tensor): The one-hot encoded vectors.
        """
        # Get the lengths of the lists in the input dictionary
        lengths: set = set([len(v) for v in batch])
        # Check if all input fields have the same length in this batch
        assert len(lengths) == 1, "Not all input fields have the same length in this batch"
        # Initialize an empty list to hold the one-hot vectors for the whole batch
        all_one_hot_vectors: list = []
        # Iterate over each batch element getting all the fields for each category
        if fields_type == "inputs":
            fields: list[str] = self.input_categories
        elif fields_type == "outputs":
            fields: list[str] = self.output_categories
        else:
            raise ValueError("fields_type must be either 'inputs' or 'outputs'.")
        for batch_element in zip(*batch):
            # Initialize an empty list to hold the one-hot vectors for each batch element
            one_hot_vectors: list = []
            # Iterate over each input category and its corresponding  field
            for category_id, field_value in enumerate(batch_element):
                # Get the input category and its corresponding input field index
                category: str = fields[category_id]
                # Convert the input field index to a one-hot vector
                # Check if the field value is a string representation of list
                if field_value.startswith("[") and field_value.endswith("]"):
                    # field_value is a string representation of a list
                    one_hot: torch.Tensor = self._get_multi_one_hot(category, field_value)
                else:
                    # simple string. Just get the index from mapping
                    # and put it in a tensor
                    field_ids: torch.Tensor = torch.tensor(self.category_mappings[category][field_value])
                    one_hot: torch.Tensor = functional.one_hot(
                        field_ids,
                        num_classes=len(self.category_mappings[category])
                        )
                # Append the one-hot vector to the list
                one_hot_vectors.append(one_hot)
            # Concatenate the one-hot vectors into a single tensor
            batch_element_tensor: torch.Tensor = (
                torch.cat(one_hot_vectors).to(self.device)
                )
            # Append the tensor to the list of processed inputs
            all_one_hot_vectors.append(batch_element_tensor)
        # Stack the processed inputs into a tensor
        processed: torch.Tensor = (
            torch.stack(all_one_hot_vectors).to(self.device)
            )
        # Convert the processed inputs to float
        return processed.float()

    def _get_multi_one_hot(self, category: str, field_value: str):
        """
        Preprocesses the field value to a multi-hot vector.

        Args:
            category (str): The category of the field.
            field_value (str): The field value to preprocess.
                               This should be a string representation of a list.

        Returns:
            one_hot (torch.Tensor): The multi-hot vector.
        """
        # Convert the field value to a list and get the indexes
        field_ids: torch.Tensor = torch.tensor(
            [self.category_mappings[category][value]
                for value in ast.literal_eval(field_value)]
                )
        one_hot: torch.Tensor = functional.one_hot(
            field_ids,
            num_classes=len(self.category_mappings[category])
            )
        # If there are multiple field indexes, sum them to get the multi-hot vector
        one_hot = one_hot.sum(dim=0)
        # Append the one-hot vector to the list
        return one_hot

    def _get_batch_loss(
            self,
            loss_fn: nn.Module,
            batch_logits: torch.Tensor,
            y: torch.Tensor,
            return_y_one_hot: bool = False
            ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Computes the loss for a batch of data.

        Args:
            loss_fn (torch.nn.Module): The loss function used to compute the loss.
            batch_logits (torch.Tensor): The logits from the model.
            y (torch.Tensor): The target values.
            return_y_one_hot (bool): Whether to return the processed y or not.
                                       Useful if needed for testing loop

        Returns:
            loss (torch.Tensor): The computed loss.
        """
        if self._use_output_embedding:
            assert isinstance(loss_fn, torch.nn.CosineEmbeddingLoss), "Output embeddings require CosineEmbeddingLoss."
            # normalize the logits to avoid exploding gradients
            batch_logits: torch.Tensor = functional.normalize(
                batch_logits, p=2, dim=1
                )
            # if the output embeddings are NLP embeddings
            # we need to convert the output strings to embeddings
            y_embeddings: torch.Tensor = self._process_batch_to_embedding(
                batch=y
                )
            # normalize the output embeddings to avoid exploding gradients
            y_embeddings_normalized: torch.Tensor = functional.normalize(
                y_embeddings, p=2, dim=1
                )
            # in both cases, y_embeddings is a tensor of size (batch_size, embedding_size)
            # and each row is the embedding of the output category
            target: torch.Tensor = torch.ones(batch_logits.size(0), device=self.device)
            loss: torch.Tensor = loss_fn(batch_logits, y_embeddings_normalized, target)
            if return_y_one_hot:
                # if y is to be returned as one hot, we have to do it now
                yp: torch.Tensor = self._process_batch_to_one_hot(y, fields_type="outputs")
                return loss, yp
            else:
                return loss
        else:
            yp: torch.Tensor = self._process_batch_to_one_hot(y, fields_type="outputs")
            loss = loss_fn(batch_logits, yp)
        if return_y_one_hot:
            return loss, yp
        else:
            return loss

    def train_loop(
            self,
            dataloader: torch.utils.data.DataLoader,
            loss_fn: nn.Module,
            optimizer: torch.optim.Optimizer,
            ) -> None:
        """
        Trains the model using the given dataloader, loss function, and optimizer.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader containing the training dataset.
            loss_fn (torch.nn.Module): The loss function used to compute the loss.
            optimizer (torch.optim.Optimizer): The optimizer used to update the model's parameters.

        Returns:
            None
        """
        # Save initial weights
        initial_weights: list[torch.Tensor] = [param.clone() for param in self.parameters()]

        # Get the size of the dataloader
        size = len(dataloader)
        # Set the model to training mode - important for batch normalization and dropout layers
        self.train()
        for batch, (x, y) in enumerate(dataloader):
            # Compute prediction and loss
            batch_logits: torch.Tensor = self(x)
            loss: torch.Tensor = self._get_batch_loss(loss_fn, batch_logits, y)
            # Backpropagation - ORDER MATTERS!
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Check if weights have changed
            weights_changed: bool = any(
                [(initial_weights[i] != param).any().item()
                 for i, param in enumerate(self.parameters())
                 ]
                )
            if not weights_changed:
                logger.warning("Warning: Weights did not change for batch %s", str(batch))

            # Print the loss every 100 batches
            if batch % 100 == 0 or batch == size - 1:
                loss, current = loss.item(), (batch + 1)
                printer.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test_loop(
            self,
            dataloader: torch.utils.data.DataLoader,
            loss_fn: nn.Module
            ) -> tuple[float, float, float]:
        """
        Function to evaluate the model on a test dataset.

        It computes the F1 score, precision, and recall of the model for the test dataset.

        Args:
            dataloader (torch.utils.data.DataLoader): The data loader for the test dataset.
            loss_fn (torch.nn.Module): The loss function used for evaluation.

        Returns:
            f1_score (float): The F1 score of the model.
            precision (float): The precision of the model.
            recall (float): The recall of the model.
        """
        # Set the model to evaluation mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        self.eval()
        num_batches: int = len(dataloader)
        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            if not self._use_output_embedding:
                correct_positives, false_positives, total_positives = 0, 0, 0
                test_loss: float = 0
                for x, y in dataloader:
                    # Compute prediction and loss
                    batch_logits: torch.Tensor = self(x)
                    batch_loss, y_one_hot = self._get_batch_loss(loss_fn, batch_logits, y, return_y_one_hot=True)
                    test_loss += batch_loss.item()
                    # Calculate the relevance (or priority) of each option
                    # Has to be done for each batch element separately
                    for idx, logits in enumerate(batch_logits):
                        # Calculate the probabilities of the model
                        # We interpret the sigmoid as independent probabilities
                        probabilities: torch.Tensor = torch.sigmoid(logits)
                        # Calculate the estimated correct classification
                        # Get indices where relevance > 0.5 (default threshold)
                        correct_choices, incorrect_choices, real_positives = (
                            self._get_performance(
                                y_one_hot[idx],
                                probabilities))
                        # The count of values correctly reurned as positive
                        correct_positives += correct_choices
                        # Count of values that wrongly returned as positive
                        false_positives += incorrect_choices
                        # Count of total values that should return as positive
                        total_positives += real_positives
                # Compute the precision, recall, and F1 score for the batch
                precision, recall, f1_score = self._compute_total_metrics(
                    correct_positives,
                    false_positives,
                    total_positives
                    )
                # Compute the average loss for the batch
                test_loss /= num_batches
            else:
                # Output embeddings - needed to identify relevant category by
                # similarity in the embedded space
                category_embedding_normalized: torch.Tensor = self.output_category_embeddings_nlp
                # Initialize a dictionary to hold the metrics for each threshold
                thresholds_metrics: dict = {}
                # Iterate over a range of thresholds to find the best one
                test_loss: float = 0
                # Iterate over the test dataset
                printer.debug("Starting test with thresholds")
                threshold_candidates = list(x / 100 for x in range(75, 100, 5))
                for x, y in dataloader:
                    # Compute prediction and loss
                    batch_logits: torch.Tensor = self(x)
                    # Compute the loss for the batch
                    batch_loss, y_one_hot = self._get_batch_loss(loss_fn, batch_logits, y, return_y_one_hot=True)
                    # Accumulate the loss for the batch
                    test_loss += batch_loss.item()
                    # Calculate the relevance (or priority) of each option
                    # Has to be done for each batch element separately
                    for idx, logits in enumerate(batch_logits):
                        # Normalize the logits to focus on the cosine similarity
                        logits: torch.Tensor = functional.normalize(logits, p=2, dim=0)
                        # Calculate the cosine similarity between the logits and the category embeddings
                        cos_sim: torch.Tensor = self.cos(logits.unsqueeze(0), category_embedding_normalized)
                        # Calculate the probabilities of the model
                        # We identify the cosine similarities
                        # between the logits and the category embeddings
                        # as the probabilities. This is not strictly correct
                        # but it is a good approximation
                        probabilities: torch.Tensor = cos_sim
                        for threshold in threshold_candidates:
                            if thresholds_metrics.get(threshold) is None:
                                thresholds_metrics[threshold] = {
                                    "correct_positives": 0,
                                    "false_positives": 0,
                                    "total_positives": 0,
                                    "test_loss": 0
                                    }
                            # Initialize the metrics for the threshold
                            correct_choices, incorrect_choices, real_positives = (
                                self._get_performance(
                                    y_one_hot[idx],
                                    probabilities,
                                    threshold=threshold)
                                    )
                            # Accumulate the metrics for the batch
                            thresholds_metrics[threshold]["correct_positives"] += correct_choices
                            thresholds_metrics[threshold]["false_positives"] += incorrect_choices
                            thresholds_metrics[threshold]["total_positives"] += real_positives

                # Compute the precision, recall, and F1 score for the whole test
                printer.debug("starting threshold assessment")
                for threshold in threshold_candidates:
                    precision, recall, f1_score = (
                        self._compute_total_metrics(
                            thresholds_metrics[threshold]["correct_positives"],
                            thresholds_metrics[threshold]["false_positives"],
                            thresholds_metrics[threshold]["total_positives"]
                            )
                    )
                    # Store the total metrics for the threshold
                    thresholds_metrics[threshold].update(
                        {
                            "precision": precision,
                            "recall": recall,
                            "f1": f1_score,
                            "test_loss": test_loss/num_batches
                        })
                    #
                self._similarity_threshold: float = max(
                    thresholds_metrics,
                    key=lambda x: thresholds_metrics[x]["f1"]
                    )
                # update it in the config
                self._config["similarity_threshold"] = self._similarity_threshold
                printer.info("Best threshold: %.2f", self._similarity_threshold)
                # Get the metrics for the best threshold into an easy-to-access variable
                best_metrics: dict[str, Union[float, int]] = thresholds_metrics[self._similarity_threshold]
                # Get the metrics for the best threshold to printout
                precision, recall, f1_score, test_loss, correct_positives, false_positives, total_positives = (
                    best_metrics["precision"],
                    best_metrics["recall"],
                    best_metrics["f1"],
                    best_metrics["test_loss"],
                    best_metrics["correct_positives"],
                    best_metrics["false_positives"],
                    best_metrics["total_positives"]
                    )
        # Print the results
        message_printout: str = (
            f"\nTest Result: Avg loss: {test_loss:>8.4f} "
            f"\nF1 Score: {f1_score:>12.8f}, \nPrecision: {(precision):>6.4f}, Recall: {(recall):>6.4f}"
            f"\nCorrect: {correct_positives:>d} / {total_positives}, False Positives: {false_positives:>d}\n"
        )
        if self._similarity_threshold != 0:
            message_printout = (
                f"\nBest threshold for similarity with embedded outputs: {self._similarity_threshold:.2f}"
                + message_printout
            )
        printer.info(message_printout)
        if self._similarity_threshold is not None:
            printer.debug("Best threshold is chosen by trying and see which one gives the best F1 score.")
        return f1_score, precision, recall

    def _compute_total_metrics(
            self,
            correct_positives: int,
            false_positives: int,
            total_positives: int
            ) -> tuple[float, float, float]:
        """
        Computes the precision, recall, and F1 score from the performance values.

        Args:
            correct_positives (int): The number of correct positives.
            false_positives (int): The number of false positives.
            total_positives (int): The total number of real positives.

        Returns:
            precision (float): The precision of the model.
            recall (float): The recall of the model.
            f1_score (float): The F1 score of the model.
        """
        try:
            precision: float = correct_positives / (correct_positives + false_positives)
        except ZeroDivisionError:
            precision: float = 0
        try:
            recall: float = correct_positives / total_positives
        except ZeroDivisionError:
            recall: float = 0
        try:
            # compute F1 score
            f1_score: float = 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            f1_score: float = 0
        return precision, recall, f1_score

    def _get_performance(
            self,
            y_one_hot: torch.Tensor,
            probabilities: torch.Tensor,
            threshold: float = 0.5
            ) -> tuple[int, int, int]:
        """
        Computes the metrics for an individual input in the batch of data.

        Args:
            y_one_hot (torch.Tensor): The one-hot encoded target values.
            probabilities (torch.Tensor): The probabilities of the model.
            threshold (float): The threshold for the probabilities.

        Returns:
            correct_choices (int): The number of correct positive choices.
            incorrect_choices (int): The number of incorrect positive choices.
            total_positives (int): The total number of real positives.
        """
        prb_indices: list = (probabilities > threshold).nonzero().squeeze().tolist()
        # Get indices where y != 0. This is the correct classification
        y_indices: list = (y_one_hot).nonzero().squeeze().tolist()
        # Check if the indices are integers
        # If so, convert them to 1-element lists
        if isinstance(prb_indices, int):
            prb_indices: list = [prb_indices]
        if isinstance(y_indices, int):
            y_indices: list = [y_indices]
        # Convert indices to sets for comparison
        prb_set: set = set(prb_indices)
        y_set: set = set(y_indices)
        # Find the intersection of the sets
        correct_choices: int = len(prb_set.intersection(y_set))
        incorrect_choices: int = len(prb_set.difference(y_set))
        total_positives: int = len(y_indices)
        return correct_choices, incorrect_choices, total_positives

    def evaluate(
            self,
            inp,
            mode: str = "monolabel",
            ):
        """
        Evaluates the model on a single input.

        This is a low-level function that can be used to evaluate the model
        on a single input. For a higher-level function that can be used
        to evaluate the model on a batch of inputs, use the execute function.

        Args:
            input (list): The input data.
                          It is a list, each element of which is a list of strings.
                          each element of the list corresponds to a different input category.
                          each element of the inner list corresponds to a different batch element.

            mode (str): The mode of evaluation.
                        Can be either 'monolabel' or 'multilabel' or 'priority'.
        Returns:
            chosen_categories (list): The chosen categories.
                                      if mode is 'monolabel', single element list.

        """
        assert mode in [
            "monolabel", "multilabel", "priority"
            ], \
            "Mode must be either 'monolabel' or 'multilabel' or 'priority'."
        logits = self(inp)
        output_possibilites: dict[int, str] = {
            i: x for x, i in self.category_mappings[
                list(self.output_categories.values())[0]
                ].items()
                }

        if self._use_output_embedding:
            # if using output embedding we need to run the cosine similarity\
            # with each of the possible outputs
            # NLP embeddings - needed to identify relevant category by
            # similarity in the embedded space
            category_embedding_normalized: torch.Tensor = self.output_category_embeddings_nlp
            # Normalize the logits to focus on the cosine similarity
            # because here logits have shape [1, n] where 1 is the batch size here
            # we need to normalize the logits along the 1st dimension (rows)
            logits: torch.Tensor = functional.normalize(logits, p=2, dim=1)
            # Calculate the cosine similarity between the logits and the category embeddings
            cos_sim: torch.Tensor = self.cos(logits, category_embedding_normalized)
            # Calculate the probabilities of the model
            # We identify the cosine similarities
            # between the logits and the category embeddings
            # as the probabilities. This is not strictly correct
            # but it is a good approximation
            probabilities: torch.Tensor = cos_sim
            # functional.cosinesimilarity automatically squeezes the tensor
            # if it is a 1D tensor. We need to unsqueeze it to make it a 2D tensor
            # to be consistent with the code below.
            if probabilities.dim() == 1:
                probabilities = probabilities.unsqueeze(0)

            threshold: float = self._similarity_threshold
        else:
            # if not using output embedding we need to process the one_hot vectors
            # and return the categories associated with the one hot vectors
            # Get the indices of the highest probabilities by filter with
            # a threshold of 0.5
            probabilities = torch.sigmoid(logits)
            threshold = 0.5

        for category_outcome in probabilities:
            if mode == "monolabel":
                prb_indices: list = (category_outcome == category_outcome.max()).nonzero().squeeze().tolist()
                assert isinstance(prb_indices, int), "using monolabel mode but multiple categories are chosen"
                # Get the category name
                chosen: list = [output_possibilites[prb_indices]]
            elif mode == "multilabel":
                prb_indices: list = (category_outcome > threshold).nonzero().squeeze().tolist()
                if isinstance(prb_indices, int):
                    prb_indices = [prb_indices]
                # Get the categor y names
                chosen: list = [output_possibilites[idx] for idx in prb_indices]
            else:
                # priority mode
                # create a list of tuples with the category and the probability
                # sort the list by probability and return the categories
                prb_indices: list = sorted(
                    [(idx, category_outcome[idx]) for idx in range(len(category_outcome))],
                    key=lambda x: x[1],
                    reverse=True
                    )
                chosen: list = [output_possibilites[idx] for idx, _ in prb_indices]
            return chosen

    def execute(self, data: list | str, mode: str = "monolabel") -> list:
        """
        Applies the model on a batch of data.

        This is a high-level function that can be used to execute the model
        on a single input of data. It is just a wrapper around the evaluate function,
        to make it easier are more intuitive to use.

        Args:
            data (list|str): The data to execute the model on.
                         For a single input category, it can be:
                           A string
                           A list of strings.
                         For multiple input categories, it should be a list of lists
                          Each element of the outer list corresponds to a different input category.
                          Each element of the inner list corresponds to a value for that category.
            mode (str): The mode of evaluation.
                        Can be either 'monolabel' or 'multilabel' or 'priority'.

        Returns:
            chosen_categories (list): The chosen categories.
                                      if mode is 'monolabel', single element list.
        """
        if isinstance(data, str):
            if len(self.input_categories) != 1:
                self._raise_value_error_for_multiple_input_categories()
            else:
                single_element_batch = [data]
                return self.evaluate([single_element_batch], mode=mode)
        elif isinstance(data, list):
            if all(isinstance(element, str) for element in data):
                if len(self.input_categories) != 1:
                    self._raise_value_error_for_multiple_input_categories()
                else:
                    outputs = [self.execute(element, mode=mode) for element in data]
                    return outputs
            elif all(isinstance(element, list) for element in data):
                assert len(data) == len(self.input_categories), \
                                     "The number of input categories does not match the input data."
                printer.debug("Make sure that the order of the input categories is correct: %s", self.input_order)
                # make a list of lists of eacn n-th element of each list in data
                batch_length = len(data[0])  # all lists should have the same length
                assert all(len(element) == batch_length for element in data), \
                    "The number of elements in each input category does not match."
                # if the input data is correctly formatted
                # we can evaluate it one by one
                # and return the results
                outputs = []
                # if the batch length is 1 we can return the result directly
                if batch_length == 1:
                    return self.evaluate(data, mode=mode)
                # if the batch length is greater than 1
                # we have to evaluate each element separately
                else:
                    for idx in range(batch_length):
                        single_element_batch = [[element[idx]] for element in data]
                        outputs.append(self.execute(single_element_batch, mode=mode))
                    return outputs
            else:
                raise ValueError("The input data is not correctly formatted.")
            return self.evaluate(single_element_batch, mode=mode)

    def _raise_value_error_for_multiple_input_categories(self):
        """
        Formats and logs an error message if the input data does
        not match the expected format or number of categories.

        Returns:
            None
        """
        error_message = []
        error_message.append("Multiple input categories are expected.")
        error_message.append("Please provide the input data as a list of lists.")
        error_message.append("Each element of the outer list should correspond to a different input category.")
        error_message.append(f"The order of categories MUST be as follows: {self.input_order}")
        printer.error("%s", "\n".join(error_message))
        raise ValueError

    @property
    def device(self) -> str:
        """
        The device used for training the model.

        Returns:
            device (torch.device): The device used for training the model.
        """
        return self._device

    @property
    def output_category_embeddings_nlp(self) -> torch.Tensor:
        """
        The output category embeddings for the model.
        Consider it as base against which to compare the embedded logits.
        Consider it as a property to avoid accidental changes.

        Returns:
            output_category_embeddings_nlp (torch.Tensor): The output category embeddings.

        Raises:
            AttributeError: If the embeddings have not been initialized.
        """
        if self._nlp_embedding_model is None:
            raise AttributeError("Embeddings have not been initialized.")
        # Make a list of all the output texts which we have to embed
        if not self._train_nlp_embedding and self._output_category_embeddings_nlp is not None:
            # if we are not training the embeddings and they are already computed
            return self._output_category_embeddings_nlp
        else:
            # if we are training the embeddings or they are not computed
            # we have to compute them also if it is the first time
            if not self._train_nlp_embedding:
                printer.debug(
                    "Embeddings are not computed yet. Computing them now."
                    "\nThis should only happen once."
                )
            all_outputs_texts = [
                [field for field in self.category_mappings[category]]
                for category in self.output_categories.values()
                ]
            self._output_category_embeddings_nlp = functional.normalize(
                self._process_batch_to_embedding(batch=all_outputs_texts),
                p=2,
                dim=1
                )
            return self._output_category_embeddings_nlp

    @property
    def input_categories(self) -> dict:
        """
        The input categories for the model.

        Returns:
            input_categories (dict): The input categories for the model.
        """
        return self.category_mappings["input_categories"]

    @property
    def input_order(self) -> list:
        """
        The order of the input categories for the model.

        Returns:
            input_order (list): The order of the input categories for the model.
        """
        return list(
            v for _, v in sorted(
                self.input_categories.items(),
                key=lambda x: x[0]
                )
                )

    @property
    def output_categories(self) -> dict:
        """
        The output categories for the model.

        Returns:
            output_categories (dict): The output categories for the model.
        """
        return self.category_mappings["output_categories"]

    @staticmethod
    def assess_device() -> str:
        """
        Determines the device to be used for computation.

        Returns:
            device (str): The device to be used for computation.
        """
        device: str = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        logger.info("Using %s device", device)
        return device

    def add_output_possibility(self, category: str, possibility: str) -> None:
        """
        Adds an output possibility to the model.

        Args:
            category (str): The category to add the possibility to.
            possibility (str): The possibility to add to the category.

        Returns:
            None
        """
        if category not in self.output_categories.values():
            raise ValueError(f"Category {category} is not in the output categories.")
        if not self._use_output_embedding:
            logger.error(
                "Cannot add output possibilities when not using output embeddings."
                "This would require changing the output size of the model."
                )
            raise ValueError("Cannot add output possibilities when not using output embeddings.")
        self.category_mappings[category][possibility] = len(self.category_mappings[category])
        logger.info("Added possibility %s to category %s as a new possible output", possibility, category)
        # eliminate the output embeddings to recalculate them when needed
        # This can be done more efficiently (i.e. just appending the new, but for now this is experimental)
        self._output_category_embeddings_nlp = None


if __name__ == "__main__":
    # Do nothing
    # This is just a package for definitions
    pass
