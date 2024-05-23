"""
This module contains the definition models for the neural network.
"""
import logging
from typing import Union
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from icecream import ic


logger = logging.getLogger("TFM")
printer = logging.getLogger("printer")
printer.setLevel(logging.INFO)


class StopTraining(Exception):
    """
    Custom exception to stop training the model.
    """


class CategoricNeuralNetwork(nn.Module):
    """
    Neural Network model for classification for a single category.

    Args:
        dataset (CategoricDataset): The dataset used for training the model.
        max_hidden_neurons (int): The number of neurons in the hidden layers.
        use_embedding (bool): Whether to use embeddings or not.

    Attributes:
        linear_relu_stack (nn.Sequential): Sequential module containing linear and ReLU layers.
        tokenizer (AutoTokenizer): The tokenizer used for tokenizing the input data.
        embedding_model (AutoModel): The model used for generating embeddings.
        device (torch.device): The device used for training the model.

    Methods:
        forward(x): Performs forward pass through the network.
        train_loop(dataloader, loss_fn, optimizer): Trains the model using the given dataloader, loss function, and optimizer.
        test_loop(dataloader, loss_fn): Evaluates the model on a test dataset.
        _initialize_weights(): Initializes the weights and biases of the model.
    """

    def __init__(
            self,
            /,
            category_mappings: dict[dict] = None,
            max_hidden_neurons: int = 2**13,
            hidden_layers: int = 2,
            use_input_embedding: bool = False,
            use_output_embedding: bool = False,
            output_embedding_type: str = "categorical",
            train_nlp_embedding: bool = False,
            f1_target: float = 0.7
            ):
        super().__init__()
        self._device: str = self.assess_device()
        self.f1_target = f1_target
        self._use_input_embedding: bool = use_input_embedding
        self._use_output_embedding: bool = use_output_embedding
        # make a dict of each element of dataset.input_columns and their corresponding index in the list
        self.category_mappings: dict[dict] = category_mappings
        self.input_categories: dict[int, str] = self.category_mappings["input_categories"]
        self.output_categories: dict[int, str] = self.category_mappings["output_categories"]
        # "Private" variables
        self._output_category_embeddings_nlp: torch.Tensor = None
        self._similarity_threshold: None = None
        self._nlp_tokenizer: AutoTokenizer = None
        self._nlp_embedding_model: AutoModel = None
        self._train_nlp_embedding: bool = train_nlp_embedding
        self._output_embedding_type: str = output_embedding_type
        self._input_size: int = None
        self._output_size: int = None

        # Initialize the input size
        self.configure_input()
        # Initialize the output size
        self.configure_output(output_embedding_type)

        if self._train_nlp_embedding:
            printer.warning(
                "Training the NLP embedding model can be very slow and resource-intensive.\n"
                "It is recommended to not traine pretrained models further."
                "It can also be very unstable and might not converge.\n"
                )
            for param in self._nlp_embedding_model.parameters():
                param.requires_grad = True
        elif self._train_nlp_embedding is False and self._nlp_embedding_model is not None:
            for param in self._nlp_embedding_model.parameters():
                param.requires_grad = False
        printer.info(f"Input size: {self._input_size}, Output size: {self._output_size}")

        # Initialize the hidden layers
        # Define the neural network
        self.linear_relu_stack = nn.Sequential()
        printer.debug("Estimating the number of neurons needed in the hidden layers.")
        neurons: int = 0
        i: int = 1
        while neurons < (self._input_size + self._output_size) // 2:
            neurons = 2**i
            i += 1
        # hardcoded limit to avoid memory issues
        neurons = min(neurons, max_hidden_neurons, self._input_size)
        self.build_neural_network(neurons, number_layers=hidden_layers)

        printer.info(
            f"Using a neural network {neurons} neurons in the hidden layers.\n"
            f"Mapping to {self._output_size} output categories.")

        # Proceed to initialize the weights and biases
        self._initialize_weights()

        # Define the softmax and cosine similarity functions for easy access
        self.softmax: nn.Softmax = nn.Softmax(dim=-1)
        self.cos: nn.CosineSimilarity = nn.CosineSimilarity(dim=1)

        # Print the parameters that will be trained
        for name, param in self.named_parameters():
            if param.requires_grad:
                printer.debug(f"{name} will be part of the learning layer.")

    def build_neural_network(self, neurons, number_layers):
        """
        Builds the neural network.
        """
        neurons_per_layer = list()
        # decrease the number of neurons per layer gradually into the output size
        for i in range(0, number_layers):
            neurons_per_layer.append(int(neurons - (neurons - self._output_size) * i / number_layers))
        neurons_per_layer.append(self._output_size)
        ic(neurons_per_layer)
        self.add_layer("Layer0", nn.Linear, self._input_size, neurons_per_layer[0])
        for i in range(number_layers):
            self.add_layer(f"Layer{1+i}", nn.Linear, neurons_per_layer[i], neurons_per_layer[i+1])
        self.add_layer(f"Layer{1+number_layers}", nn.Linear, neurons_per_layer[-1], self._output_size, add_activation=False, add_drop=False, add_normalization=False)

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
            self.linear_relu_stack.add_module(name+"_3", nn.Dropout(0.1))

    def _configure_nlp_embedding(self):
        """
        Initializes the tokenizer and embedding model for NLP embeddings.

        Returns:
            None
        """
        if self._nlp_tokenizer is None:
            #self._nlp_tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
            self._nlp_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        if self._nlp_embedding_model is None:
            #self._nlp_embedding_model: AutoModel = AutoModel.from_pretrained('microsoft/codebert-base').to(self.device)
            # maybe use distilbert-base-uncased for faster training
            self._nlp_embedding_model = AutoModel.from_pretrained('distilbert-base-uncased').to(self.device)

    def configure_input(self):
        """
        Initializes the input size.
        It also initializes the tokenizer and embedding model if needed.

        Returns:
            None

        """
        if self._use_input_embedding:
            self._configure_nlp_embedding()
            self._input_size: int = self._nlp_embedding_model.config.hidden_size * len(self.input_categories)
        else:
            self._input_size: int = sum([len(self.category_mappings[col]) for col in self.input_categories.values()])

    def configure_output(self, output_embedding_type):
        """
        Initializes the output size.
        It also initializes the output embedding model if needed.

        Returns:
            None
        """
        total_output_fields: int = sum([len(self.category_mappings[col]) for col in self.output_categories.values()])
        if self._use_output_embedding:
            if output_embedding_type == "categorical":
                # Take a naive guess that half the dimensions should be enough
                self._output_size: int = int(total_output_fields//2) if total_output_fields > 10 else total_output_fields
                printer.info(
                    f"Using output embeddings. "
                    f"Condensing the {total_output_fields} output "
                    f"categories to a "
                    f"{'lower-' if total_output_fields > self._output_size else '':s}"
                    f" dimensional space "
                    f"of {self._output_size} dimensions."
                    )
                printer.warning(
                    "Output embeddings might imply loss of precision due"
                    " to dimensionality reduction.\n "
                    "Also, using these embeddings might make it difficult "
                    "as it might find correlations, but purely between categories."
                    " This is not the same as finding correlations between inputs and outputs."
                    "This is a naive approach, but there might be something to learn from it."
                    )
                self._categorical_embedding: EmbeddingModel = EmbeddingModel(
                    total_output_fields,
                    self._output_size
                    ).to(self.device)
                # What is considered a good similarity can vary according to the problem
                # This is a naive guess, but it can be dynamically adjusted during training
                # by monitoring the F1 score
                self._similarity_threshold: float = 0.5
                for param in self._categorical_embedding.parameters():
                    # The output embedding model should be trained
                    param.requires_grad = True
            elif output_embedding_type == "nlp_embedding":
                # Use a pretrained model for the output embeddings
                self._configure_nlp_embedding()
                self._output_size: int = self._nlp_embedding_model.config.hidden_size * len(self.output_categories)
                printer.info(
                    f"Using output embeddings. "
                    f"Using pretrained model "
                    f"of {self._output_size} embedded dimensions."
                    )
        else:
            self._output_size: int = total_output_fields

    def _initialize_weights(self):
        """
        Initializes the weights and biases of the model.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputs: dict) -> torch.Tensor:
        """
        Performs forward pass through the network.

        Args:
            inputs (dict): A dictionary containing the input data.

        Returns:
            torch.Tensor: The output of the network (logits).
        """
        if self._use_input_embedding:
            pre_processed_inputs: torch.Tensor = self.process_batch_to_embedding(
                batch=inputs,
                nlp=True
                )
        else:
            # No embeddings
            pre_processed_inputs: torch.Tensor = self.process_batch_to_one_hot(
                batch=inputs,
                fields_type="inputs"
                )
        logits: torch.Tensor = self.linear_relu_stack(pre_processed_inputs)
        return logits

    def _apply_nlp_embedding(self, batch_element: torch.Tensor) -> torch.Tensor:
        """
        Applies the NLP embedding to the input data.

        Args:
            batch_element (torch.Tensor): The input data to preprocess.

        Returns:
            None
        """
        # tokenize each of the elements
        tokenized: torch.Tensor = self._nlp_tokenizer(
            batch_element,
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
        output: torch.Tensor = self._nlp_embedding_model(
            input_ids=input_ids,
            attention_mask=attention_mask
            )
        pooling_strategy: str = "cls"  # Change this to the desired pooling strategy
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
        return pooled

    def process_batch_to_embedding(
            self,
            batch: torch.Tensor,
            nlp: bool = False
            ) -> torch.Tensor:
        """
        Preprocesses the batch of elements to embeddings.

        Args:
            batch (list): The list of elements to preprocess.
            nlp (bool): Whether to use NLP embeddings or not.

        Returns:
            embeddings (torch.Tensor): The embeddings.
        """
        # Get the length of the lists in the input dictionary
        lengths: set = set([len(v) for v in batch])
        try:
            assert len(lengths) == 1, "Not all input fields have the same length in this batch"
        except AssertionError:
            printer.debug(batch)
            raise

        # Initialize an empty list to hold the embeddings
        embeddings: list = []
        # With the following code use the embeddings
        for batch_element in zip(*batch):
            # For each index, get the text from each list in the input dictionary
            if nlp is True:
                pooled: torch.Tensor = self._apply_nlp_embedding(batch_element)
            else:
                # if no tokenizer is provided, assume the input is already tokenized
                # i.e. batch_element is a list of tokenized inputs
                # which are really just indexes for the embedding model
                output: torch.Tensor = self._categorical_embedding(
                    torch.tensor(batch_element).to(self.device)
                    )
                # if the output is a 2D tensor, we need to add a dimension
                # to make it a 3D tensor with a batch size of 1
                if len(output.shape) == 2:
                    output: torch.Tensor = output.unsqueeze(1)
                pooling_strategy = "sum"  # Change this to the desired pooling strategy
                if pooling_strategy == "mean":
                    pooled = torch.mean(output, dim=1)
                elif pooling_strategy == "max":
                    pooled = torch.max(output, dim=1).values
                elif pooling_strategy == "sum":
                    pooled = torch.sum(output, dim=1)
                else:
                    raise ValueError("Invalid pooling strategy. Please choose from 'mean', 'max', 'sum', 'cls'.")
            # Normalize the embedding to avoid exploding gradients
            pooled: torch.Tensor = torch.nn.functional.normalize(pooled, p=2, dim=1)
            # Reshape the tensor to a 1D tensor representing the whole batch element
            pooled: torch.Tensor = pooled.reshape(-1)
            embeddings.append(pooled)
        # Stack the embeddings into a tensor. Represents the whole batch
        stacked_embeddings: torch.Tensor = torch.stack(embeddings).to(self.device)
        return stacked_embeddings

    def process_batch_to_one_hot(
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
                field_id: int = self.category_mappings[category][field_value]
                # Convert the input field index to a one-hot vector
                one_hot: torch.Tensor = nn.functional.one_hot(
                    torch.tensor(field_id),
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

    def get_batch_loss(
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
            batch_logits: torch.Tensor = nn.functional.normalize(
                batch_logits, p=2, dim=1
                )
            if self._output_embedding_type == "categorical":
                # if the output embeddings are index embeddings
                # we need to convert the indexes to embeddings
                indexes_to_embed: list = []
                for category_id, batch_element in enumerate(y):
                    output_category: str = self.output_categories[category_id]
                    # get the index of the field in the category
                    indexes: list[int] = [self.category_mappings[output_category][field] for field in batch_element]
                    # consider indexes_to_embed to be a list of tokenized inputs
                    indexes_to_embed.append(indexes)
                # each element in the list is a tokenized input for a possible output category
                # each tensor in the list is a list of indexes for one element of the batch
                y_embeddings: torch.Tensor = self.process_batch_to_embedding(
                    batch=indexes_to_embed,
                    nlp=False
                    )
            else:
                # if the output embeddings are NLP embeddings
                # we need to convert the output strings to embeddings
                y_embeddings: torch.Tensor = self.process_batch_to_embedding(
                    batch=y,
                    nlp=True
                    )
            # in both cases, y_embeddings is a tensor of size (batch_size, embedding_size)
            # and each row is the embedding of the output category
            target: torch.Tensor = torch.ones(batch_logits.size(0), device=self.device)
            loss: torch.Tensor = loss_fn(batch_logits, y_embeddings, target)
            if return_y_one_hot:
                # if y is to be returned as one hot, we have to do it now
                yp: torch.Tensor = self.process_batch_to_one_hot(y, fields_type="outputs")
                return loss, yp
            else:
                return loss
        else:
            yp: torch.Tensor = self.process_batch_to_one_hot(y, fields_type="outputs")
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
            scheduler=None
            ) -> None:
        """
        Trains the model using the given dataloader, loss function, and optimizer.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader containing the training dataset.
            loss_fn (torch.nn.Module): The loss function used to compute the loss.
            optimizer (torch.optim.Optimizer): The optimizer used to update the model's parameters.
            scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler used to adjust the learning rate.

        Returns:
            None
        """
        # Save initial weights
        initial_weights: list[torch.Tensor] = [param.clone() for param in self.parameters()]
        size = len(dataloader)
        # Set the model to training mode - important for batch normalization and dropout layers
        self.train()
        for batch, (x, y) in enumerate(dataloader):
            # Compute prediction and loss
            batch_logits: torch.Tensor = self(x)
            loss: torch.Tensor = self.get_batch_loss(loss_fn, batch_logits, y)
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
                printer.warning(f"Warning: Weights did not change for batch {batch}")

            if batch % 100 == 0 or batch == size - 1:
                loss, current = loss.item(), (batch + 1)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test_loop(
            self,
            dataloader: torch.utils.data.DataLoader,
            loss_fn: nn.Module
            ) -> float:
        """
        Function to evaluate the model on a test dataset.

        Args:
            dataloader (torch.utils.data.DataLoader): The data loader for the test dataset.
            loss_fn (torch.nn.Module): The loss function used for evaluation.

        Returns:
            None
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
                    batch_loss, y_one_hot = self.get_batch_loss(loss_fn, batch_logits, y, return_y_one_hot=True)
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
                            self.get_performance(
                                y_one_hot,
                                idx,
                                probabilities))
                        # The count of values correctly reurned as positive
                        correct_positives += correct_choices
                        # Count of values that wrongly returned as positive
                        false_positives += incorrect_choices
                        # Count of total values that should return as positive
                        total_positives += real_positives
                # Compute the precision, recall, and F1 score for the batch
                precision, recall, f1_score = self.compute_total_metrics(
                    correct_positives,
                    false_positives,
                    total_positives
                    )
                # Compute the average loss for the batch
                test_loss /= num_batches
            else:
                # Output embeddings - needed to identify relevant category by
                # similarity in the embedded space
                if self._output_embedding_type == "categorical":
                    category_embedding: torch.Tensor = self._categorical_embedding.embedding.weight
                    # Normalize the embeddings to avoid exploding gradients
                    category_embedding_normalized: torch.Tensor = nn.functional.normalize(category_embedding, p=2, dim=1)
                else:
                    # NLP embeddings - needed to identify relevant category by
                    # similarity in the embedded space
                    category_embedding_normalized: torch.Tensor = self.output_category_embeddings_nlp
                # Initialize a dictionary to hold the metrics for each threshold
                thresholds_metrics: dict = {}
                # Iterate over a range of thresholds to find the best one
                test_loss: float = 0
                # Iterate over the test dataset
                printer.debug("Starting test with thresholds")
                threshold_candidates = list(x / 100 for x in range(85, 100, 5))
                for x, y in dataloader:
                    # Compute prediction and loss
                    batch_logits: torch.Tensor = self(x)
                    # Compute the loss for the batch
                    batch_loss, y_one_hot = self.get_batch_loss(loss_fn, batch_logits, y, return_y_one_hot=True)
                    # Accumulate the loss for the batch
                    test_loss += batch_loss.item()
                    # Calculate the relevance (or priority) of each option
                    # Has to be done for each batch element separately
                    for idx, logits in enumerate(batch_logits):
                        # Normalize the logits to focus on the cosine similarity
                        logits: torch.Tensor = nn.functional.normalize(logits, p=2, dim=0)
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
                                self.get_performance(
                                    y_one_hot,
                                    idx,
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
                        self.compute_total_metrics(
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
                printer.info(f"Best threshold: {self._similarity_threshold:.2f}")
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
            f"\nTest Result:\nPrecision: {(precision):>6.4f}, Recall: {(recall):>6.4f}, Avg loss: {test_loss:>8.4f} "
            f"\nF1 Score: {f1_score:>12.8f}"
            f"\nCorrect: {correct_positives:>1.0f} / {total_positives}, False Positives: {false_positives:>1.0f}\n"
        )
        if self._similarity_threshold is not None:
            message_printout = (
                f"\nBest threshold for similarity with embedded outputs: {self._similarity_threshold:.2f}"
                + message_printout
            )
        printer.info(message_printout)
        if self._similarity_threshold is not None:
            printer.debug("Best threshold is chosen by trying and see which one gives the best F1 score.")
        if f1_score > self.f1_target:
            raise StopTraining(f"F1 score is above {self.f1_target}. Stopping training.")
        return f1_score

    def compute_total_metrics(
            self,
            correct_positives: int,
            false_positives: int,
            total_positives: int
            ) -> tuple[float, float, float]:
        """
        Computes the precision, recall, and F1 score for a batch of data.

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

    def get_performance(
            self,
            y_one_hot: torch.Tensor,
            idx: int,
            probabilities: torch.Tensor,
            threshold: float = 0.5
            ) -> tuple[int, int, int]:
        """
        Computes the metrics for an individual input in the batch of data.

        Args:
            y_one_hot (torch.Tensor): The one-hot encoded target values.
            idx (int): The index of the batch element.
            probabilities (torch.Tensor): The probabilities of the model.
            threshold (float): The threshold for the probabilities.

        Returns:
            correct_choices (int): The number of correct positive choices.
            incorrect_choices (int): The number of incorrect positive choices.
            total_positives (int): The total number of real positives.
        """
        prb_indices: list = (probabilities > threshold).nonzero().squeeze().tolist()
        # Get indices where y != 0. This is the correct classification
        y_indices: list = (y_one_hot[idx]).nonzero().squeeze().tolist()
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
        input,
        multilabel: bool = False
        ):
        """
        Evaluates the model on a single input.

        Args:
            input (dict): The input data.
            multilabel (bool): Whether the output is multilabel or not.

        Returns:
            chosen_categories (list): The chosen categories.
        """
        logits = self(input)
        if self._use_output_embedding:
            # if using output embedding we need to run the cosine similarity\
            # with each of the possible outputs
            if self._output_embedding_type == "categorical":
                category_embedding: torch.Tensor = self._categorical_embedding.embedding.weight
                # Normalize the embeddings to avoid exploding gradients
                category_embedding_normalized: torch.Tensor = nn.functional.normalize(category_embedding, p=2, dim=1)
            else:
                # NLP embeddings - needed to identify relevant category by
                # similarity in the embedded space
                category_embedding_normalized: torch.Tensor = self.output_category_embeddings_nlp
            # Normalize the logits to focus on the cosine similarity
            logits: torch.Tensor = nn.functional.normalize(logits, p=2, dim=0)
            # Calculate the cosine similarity between the logits and the category embeddings
            cos_sim: torch.Tensor = self.cos(logits.unsqueeze(0), category_embedding_normalized)
            # Calculate the probabilities of the model
            # We identify the cosine similarities
            # between the logits and the category embeddings
            # as the probabilities. This is not strictly correct
            # but it is a good approximation
            probabilities: torch.Tensor = cos_sim
            # apply selection with the best threshold
            output_possibilites: dict[int, str] = {i: x for x, i in self.category_mappings[list(self.output_categories.values())[0]].items()}
            chosen_categories: list = []
            for output_category_idx, category_outcome in enumerate(probabilities):
                if not multilabel:
                    prb_indices: list = (category_outcome == category_outcome.max()).nonzero().squeeze().tolist()
                    assert isinstance(prb_indices, int)
                    chosen: list = output_possibilites[prb_indices]
                else:
                    prb_indices: list = (category_outcome > self._similarity_threshold).nonzero().squeeze().tolist()
                    chosen: list = [output_possibilites[idx] for idx in prb_indices]
                # Get the category names
                chosen_categories.append(chosen)
            return chosen_categories
        else:
            # if not using output embedding we need to process the one_hot vectors
            # and return the categories associated with the one hot vectors
            # Get the indices of the highest probabilities by filter with
            # a threshold of 0.5
            probabilities = torch.sigmoid(logits)
            output_possibilites: dict[int, str] = {i: x for x, i in self.category_mappings[list(self.output_categories.values())[0]].items()}
            chosen_categories: list = []
            ic(probabilities)
            for output_category_idx, category_outcome in enumerate(probabilities):
                if not multilabel:
                    prb_indices: list = (category_outcome ==category_outcome.max()).nonzero().squeeze().tolist()
                    assert isinstance(prb_indices, int)
                    chosen: list = output_possibilites[prb_indices]
                else:
                    prb_indices: list = (category_outcome > 0.5).nonzero().squeeze().tolist()
                    chosen: list = [output_possibilites[idx] for idx in prb_indices]
                # Get the category names
                chosen_categories.append(chosen)
            return chosen_categories

    @property
    def device(self) -> str:
        """
        The device used for training the model.

        Returns:
            device (torch.device): The device used for training the model.
        """
        return self._device

    @property
    def output_category_embeddings_nlp(self) -> AutoModel:
        """
        The output category embeddings for the model.
        Consider it as base against which to compare the embedded logits.
        Consider it as a property to avoid accidental changes.

        Returns:
            output_category_embeddings (AutoModel): The output category embeddings for the model.
        """
        if self._output_embedding_type != "nlp_embedding":
            raise AttributeError("Output embeddings are not enabled.")
        if self._nlp_embedding_model is None:
            raise AttributeError("Embeddings have not been initialized.")

        # Make a list of all the output texts which we have to embed
        if not self._train_nlp_embedding and self._output_category_embeddings_nlp is not None:
            return self._output_category_embeddings_nlp
        else:
            printer.debug("this should only happen once")
            all_outputs_texts = [
                [field for field in self.category_mappings[category]]
                for category in self.output_categories.values()
                ]
            self._output_category_embeddings_nlp = self.process_batch_to_embedding(
                batch=all_outputs_texts,
                nlp=True
                )
            return self._output_category_embeddings_nlp

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
        logger.info(f"Using {device} device")
        return device


class EmbeddingModel(nn.Module):
    """
    A simple model that uses an embedding layer to represent categories.
    """
    def __init__(self, num_categories, embedding_dim):
        """
        Initializes a new instance of the EmbeddingModel class.

        Args:
            num_categories (int): The number of categories.
            embedding_dim (int): The dimension of the embedding.

        Returns:
            None
        """
        super().__init__()
        self.embedding = nn.Embedding(num_categories, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x: The input tensor (expects index values in the tensor).

        Returns:
            The output tensor.
        """
        return nn.functional.normalize(self.embedding(x), p=2, dim=1)


if __name__ == "__main__":
    # Do nothing
    # This is just a package for definitions
    pass
