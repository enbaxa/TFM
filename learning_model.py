"""
This module contains the definition models for the neural network.
"""
import logging
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel


logger = logging.getLogger("TFM")
printer = logging.getLogger("printer")
printer.setLevel(logging.INFO)


class StopTraining(Exception):
    """
    Custom exception to stop training the model.
    """
    pass

class CategoricNeuralNetwork(nn.Module):
    """
    Neural Network model for classification for a single category.

    Args:
        dataset (CategoricDataset): The dataset used for training the model.
        hidden_neurons (int): The number of neurons in the hidden layers.
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
            category_mappings=None,
            hidden_neurons=None,
            use_input_embedding=False,
            freeze_input_embedding=True,
            use_output_embedding=False,
            freeze_output_embedding=False
            ):
        super().__init__()
        self._device = self.assess_device()
        self._use_input_embedding = use_input_embedding
        self._use_output_embedding = use_output_embedding
        # make a dict of each element of dataset.input_columns and their corresponding index in the lsit
        self.category_mappings = category_mappings
        self.input_categories = self.category_mappings["input_categories"]
        self.output_categories = self.category_mappings["output_categories"]
        self._similarity_threshold = None

        # Initialize the input size
        if self._use_input_embedding:
            self._tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
            self._input_embedding_model = AutoModel.from_pretrained('microsoft/codebert-base').to(self.device)
            # maybe use distilbert-base-uncased for faster training
            # self._tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            # self._input_embedding_model = AutoModel.from_pretrained('distilbert-base-uncased').to(self.device)
            self._input_size = self._input_embedding_model.config.hidden_size * len(self.input_categories)
            if freeze_input_embedding:
                for param in self._input_embedding_model.parameters():
                    # Letting the embedding train makes it difficult to train the model
                    # The pretrained model is already trained and should not be retrained
                    param.requires_grad = False
        else:
            self._input_size = sum([len(self.category_mappings[col]) for col in self.input_categories.values()])

        # Initialize the output size
        total_output_fields = sum([len(self.category_mappings[col]) for col in self.output_categories.values()])
        if self._use_output_embedding:
            # Take a naive guess that half the dimensions should be enough
            self._output_size = int(total_output_fields // 2)
            printer.info(
                f"Using output embeddings. "
                f"Condensing the {total_output_fields} output "
                f"categories to a lower-dimensional space "
                f"of {self._output_size} dimensions."
                )
            printer.warning(
                "Output embeddings imply loss of precision due to dimensionality reduction.\n "
                "Also the logits are unstable and may not be suitable for training.\n"
                )
            self._output_embedding_model = EmbeddingModel(total_output_fields, self._output_size).to(self.device)
            # What is considered a good similarity can vary according to the problem
            # This is a naive guess, but it can be dynamically adjusted during training
            # by monitoring the F1 score
            self._similarity_threshold = 0.5
            if freeze_output_embedding:
                printer.warning(
                    "Freezing the output embedding model."
                    " This may not be a good idea."
                    " The output embedding model should be trained."
                    )
                for param in self._output_embedding_model.parameters():
                    # The output embedding model should be trained
                    param.requires_grad = False
        else:
            self._output_size = total_output_fields
        printer.info(f"Input size: {self._input_size}, Output size: {self._output_size}")

        # Initialize the hidden layers
        if not hidden_neurons:
            printer.debug("Estimating the number of neurons needed in the hidden layers.")
            neurons = 0
            i = 1
            while neurons < (self._input_size + self._output_size) // 2:
                neurons = 2**i
                i += 1
            # hardcoded limit to avoid memory issues
            neurons = min(neurons, 2**13)
        else:
            neurons = hidden_neurons

        printer.info(
            f"Using a neural network {neurons} neurons in the hidden layers.\n"
            f"Mapping to {self._output_size} output categories.")

        # Define the neural network
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self._input_size, neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            #nn.BatchNorm1d(neurons),
            nn.Linear(neurons, neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            #nn.BatchNorm1d(neurons),
            nn.Linear(neurons, neurons),
            nn.LeakyReLU(),
            nn.Linear(neurons, self._output_size)
        )

        # Proceed to initialize the weights and biases
        self._initialize_weights()

        # Define the softmax and cosine similarity functions for easy access
        self.softmax = nn.Softmax(dim=-1)
        self.cos = nn.CosineSimilarity(dim=1)

        # Print the parameters that will be trained
        for name, param in self.named_parameters():
            if param.requires_grad:
                printer.debug(f"{name} will be part of the learning layer.")

    def _initialize_weights(self):
        """
        Initializes the weights and biases of the model.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
        #nn.init.xavier_uniform(self._output_embedding_model.embedding.weight)

    def forward(self, inputs):
        """
        Performs forward pass through the network.

        Args:
            inputs (dict): A dictionary containing the input data.

        Returns:
            logits (torch.Tensor): The output of the network.
        """
        if self._use_input_embedding:
            pre_processed_inputs = self.process_batch_to_embedding(
                batch=inputs,
                embedding_model=self._input_embedding_model,
                tokenizer=self._tokenizer
                )
        else:
            # No embeddings
            pre_processed_inputs = self.process_batch_to_one_hot(batch=inputs, fields_type="inputs")
        logits = self.linear_relu_stack(pre_processed_inputs)
        return logits

    def process_batch_to_embedding(self, batch, embedding_model, tokenizer=None):
        """
        Preprocesses the batch of elements to embeddings.

        Args:
            batch (list): The list of elements to preprocess.
            embedding_model (AutoModel): The model used for generating embeddings.
            tokenizer (AutoTokenizer): The tokenizer used for tokenizing the input data.
                                        Can be None if the input is already tokenized.

        Returns:
            embeddings (torch.Tensor): The embeddings.
        """
        # Get the length of the lists in the input dictionary
        lengths = set([len(v) for v in batch])
        assert len(lengths) == 1, "Not all input fields have the same length in this batch"

        # Initialize an empty list to hold the embeddings
        embeddings = []
        # With the following code use the embeddings
        for batch_element in zip(*batch):
            # For each index, get the text from each list in the input dictionary
            if tokenizer is not None:
                # tokenize each of the elements
                tokenized = tokenizer(batch_element, return_tensors="pt", padding="max_length", truncation=True, max_length=16)
                # Get the input_ids and attention_mask from tokenizer
                input_ids, attention_mask = tokenized['input_ids'].to(self.device), tokenized['attention_mask'].to(self.device)
                # apply the embedding model on the input ids and attention mask
                output = self._input_embedding_model(input_ids=input_ids, attention_mask=attention_mask)
                pooling_strategy = "cls"  # Change this to the desired pooling strategy
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
            else:
                # if no tokenizer is provided, assume the input is already tokenized
                # i.e. batch_element is a list of tokenized inputs
                # which are really just indexes for the embedding model
                output = embedding_model(torch.tensor(batch_element).to(self.device))
                if len(output.shape) == 2:
                    output = output.unsqueeze(1)
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
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            # Reshape the tensor to a 1D tensor representing the whole batch element
            pooled = pooled.reshape(-1)
            embeddings.append(pooled)
        # Stack the embeddings into a tensor. Represents the whole batch
        stacked_embeddings = torch.stack(embeddings).to(self.device)
        return stacked_embeddings

    def process_batch_to_one_hot(
            self,
            batch: list,
            fields_type: str
            ):
            """
            Preprocesses the batch of elements (either input or output) to one-hot vectors.

            Args:
                batch (list): The list of elements to preprocess.
                fields_type (str): The fields to preprocess. (inputs or outputs)

            Returns:
                one_hot_vectors (torch.Tensor): The one-hot encoded vectors.
            """
            # Get the lengths of the lists in the input dictionary
            lengths = set([len(v) for v in batch])
            # Check if all input fields have the same length in this batch
            assert len(lengths) == 1, "Not all input fields have the same length in this batch"
            # Initialize an empty list to hold the one-hot vectors for the whole batch
            all_one_hot_vectors = []
            # Iterate over each batch element getting all the fields for each category
            if fields_type == "inputs":
                fields = self.input_categories
            elif fields_type == "outputs":
                fields = self.output_categories
            else:
                raise ValueError("fields_type must be either 'inputs' or 'outputs'.")
            for batch_element in zip(*batch):
                # Initialize an empty list to hold the one-hot vectors for each batch element
                one_hot_vectors = []
                # Iterate over each input category and its corresponding  field
                for category_id, field_value in enumerate(batch_element):
                    # Get the input category and its corresponding input field index
                    category = fields[category_id]
                    field_idx = self.category_mappings[category][field_value]
                    # Convert the input field index to a one-hot vector
                    one_hot = nn.functional.one_hot(
                        torch.tensor(field_idx),
                        num_classes=len(self.category_mappings[category])
                        )
                    # Append the one-hot vector to the list
                    one_hot_vectors.append(one_hot)
                # Concatenate the one-hot vectors into a single tensor
                batch_element_tensor = torch.cat(one_hot_vectors).to(self.device)
                # Append the tensor to the list of processed inputs
                all_one_hot_vectors.append(batch_element_tensor)
            # Stack the processed inputs into a tensor
            processed = torch.stack(all_one_hot_vectors).to(self.device)
            # Convert the processed inputs to float
            return processed.float()

    def get_batch_loss(self, loss_fn, batch_logits, y, return_y_one_hot=False):
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
            #assert isinstance(loss_fn, torch.nn.CosineEmbeddingLoss), "Output embeddings require CosineEmbeddingLoss."
            indexes_to_embed= []
            for category_id, batch_element in enumerate(y):
                output_category = self.output_categories[category_id]
                indexes = [self.category_mappings[output_category][field] for field in batch_element]
                # consider indexes_to_embed to be a list of tokenized inputs
                indexes_to_embed.append(indexes)
            # each element in the list is a tokenized input for a possible output category
            # each tensor in the list is a list of indexes for one element of the batch
            batch_logits = nn.functional.normalize(batch_logits, p=2, dim=1)
            y_embeddings = self.process_batch_to_embedding(indexes_to_embed, embedding_model=self._output_embedding_model)
            target = torch.ones(batch_logits.size(0), device=self.device)
            loss = loss_fn(batch_logits, y_embeddings, target)
            if return_y_one_hot:
                # if y is to be returned as one hot, we have to do it now
                yp = self.process_batch_to_one_hot(y, fields_type="outputs")
                return loss, yp
            else:
                return loss
        else:
            yp = self.process_batch_to_one_hot(y, fields_type="outputs")
            loss = loss_fn(batch_logits, yp)
        if return_y_one_hot:
            return loss, yp
        else:
            return loss

    def train_loop(self, dataloader, loss_fn, optimizer, scheduler=None):
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
        initial_weights = [param.clone() for param in self.parameters()]
        size = len(dataloader)
        # Set the model to training mode - important for batch normalization and dropout layers
        self.train()
        for batch, (x, y) in enumerate(dataloader):
            # Compute prediction and loss
            batch_logits = self(x)
            loss = self.get_batch_loss(loss_fn, batch_logits, y)
            # Backpropagation - ORDER MATTERS!
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Check if weights have changed
            weights_changed = any([(initial_weights[i] != param).any().item() for i, param in enumerate(self.parameters())])
            if not weights_changed:
                printer.warning(f"Warning: Weights did not change for batch {batch}")

            if batch % 100 == 0 or batch == size - 1:
                loss, current = loss.item(), (batch + 1)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        if scheduler is not None:
            scheduler.step()

    def test_loop(self, dataloader, loss_fn):
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
        num_batches = len(dataloader)
        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            if not self._use_output_embedding:
                correct_positives, false_positives, total_positives = 0, 0, 0
                test_loss = 0
                for x, y in dataloader:
                    # Compute prediction and loss
                    batch_logits = self(x)
                    batch_loss, y_one_hot = self.get_batch_loss(loss_fn, batch_logits, y, return_y_one_hot=True)
                    test_loss += batch_loss.item()
                    # Calculate the relevance (or priority) of each option
                    # Has to be done for each batch element separately
                    for idx, logits in enumerate(batch_logits):
                        # Calculate the probabilities of the model
                        # We interpret the sigmoid as independent probabilities
                        probabilities = torch.sigmoid(logits)
                        # Calculate the estimated correct classification
                        # Get indices where relevance > 0.5 (default threshold)
                        correct_choices, incorrect_choices, real_positives = (
                            self.get_metrics(
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
                precision, recall, f1_score = self.compute_batch_metrics(
                    correct_positives,
                    false_positives,
                    total_positives
                    )
                # Compute the average loss for the batch
                test_loss /= num_batches
            else:
                # Output embeddings - needed to identify relevant category by
                # similarity in the embedded space
                category_embedding = self._output_embedding_model.embedding.weight
                # Normalize the embeddings to avoid exploding gradients
                category_embedding_normalized = nn.functional.normalize(category_embedding, p=2, dim=1)
                # Initialize a dictionary to hold the metrics for each threshold
                thresholds_metrics = {}
                # Iterate over a range of thresholds to find the best one
                for threshold in torch.arange(0.6, 1.0, 0.1):
                    # Initialize the metrics for the threshold
                    correct_positives, false_positives, total_positives = 0, 0, 0
                    test_loss = 0
                    # Iterate over the test dataset
                    for x, y in dataloader:
                        # Compute prediction and loss
                        batch_logits = self(x)
                        # Compute the loss for the batch
                        batch_loss, y_one_hot = self.get_batch_loss(loss_fn, batch_logits, y, return_y_one_hot=True)
                        # Accumulate the loss for the batch
                        test_loss += batch_loss.item()
                        # Calculate the relevance (or priority) of each option
                        # Has to be done for each batch element separately
                        for idx, logits in enumerate(batch_logits):
                            # Normalize the logits to focus on the cosine similarity
                            logits = nn.functional.normalize(logits, p=2, dim=0)
                            # Calculate the cosine similarity between the logits and the category embeddings
                            cos_sim = self.cos(logits.unsqueeze(0), category_embedding_normalized)
                            # Calculate the probabilities of the model
                            # We identify the cosine similarities
                            # between the logits and the category embeddings
                            # as the probabilities. This is not strictly correct
                            # but it is a good approximation
                            probabilities = cos_sim
                            # Calculate the estimated correct classification
                            correct_choices, incorrect_choices, real_positives = (
                                self.get_metrics(
                                    y_one_hot,
                                    idx,
                                    probabilities,
                                    threshold=threshold)
                                    )
                            # Accumulate the metrics for the batch
                            correct_positives += correct_choices
                            false_positives += incorrect_choices
                            total_positives += real_positives
                    # Compute the precision, recall, and F1 score for the batch
                    precision, recall, f1_score = (
                        self.compute_batch_metrics(
                            correct_positives,
                            false_positives,
                            total_positives
                            )
                    )
                    # Store the metrics for the threshold
                    thresholds_metrics[threshold] = {
                        "f1": f1_score, "precision": precision, "recall": recall,
                        "correct": correct_positives, "false_positives": false_positives,
                        "total_positives": total_positives,
                        "test_loss": test_loss/num_batches
                        }
                #
                self._similarity_threshold = max(thresholds_metrics, key=lambda x: thresholds_metrics[x]["f1"])
                printer.info(f"Best threshold: {self._similarity_threshold:.2f}")
                # Get the metrics for the best threshold into an easy-to-access variable
                best_metrics = thresholds_metrics[self._similarity_threshold]
                # Get the metrics for the best threshold to printout
                precision, recall, f1_score, test_loss, correct_positives, false_positives, total_positives = (
                    best_metrics["precision"],
                    best_metrics["recall"],
                    best_metrics["f1"],
                    best_metrics["test_loss"],
                    best_metrics["correct"],
                    best_metrics["false_positives"],
                    best_metrics["total_positives"]
                    )
        # Print the results
        message_printout = (
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
        if f1_score > 0.9:
            raise StopTraining("F1 score is above 0.9. Stopping training.")


    def compute_batch_metrics(self, correct_positives, false_positives, total_positives):
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
            precision = correct_positives / (correct_positives + false_positives)
        except ZeroDivisionError:
            precision = 0
        try:
            recall = correct_positives / total_positives
        except ZeroDivisionError:
            recall = 0
        try:
            # compute F1 score
            f1_score = 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            f1_score = 0
        return precision, recall, f1_score

    def get_metrics(self, y_one_hot, idx, probabilities, threshold=0.5):
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
        prb_indices = (probabilities > threshold).nonzero().squeeze().tolist()
                    # Get indices where y != 0. This is the correct classification
        y_indices = (y_one_hot[idx]).nonzero().squeeze().tolist()
                    # Check if the indices are integers
                    # If so, convert them to 1-element lists
        if isinstance(prb_indices, int):
            prb_indices = [prb_indices]
        if isinstance(y_indices, int):
            y_indices = [y_indices]
                    # Convert indices to sets for comparison
        prb_set = set(prb_indices)
        y_set = set(y_indices)
                    # Find the intersection of the sets
        correct_choices = len(prb_set.intersection(y_set))
        incorrect_choices = len(prb_set.difference(y_set))
        total_positives = len(y_indices)
        return correct_choices, incorrect_choices, total_positives

    @staticmethod
    def assess_device():
        """
        Determines the device to be used for computation.

        Returns:
            device (str): The device to be used for computation.
        """
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        logger.info(f"Using {device} device")
        return device

    @property
    def device(self):
        """
        The device used for training the model.

        Returns:
            device (torch.device): The device used for training the model.
        """
        return self._device

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

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """
        return nn.functional.normalize(self.embedding(x), p=2, dim=1)

def dot_product(x, y):
    return torch.sum(x * y, dim=1)

if __name__ == "__main__":
    # Do nothing
    # This is just a package for definitions
    pass
