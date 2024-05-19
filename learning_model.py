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

    def __init__(self, dataset, hidden_neurons=None, use_input_embedding=True, use_output_embedding=False):
        super(CategoricNeuralNetwork, self).__init__()
        self._device = dataset.device
        self._use_input_embedding = use_input_embedding
        self._use_output_embedding = use_output_embedding
        self.dataset = dataset
        self.category_mappings = dataset.category_mappings
        if self._use_input_embedding:
            #self._tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
            #self._input_embedding_model = AutoModel.from_pretrained('microsoft/codebert-base').to(self._device)
            # maybe use distilbert-base-uncased for faster training
            self._tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            self._input_embedding_model = AutoModel.from_pretrained('distilbert-base-uncased').to(self._device)
            self._input_size = self._input_embedding_model.config.hidden_size * dataset.number_input_categories
            for param in self._input_embedding_model.parameters():
                # Letting the embedding train makes it difficult to train the model
                param.requires_grad = False
        else:
            self._input_size = sum([len(dataset.category_mappings[col]) for col in dataset.input_columns])
        if self._use_output_embedding:
            # Take a naive guess that half the dimensions should be enough
            self._output_size = int(dataset.number_output_categories / 2)
            printer.info(
                f"Using output embeddings. "
                f"Condensing the {dataset.number_output_categories} output "
                f"categories to a lower-dimensional space "
                f"of {self._output_size} dimensions."
                )
            printer.warning(
                "Output embeddings imply loss of precision due to dimensionality reduction.\n "
                "Also the logits are unstable and may not be suitable for training.\n"
                )
            self._output_embedding_model = EmbeddingModel(dataset.number_output_categories, self._output_size).to(self._device)
            #for param in self._output_embedding_model.parameters():
            #    param.requires_grad = False
        else:
            self._output_size = dataset.number_output_categories

        printer.info(f"Input size: {self._input_size}, Output size: {self._output_size}")
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
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self._input_size, neurons),
            nn.LeakyReLU(),
            nn.Linear(neurons, neurons),
            nn.LeakyReLU(),
            # nn.Linear(neurons, neurons),
            # nn.LeakyReLU(),
            nn.Linear(neurons, self._output_size)
        )
        # Proceed to initialize the weights and biases
        self._initialize_weights()
        self.softmax = nn.Softmax(dim=-1)
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

    def forward(self, inputs):
        """
        Performs forward pass through the network.

        Args:
            inputs (dict): A dictionary containing the input data.

        Returns:
            logits (torch.Tensor): The output of the network.
        """
        if self._use_input_embedding:
            pre_processed_inputs = self.preprocess_batch_with_input_embedding(inputs)
        else:
            # No embeddings
            pre_processed_inputs = self.preprocess_batch_without_input_embedding(inputs)
        logits = self.linear_relu_stack(pre_processed_inputs)
        return logits

    def preprocess_batch_with_input_embedding(self, inputs):
            """
            Preprocesses the batch of inputs using embeddings.

            Args:
                inputs (list of tuples): A dictionary containing the input data.

            Returns:
                concatenated_embeddings (torch.Tensor): The concatenated embeddings.
            """
            # Get the length of the lists in the input dictionary
            lengths = set([len(v) for v in inputs])
            assert len(lengths) == 1, "Not all input fields have the same length in this batch"

            # Initialize an empty list to hold the embeddings
            embeddings = []
            # With the following code use the embeddings
            for inp in zip(*inputs):
                # For each index, get the text from each list in the input dictionary
                tokenized = self._tokenizer(inp, return_tensors="pt", padding="max_length", truncation=True, max_length=16)
                input_ids, attention_mask = tokenized['input_ids'].to(self._device), tokenized['attention_mask'].to(self._device)
                output = self._input_embedding_model(input_ids=input_ids, attention_mask=attention_mask)
                # Choose a pooling strategy
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
                # Normalize the embedding to avoid exploding gradients
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                pooled = pooled.reshape(-1)
                embeddings.append(pooled)
            # Stack the embeddings into a tensor
            stacked_embeddings = torch.stack(embeddings).to(self._device)
            return stacked_embeddings

    def preprocess_batch_without_input_embedding(self, inputs):
            """
            Preprocesses the batch of inputs without using embeddings.

            Args:
                inputs (list of tuples): A dictionary containing the input data.

            Returns:
                processed_inputs (torch.Tensor): The processed inputs.
                                                 (one long multi-hot encoded tensor)
            """
            # Get the lengths of the lists in the input dictionary
            lengths = set([len(v) for v in inputs])
            # Check if all input fields have the same length in this batch
            assert len(lengths) == 1, "Not all input fields have the same length in this batch"
            # Initialize an empty list to hold the one-hot vectors
            all_one_hot_vectors = []
            # Iterate over each input field in the batch
            for inp in zip(*inputs):
                # Initialize an empty list to hold the one-hot vectors for each input category
                one_hot_vectors = []
                # Iterate over each input category and its corresponding input field
                for input_category_id, inp_field in enumerate(inp):
                    # Get the input category and its corresponding input field index
                    input_category = self.dataset.input_columns[input_category_id]
                    inp_field_idx = self.category_mappings[input_category][inp_field]
                    # Convert the input field index to a one-hot vector
                    one_hot = nn.functional.one_hot(torch.tensor(inp_field_idx), num_classes=len(self.category_mappings[input_category]))
                    # Append the one-hot vector to the list
                    one_hot_vectors.append(one_hot)
                # Concatenate the one-hot vectors into a single tensor
                inp_tensor = torch.cat(one_hot_vectors).to(self._device)
                # Append the tensor to the list of processed inputs
                all_one_hot_vectors.append(inp_tensor)
            # Stack the processed inputs into a tensor
            processed_inputs = torch.stack(all_one_hot_vectors).to(self._device)
            # Convert the processed inputs to float
            return processed_inputs.float()

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
            if self._use_output_embedding:
                y = y.to(int)
                batch_logits = nn.functional.normalize(batch_logits, p=2, dim=1)
                assert isinstance(loss_fn, torch.nn.CosineEmbeddingLoss), "Output embeddings require CosineEmbeddingLoss."
                y_embeddings = torch.sum(self._output_embedding_model(y), dim=1)
                target = torch.ones(batch_logits.size(0), device=self._device)
                loss = loss_fn(batch_logits, y_embeddings, target)
            else:
                y = y.to(float)
                loss = loss_fn(batch_logits, y)
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
        test_loss, correct_positives, false_positives = 0, 0, 0
        total_positives = 0

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            for x, y in dataloader:
                batch_logits = self(x)
                if self._use_output_embedding:
                    assert isinstance(loss_fn, torch.nn.CosineEmbeddingLoss), "Output embeddings require CosineEmbeddingLoss."
                    batch_logits = nn.functional.normalize(batch_logits, p=2, dim=1)
                    y = y.to(int)
                    y_embeddings = torch.sum(self._output_embedding_model(y), dim=1)
                    target = torch.ones(batch_logits.size(0), device=self._device)
                    test_loss += loss_fn(batch_logits, y_embeddings, target)
                else:
                    y = y.to(float)
                    test_loss += loss_fn(batch_logits, y)
                # Calculate the relevance (or priority) of each option
                for idx, logits in enumerate(batch_logits):
                    # use sigmoid to rescale values between 0 and 1 - sigmoid approach
                    if self._use_output_embedding:
                        cos_sim = self.compute_cosine_similarity(logits)
                        probabilities = torch.sigmoid(cos_sim)
                    else:
                        probabilities = torch.sigmoid(logits)
                    # softmax approach.
                    #probabilities = self.softmax(instance) / self.softmax(instance).max()
                    # Calculate the estimated correct classification
                    # Get indices where relevance > 0.5
                    prb_indices = (probabilities > 0.5).nonzero().squeeze().tolist()
                    # Get indices where y != 0
                    y_indices = (y[idx] != 0).nonzero().squeeze().tolist()
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
                    # The count of values correctly reurned as positive
                    correct_positives += correct_choices
                    # Count of values that wrongly returned as positive
                    false_positives += incorrect_choices
                    # Count of total values that should return as positive
                    total_positives += len(y_indices)
        test_loss /= num_batches
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

        printer.info(
            f"\nTest Result:\nPrecision: {(precision):>0.4f}, Recall: {(recall):>0.4f}, Avg loss: {test_loss:>8f} "
            f"\nF1 Score: {f1_score:>12.8f}"
            f"\nCorrect: {correct_positives:>1.0f} / {total_positives}, False Positives: {false_positives:>1.0f}\n"
            )
        if f1_score > 0.9:
            raise StopTraining("F1 score is above 0.9. Stopping training.")

    def compute_cosine_similarity(self, logits):
        """
        Computes the cosine similarity between the logits and the output embeddings.

        Args:
            logits (torch.Tensor): The logits from the model.

        Returns:
            cos_sim (torch.Tensor): The cosine similarities between the logits and the output embeddings.
        """
        num_classes = self.dataset.number_output_categories
        embedded_base = self._output_embedding_model(
            torch.arange(num_classes).to(self._device)
            )

        cos_sim = nn.functional.cosine_similarity(
            logits,
            embedded_base)
        return cos_sim

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
        super(EmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(num_categories, embedding_dim)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """
        return self.embedding(x)

if __name__ == "__main__":
    # Do nothing
    # This is just a package for definitions
    pass
