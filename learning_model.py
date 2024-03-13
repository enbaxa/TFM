"""
This module contains the definition models for the neural network.
"""
import logging
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel


logger = logging.getLogger("TFM")


class CategoricNeuralNetwork(nn.Module):
    """
    Neural Network model for classification for a single category.

    Args:
        input_parameters (int): Number of input parameters.
        output_parameters (int): Number of output parameters.

    Attributes:
        linear_relu_stack (nn.Sequential): Sequential module containing linear and ReLU layers.

    Methods:
        forward(x): Performs forward pass through the network.

    """

    def __init__(self, dataset, hidden_neurons=None, use_embedding=True):
        super(CategoricNeuralNetwork, self).__init__()
        self.use_embedding = use_embedding
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
        self.embedding_model = AutoModel.from_pretrained('microsoft/codebert-base')
        self.device = dataset.device
        input_size = self.embedding_model.config.hidden_size * dataset.number_input_categories
        #input_size = 256
        output_size = dataset.number_output_categories
        if not hidden_neurons:
            logging.debug("Estimating the number of neurons in the hidden layers.")
            neurons = 0
            i = 1
            while neurons < (input_size + output_size) // 1:
                neurons = 2**i
                i += 1
            # hardcoded limit to avoid memory issues
            neurons = min(neurons, 128)
        else:
            neurons = hidden_neurons
        logger.info(
            f"Using a neural network {neurons} neurons in the hidden layers.\n"
            f"Mapping to {output_size} output categories.")
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, neurons),
            nn.ReLU(),
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            nn.Linear(neurons, output_size),
        )
        # Proceed to initialize the weights and biases
        self._initialize_weights()

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
        embeddings = []
        for key, text_list in inputs.items():
            tokenized = self.tokenizer(text_list, return_tensors="pt", padding="max_length", truncation=True, max_length=32)
            input_ids, attention_mask = tokenized['input_ids'].to(self.device), tokenized['attention_mask'].to(self.device)
            #with torch.no_grad():
            output = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = torch.mean(output.last_hidden_state, dim=1)
            embeddings.append(pooled)
        # Assuming simple concatenation of embeddings; adjust based on your architecture
        concatenated_embeddings = torch.cat(embeddings, dim=1)
        logits = self.linear_relu_stack(concatenated_embeddings)
        return logits

        # With the following code use the input_ids directly
        for key, text_list in inputs.items():
            tokenized = self.tokenizer(text_list, return_tensors="pt", padding="max_length", truncation=True, max_length=256)
            input_ids, attention_mask = tokenized['input_ids'].to(self.device), tokenized['attention_mask'].to(self.device)
            embeddings.append(input_ids)
        # Assuming simple concatenation of embeddings; adjust based on your architecture
        concatenated_embeddings = torch.cat(embeddings, dim=1).float()
        logits = self.linear_relu_stack(concatenated_embeddings)
        return logits

    def train_loop(self, dataloader, loss_fn, optimizer):
        """
        Trains the model using the given dataloader, loss function, and optimizer.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader containing the training dataset.
            loss_fn (torch.nn.Module): The loss function used to compute the loss.
            optimizer (torch.optim.Optimizer): The optimizer used to update the model's parameters.

        Returns:
            None
        """
        size = len(dataloader.dataset)
        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        self.train()
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = self(X)
            y = y.to(self.device)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

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
        test_loss, positives, false_positives = 0, 0, 0
        total_positives = 0

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            for X, y in dataloader:
                pred = self(X)
                y = y.to(self.device)
                test_loss += loss_fn(pred, y).item()
                # Calculate the relevance (or priority) of each option
                for idx, instance in enumerate(pred):
                    probabilities = torch.sigmoid(instance)
                    # Calculate the estimated correct classification
                    # Get indices where relevance > 0.5
                    prb_indices = (probabilities > 0.5).nonzero().squeeze().tolist()
                    # Get indices where y != 0
                    y_indices = (y[idx] != 0).nonzero().squeeze().tolist()
                    # Convert indices to sets for comparison
                    if isinstance(prb_indices, int):
                        prb_indices = [prb_indices]
                    if isinstance(y_indices, int):
                        y_indices = [y_indices]
                    # Convert indices to sets for comparison
                    prb_set = set(prb_indices)
                    y_set = set(y_indices)
                    # Find the intersection of the sets
                    estimated_correct = len(prb_set.intersection(y_set))
                    incorrect_choices = len(prb_set.difference(y_set))
                    # The count of values correctly reurned as positive
                    positives += estimated_correct
                    # Count of values that wrongly returned as positive
                    false_positives += incorrect_choices
                    # Count of total values that should return as positive
                    total_positives += len(y_indices)

        test_loss /= num_batches
        correct_proportion = positives / total_positives
        print(f"Test Error: \n Accuracy: {(100*correct_proportion):>0.1f}%, Avg loss: {test_loss:>8f} ")
        print(f"Correct: {positives:>0.1f} / {total_positives}, False Positives: {false_positives:>0.1f}\n")


if __name__ == "__main__":
    # Do nothing
    # This is just a package for definitions
    pass
