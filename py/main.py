import torch
from typing import List, Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)  # type: ignore


def generate_data(
    n_samples: int = 300,
    centers: int = 3,
    cluster_std: float = 0.7,
    skew_factor: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic data with Gaussian blobs arranged triangularly.

    Args:
        n_samples (int): Total number of samples to generate.
        centers (int): Number of blob centers (classes).
        cluster_std (float): Standard deviation for Gaussian blobs.
        skew_factor (float): Factor to skew some blobs for visual distinction.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Features (X) and labels (y) as tensors.
    """
    n_samples_per_class: int = n_samples // centers
    X_list: List[torch.Tensor] = []
    y_list: List[torch.Tensor] = []

    blob_centers: List[torch.Tensor] = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([3.0, 0.0]),
        torch.tensor([1.5, 2.5]),
    ]

    for i in range(centers):
        points: torch.Tensor = (
            torch.randn(n_samples_per_class, 2) * cluster_std + blob_centers[i]
        )
        if i == 1 or i == 2:
            skew_matrix: torch.Tensor = torch.tensor(
                [[1.0, skew_factor * (i - 1)], [skew_factor * (i - 1), 1.0]]
            )
            points = torch.mm(points - blob_centers[i], skew_matrix) + blob_centers[i]
        labels: torch.Tensor = torch.full((n_samples_per_class,), i, dtype=torch.long)
        X_list.append(points)
        y_list.append(labels)

    X: torch.Tensor = torch.cat(X_list, dim=0)
    y: torch.Tensor = torch.cat(y_list, dim=0)
    return X, y


# Simplified two-layer neural network for multi-class classification
class SimpleNN:
    def __init__(
        self, input_size: int = 2, hidden_size: int = 20, output_size: int = 3
    ) -> None:
        """
        Initialize a simple two-layer neural network.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of neurons in the hidden layer.
            output_size (int): Number of output classes.
        """
        # Initialize weights and biases directly with requires_grad=True for gradient tracking
        self.weight1: torch.Tensor = torch.randn(
            hidden_size, input_size, requires_grad=True
        )
        self.bias1: torch.Tensor = torch.randn(hidden_size, requires_grad=True)
        self.weight2: torch.Tensor = torch.randn(
            output_size, hidden_size, requires_grad=True
        )
        self.bias2: torch.Tensor = torch.randn(output_size, requires_grad=True)
        # Store parameters in a list for optimization
        self.parameters_list: List[torch.Tensor] = [
            self.weight1,
            self.bias1,
            self.weight2,
            self.bias2,
        ]

    def parameters(self) -> List[torch.Tensor]:
        """
        Return all parameters for optimization.

        Returns:
            List[torch.Tensor]: List of parameter tensors.
        """
        return self.parameters_list

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size) with raw logits.
        """
        # First layer (linear transformation)
        x = torch.mm(x, self.weight1.t()) + self.bias1
        # ReLU activation
        x = torch.max(torch.tensor(0.0), x)
        # Second layer (linear transformation)
        x = torch.mm(x, self.weight2.t()) + self.bias2
        return x  # Raw logits for CrossEntropyLoss

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make instance callable by delegating to forward method.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        return self.forward(x)


def primitive_cross_entropy_loss(
    outputs: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """
    Compute cross-entropy loss for multi-class classification.

    Args:
        outputs (torch.Tensor): Raw logits from the model, shape (N, C) where N is batch size, C is number of classes.
        targets (torch.Tensor): Class indices, shape (N), values in range [0, C-1].

    Returns:
        torch.Tensor: Scalar loss value.
    """
    log_probs: torch.Tensor = torch.log_softmax(outputs, dim=1)
    batch_size: int = outputs.size(0)
    # Select log probabilities for the target classes using indexing
    selected_log_probs: torch.Tensor = log_probs[range(batch_size), targets]
    # Compute the mean negative log likelihood as loss
    loss: torch.Tensor = -selected_log_probs.mean()
    return loss


def train_model(
    model: SimpleNN,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 1000,
    lr: float = 0.1,
    record_interval: int = 100,
) -> Tuple[List[float], List[int]]:
    """
    Train the neural network model.

    Args:
        model (SimpleNN): The neural network model to train.
        X (torch.Tensor): Input features tensor of shape (N, input_size).
        y (torch.Tensor): Target labels tensor of shape (N,).
        epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.
        record_interval (int): Interval at which to record loss values.

    Returns:
        Tuple[List[float], List[int]]: List of recorded loss values and corresponding epoch numbers.
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    losses: List[float] = []
    epoch_steps: List[int] = []
    for epoch in range(epochs):
        outputs: torch.Tensor = model(X)
        loss: torch.Tensor = primitive_cross_entropy_loss(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Store loss and corresponding epoch at the specified interval
        if (epoch + 1) % record_interval == 0:
            losses.append(loss.item())
            epoch_steps.append(epoch + 1)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    return losses, epoch_steps


def plot_raw_data(X: torch.Tensor, y: torch.Tensor) -> None:
    """
    Plot the raw data points before training.

    Args:
        X (torch.Tensor): Input features tensor of shape (N, 2).
        y (torch.Tensor): Target labels tensor of shape (N,).
    """
    X_list: List[List[float]] = X.detach().tolist()
    y_list: List[int] = y.detach().tolist()
    plt.scatter(
        [x[0] for x in X_list],
        [x[1] for x in X_list],
        c=y_list,
        alpha=0.8,
        cmap="viridis",
    )
    plt.title("Raw Data Points (Before Training) - Three Blobs")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(label="Class")
    plt.show()


def plot_results(X: torch.Tensor, y: torch.Tensor, model: SimpleNN) -> None:
    """
    Plot the decision boundaries and data points after training.

    Args:
        X (torch.Tensor): Input features tensor of shape (N, 2).
        y (torch.Tensor): Target labels tensor of shape (N,).
        model (SimpleNN): Trained neural network model.
    """
    X_list: List[List[float]] = X.detach().tolist()
    y_list: List[int] = y.detach().tolist()
    x_min: float = min(x[0] for x in X_list) - 1
    x_max: float = max(x[0] for x in X_list) + 1
    y_min: float = min(x[1] for x in X_list) - 1
    y_max: float = max(x[1] for x in X_list) + 1
    step: float = 0.1
    x_range: torch.Tensor = torch.arange(x_min, x_max, step)
    y_range: torch.Tensor = torch.arange(y_min, y_max, step)
    xx, yy = torch.meshgrid(x_range, y_range, indexing="xy")
    mesh_points: torch.Tensor = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    with torch.no_grad():
        outputs: torch.Tensor = model(mesh_points)
        _, Z = torch.max(outputs, dim=1)
        Z = Z.reshape(xx.shape)
    xx_list: List[List[float]] = xx.tolist()
    yy_list: List[List[float]] = yy.tolist()
    Z_list: List[List[int]] = Z.tolist()
    plt.contourf(xx_list, yy_list, Z_list, alpha=0.4, cmap="viridis")
    plt.scatter(
        [x[0] for x in X_list],
        [x[1] for x in X_list],
        c=y_list,
        alpha=0.8,
        cmap="viridis",
    )
    plt.title("Two-Layer Neural Network Decision Boundaries (After Training)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(label="Class")
    plt.show()


if __name__ == "__main__":
    X, y = generate_data(n_samples=300, centers=3, cluster_std=0.7, skew_factor=0.3)
    print("Data shape:", X.shape, y.shape)
    plot_raw_data(X, y)
    model = SimpleNN(input_size=2, hidden_size=20, output_size=3)
    print("Model architecture:\n", model)
    print("Checking if model parameters require gradients for backpropagation:")
    for i, param in enumerate(model.parameters()):
        print(
            f"Parameter {i}: shape {param.shape}, requires_grad: {param.requires_grad}"
        )
    # Example with increased epochs
    epochs: int = 3000
    record_interval: int = 100
    losses, epoch_steps = train_model(
        model, X, y, epochs=epochs, lr=0.1, record_interval=record_interval
    )
    plt.plot(epoch_steps, losses)
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    plot_results(X, y, model)
