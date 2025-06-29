import torch
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)


# Step 1: Generate synthetic data (three Gaussian blobs in a triangular arrangement)
def generate_data(n_samples=300, centers=3, cluster_std=0.7, skew_factor=0.3):
    n_samples_per_class = n_samples // centers
    X_list = []
    y_list = []

    # Define centers for three blobs in a triangular arrangement
    blob_centers = [
        torch.tensor([0.0, 0.0]),  # Center for class 0 (bottom left)
        torch.tensor([3.0, 0.0]),  # Center for class 1 (bottom right)
        torch.tensor([1.5, 2.5]),  # Center for class 2 (top middle)
    ]

    for i in range(centers):
        # Generate points from a Gaussian distribution around the center
        points = torch.randn(n_samples_per_class, 2) * cluster_std + blob_centers[i]
        # Apply a slight skew transformation for visual distinction
        if i == 1 or i == 2:  # Skew the second and third blobs
            skew_matrix = torch.tensor(
                [[1.0, skew_factor * (i - 1)], [skew_factor * (i - 1), 1.0]]
            )
            points = torch.mm(points - blob_centers[i], skew_matrix) + blob_centers[i]
        labels = torch.full((n_samples_per_class,), i, dtype=torch.long)
        X_list.append(points)
        y_list.append(labels)

    # Concatenate the data from all classes
    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    return X, y


# Primitive version of nn.Module
class BaseModule:
    def __init__(self):
        self._parameters = {}  # Dictionary to store trainable parameters
        self._modules = {}  # Dictionary to store sub-modules

    def register_parameter(self, name, param):
        """Register a parameter for gradient tracking"""
        self._parameters[name] = param

    def register_module(self, name, module):
        """Register a sub-module (e.g., layer)"""
        self._modules[name] = module

    def parameters(self):
        """Return all parameters for optimization"""
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def forward(self, x):
        """To be overridden by subclasses"""
        raise NotImplementedError("Forward method must be implemented in subclass")

    def __call__(self, *args, **kwargs):
        """Make instances callable by delegating to forward method"""
        return self.forward(*args, **kwargs)


# Primitive version of nn.Linear (fully connected layer)
class PrimitiveLinear(BaseModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Initialize weights and biases with requires_grad=True for gradient tracking
        self.weight = torch.randn(out_features, in_features, requires_grad=True)
        self.bias = torch.randn(out_features, requires_grad=True)
        # Register parameters
        self.register_parameter("weight", self.weight)
        self.register_parameter("bias", self.bias)

    def forward(self, x):
        # Matrix multiplication: output = x @ weight.T + bias
        return torch.mm(x, self.weight.t()) + self.bias


# Primitive version of nn.ReLU (activation function)
class PrimitiveReLU(BaseModule):
    def __init__(self):
        super().__init__()
        # No parameters to register for ReLU

    def forward(self, x):
        # ReLU activation: max(0, x)
        return torch.max(torch.tensor(0.0), x)


# Step 2: Define the two-layer neural network for multi-class classification
class SimpleNN(BaseModule):
    def __init__(self, input_size=2, hidden_size=20, output_size=3):
        super().__init__()
        self.layer1 = PrimitiveLinear(input_size, hidden_size)
        self.relu = PrimitiveReLU()
        self.layer2 = PrimitiveLinear(hidden_size, output_size)
        # Register sub-modules
        self.register_module("layer1", self.layer1)
        self.register_module("relu", self.relu)
        self.register_module("layer2", self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x  # Raw logits for CrossEntropyLoss


# Primitive version of CrossEntropyLoss
def primitive_cross_entropy_loss(outputs, targets):
    """
    Compute cross-entropy loss for multi-class classification.
    outputs: Raw logits from the model (shape: [N, C] where N is batch size, C is number of classes)
    targets: Class indices (shape: [N], values in range [0, C-1])
    """
    # Apply log softmax to convert raw logits to log probabilities
    log_probs = torch.log_softmax(outputs, dim=1)
    # Compute negative log likelihood loss
    # Use gather to select log probabilities corresponding to target indices
    batch_size = outputs.size(0)
    loss = 0.0
    for i in range(batch_size):
        loss -= log_probs[i, targets[i]]
    # Average loss over the batch
    loss = loss / batch_size
    return loss


# Step 3: Training function for multi-class
def train_model(model, X, y, epochs=1000, lr=0.1):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        outputs = model(X)
        loss = primitive_cross_entropy_loss(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            losses.append(loss.item())
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    return losses


# Step 4: Plotting function for raw data (before training)
def plot_raw_data(X, y):
    # Convert tensors to lists for plotting with matplotlib
    X_list = X.detach().tolist()
    y_list = y.detach().tolist()

    # Plot data points with different colors for each class
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


# Step 5: Plotting function for data and decision boundaries (after training)
def plot_results(X, y, model):
    # Convert tensors to lists for plotting
    X_list = X.detach().tolist()
    y_list = y.detach().tolist()

    # Get bounds for mesh grid
    x_min = min(x[0] for x in X_list) - 1
    x_max = max(x[0] for x in X_list) + 1
    y_min = min(x[1] for x in X_list) - 1
    y_max = max(x[1] for x in X_list) + 1

    # Create mesh grid using PyTorch
    step = 0.1
    x_range = torch.arange(x_min, x_max, step)
    y_range = torch.arange(y_min, y_max, step)
    xx, yy = torch.meshgrid(x_range, y_range, indexing="xy")
    mesh_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    # Predict over the mesh grid
    with torch.no_grad():
        outputs = model(mesh_points)
        _, Z = torch.max(outputs, dim=1)  # Get class with highest probability
        Z = Z.reshape(xx.shape)

    # Convert mesh results to lists for plotting
    xx_list = xx.tolist()
    yy_list = yy.tolist()
    Z_list = Z.tolist()

    # Plot decision boundaries and data points
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


# Main execution
if __name__ == "__main__":
    # Generate data
    X, y = generate_data(n_samples=300, centers=3, cluster_std=0.7, skew_factor=0.3)
    print("Data shape:", X.shape, y.shape)

    # Plot raw data before training
    plot_raw_data(X, y)

    # Initialize model
    model = SimpleNN(input_size=2, hidden_size=20, output_size=3)
    print("Model architecture:\n", model)

    # Train model
    losses = train_model(model, X, y, epochs=1000, lr=0.1)

    # Plot loss curve
    plt.plot(range(100, 1001, 100), losses)
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # Plot decision boundaries and data points after training
    plot_results(X, y, model)
