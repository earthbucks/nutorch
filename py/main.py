import torch

torch.manual_seed(42)  # For reproducibility

# Rest of the script remains mostly the same
def generate_data(n_samples=300, centers=3, cluster_std=0.7, skew_factor=0.3):
    n_samples_per_class = n_samples // centers
    X_list = []
    y_list = []

    blob_centers = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([3.0, 0.0]),
        torch.tensor([1.5, 2.5])
    ]

    for i in range(centers):
        points = torch.randn(n_samples_per_class, 2) * cluster_std + blob_centers[i]
        if i == 1 or i == 2:
            skew_matrix = torch.tensor([[1.0, skew_factor * (i-1)], [skew_factor * (i-1), 1.0]])
            points = torch.mm(points - blob_centers[i], skew_matrix) + blob_centers[i]
        labels = torch.full((n_samples_per_class,), i, dtype=torch.long)
        X_list.append(points)
        y_list.append(labels)

    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    return X, y

# Simplified two-layer neural network for multi-class classification
class SimpleNN:
    def __init__(self, input_size=2, hidden_size=20, output_size=3):
        # Initialize weights and biases directly with requires_grad=True for gradient tracking
        self.weight1 = torch.randn(hidden_size, input_size, requires_grad=True)
        self.bias1 = torch.randn(hidden_size, requires_grad=True)
        self.weight2 = torch.randn(output_size, hidden_size, requires_grad=True)
        self.bias2 = torch.randn(output_size, requires_grad=True)
        # Store parameters in a list for optimization
        self.parameters_list = [self.weight1, self.bias1, self.weight2, self.bias2]

    def parameters(self):
        """Return all parameters for optimization"""
        return self.parameters_list

    def forward(self, x):
        # First layer (linear transformation)
        x = torch.mm(x, self.weight1.t()) + self.bias1
        # ReLU activation
        x = torch.max(torch.tensor(0.0), x)
        # Second layer (linear transformation)
        x = torch.mm(x, self.weight2.t()) + self.bias2
        return x  # Raw logits for CrossEntropyLoss

    def __call__(self, x):
        """Make instance callable by delegating to forward method"""
        return self.forward(x)

# Primitive version of CrossEntropyLoss
def primitive_cross_entropy_loss(outputs, targets):
    """
    Compute cross-entropy loss for multi-class classification.
    outputs: Raw logits from the model (shape: [N, C] where N is batch size, C is number of classes)
    targets: Class indices (shape: [N], values in range [0, C-1])
    """
    log_probs = torch.log_softmax(outputs, dim=1)
    batch_size = outputs.size(0)
    loss = 0.0
    for i in range(batch_size):
        loss -= log_probs[i, targets[i]]
    loss = loss / batch_size
    return loss

def train_model(model, X, y, epochs=1000, lr=0.1, record_interval=100):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    losses = []
    epoch_steps = []
    for epoch in range(epochs):
        outputs = model(X)
        loss = primitive_cross_entropy_loss(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Store loss and corresponding epoch at the specified interval
        if (epoch + 1) % record_interval == 0:
            losses.append(loss.item())
            epoch_steps.append(epoch + 1)
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    return losses, epoch_steps

def plot_raw_data(X, y):
    X_list = X.detach().tolist()
    y_list = y.detach().tolist()
    plt.scatter([x[0] for x in X_list], [x[1] for x in X_list], c=y_list, alpha=0.8, cmap='viridis')
    plt.title("Raw Data Points (Before Training) - Three Blobs")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(label='Class')
    plt.show()

def plot_results(X, y, model):
    X_list = X.detach().tolist()
    y_list = y.detach().tolist()
    x_min = min(x[0] for x in X_list) - 1
    x_max = max(x[0] for x in X_list) + 1
    y_min = min(x[1] for x in X_list) - 1
    y_max = max(x[1] for x in X_list) + 1
    step = 0.1
    x_range = torch.arange(x_min, x_max, step)
    y_range = torch.arange(y_min, y_max, step)
    xx, yy = torch.meshgrid(x_range, y_range, indexing='xy')
    mesh_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    with torch.no_grad():
        outputs = model(mesh_points)
        _, Z = torch.max(outputs, dim=1)
        Z = Z.reshape(xx.shape)
    xx_list = xx.tolist()
    yy_list = yy.tolist()
    Z_list = Z.tolist()
    plt.contourf(xx_list, yy_list, Z_list, alpha=0.4, cmap='viridis')
    plt.scatter([x[0] for x in X_list], [x[1] for x in X_list], c=y_list, alpha=0.8, cmap='viridis')
    plt.title("Two-Layer Neural Network Decision Boundaries (After Training)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(label='Class')
    plt.show()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    X, y = generate_data(n_samples=300, centers=3, cluster_std=0.7, skew_factor=0.3)
    print("Data shape:", X.shape, y.shape)
    plot_raw_data(X, y)
    model = SimpleNN(input_size=2, hidden_size=20, output_size=3)
    print("Model architecture:\n", model)
    print("Checking if model parameters require gradients for backpropagation:")
    for i, param in enumerate(model.parameters()):
        print(f"Parameter {i}: shape {param.shape}, requires_grad: {param.requires_grad}")
    # Example with increased epochs
    epochs = 3000
    record_interval = 100
    losses, epoch_steps = train_model(model, X, y, epochs=epochs, lr=0.1, record_interval=record_interval)
    plt.plot(epoch_steps, losses)
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    plot_results(X, y, model)
