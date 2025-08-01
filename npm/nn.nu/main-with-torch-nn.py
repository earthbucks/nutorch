import torch
import torch.nn as nn
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


# Step 2: Define the two-layer neural network for multi-class classification
class SimpleNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=20, output_size=3):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        # No sigmoid; we'll use softmax in loss (CrossEntropyLoss applies it internally)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x  # Raw logits for CrossEntropyLoss


# Step 3: Training function for multi-class
def train_model(model, X, y, epochs=1000, lr=0.1):
    criterion = nn.CrossEntropyLoss()  # Cross Entropy Loss for multi-class
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Store loss for plotting
        if (epoch + 1) % 100 == 0:
            losses.append(loss.item())
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

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
