import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Step 1: Generate synthetic data (two Gaussian blobs) using pure PyTorch
def generate_data(n_samples=200, centers=2, cluster_std=1.0, skew_factor=0.5):
    n_samples_per_class = n_samples // centers
    X_list = []
    y_list = []

    # Define centers for two blobs
    blob_centers = [
        torch.tensor([0.0, 0.0]),  # Center for class 0
        torch.tensor([3.0, 3.0])   # Center for class 1
    ]

    for i in range(centers):
        # Generate points from a Gaussian distribution around the center
        points = torch.randn(n_samples_per_class, 2) * cluster_std + blob_centers[i]
        # Optionally apply a skew transformation (linear transformation)
        if i == 1:  # Skew the second blob for visual distinction
            skew_matrix = torch.tensor([[1.0, skew_factor], [skew_factor, 1.0]])
            points = torch.mm(points - blob_centers[i], skew_matrix) + blob_centers[i]
        labels = torch.full((n_samples_per_class, 1), float(i))
        X_list.append(points)
        y_list.append(labels)

    # Concatenate the data from both classes
    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    return X, y

# Step 2: Define the two-layer neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=10, output_size=1):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x

# Step 3: Training function
def train_model(model, X, y, epochs=1000, lr=0.01):
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
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
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    return losses

# Step 4: Plotting function for raw data (before training)
def plot_raw_data(X, y):
    # Convert tensors to numpy for plotting
    X_np = X.detach().numpy()
    y_np = y.detach().numpy().flatten()

    # Plot data points with different colors for each class
    plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np, alpha=0.8)
    plt.title("Raw Data Points (Before Training)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Step 5: Plotting function for data and decision boundary (after training)
def plot_results(X, y, model):
    # Convert tensors to numpy for plotting
    X_np = X.detach().numpy()
    y_np = y.detach().numpy().flatten()

    # Create a mesh grid for decision boundary
    x_min, x_max = X_np[:, 0].min() - 1, X_np[:, 0].max() + 1
    y_min, y_max = X_np[:, 1].min() - 1, X_np[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    # Predict over the mesh grid
    mesh_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    with torch.no_grad():
        Z = model(mesh_tensor)
        Z = (Z > 0.5).float().numpy().reshape(xx.shape)

    # Plot decision boundary and data points
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np, alpha=0.8)
    plt.title("Two-Layer Neural Network Decision Boundary (After Training)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Main execution
if __name__ == "__main__":
    # Generate data
    X, y = generate_data(n_samples=200, centers=2, cluster_std=1.0, skew_factor=0.5)
    print("Data shape:", X.shape, y.shape)

    # Plot raw data before training
    plot_raw_data(X, y)

    # Initialize model
    model = SimpleNN(input_size=2, hidden_size=10, output_size=1)
    print("Model architecture:\n", model)

    # Train model
    losses = train_model(model, X, y, epochs=1000, lr=0.1)

    # Plot loss curve
    plt.plot(range(100, 1001, 100), losses)
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # Plot decision boundary and data points after training
    plot_results(X, y, model)
