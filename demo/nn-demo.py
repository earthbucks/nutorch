import torch
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt

torch.manual_seed(42)  # reproducibility


def generate_data(
    n_samples: int = 300,
    centers: int = 3,
    cluster_std: float = 0.7,
    skew_factor: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic data with Gaussian blobs arranged triangularly.
    """
    n_samples_per_class: int = n_samples // centers
    X_parts, y_parts = [], []

    blob_centers = [
        torch.tensor([0.0, 0.0]),
        torch.tensor([3.0, 0.0]),
        torch.tensor([1.5, 2.5]),
    ]

    for i in range(centers):
        pts = torch.randn(n_samples_per_class, 2) * cluster_std + blob_centers[i]
        if i in (1, 2):
            skew = torch.tensor(
                [[1.0, skew_factor * (i - 1)], [skew_factor * (i - 1), 1.0]]
            )
            pts = torch.mm(pts - blob_centers[i], skew) + blob_centers[i]
        lbl = torch.full((n_samples_per_class,), i, dtype=torch.long)
        X_parts.append(pts)
        y_parts.append(lbl)

    X = torch.cat(X_parts, dim=0)
    y = torch.cat(y_parts, dim=0)
    return X, y


def plot_raw_data(X: torch.Tensor, y: torch.Tensor) -> None:
    """Scatter-plot the raw blobs."""
    Xl, yl = X.detach().tolist(), y.detach().tolist()
    plt.scatter([p[0] for p in Xl], [p[1] for p in Xl], c=yl, alpha=0.8, cmap="viridis")
    plt.title("Raw Data Points (Before Training) - Three Blobs")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(label="Class")
    plt.show()


Model = Dict[str, torch.Tensor]  # alias for readability


def init_model(
    input_size: int = 2,
    hidden_size: int = 20,
    output_size: int = 3,
) -> Model:
    """Return a dict holding all trainable tensors."""
    return {
        "w1": torch.randn(hidden_size, input_size, requires_grad=True),
        "b1": torch.randn(hidden_size, requires_grad=True),
        "w2": torch.randn(output_size, hidden_size, requires_grad=True),
        "b2": torch.randn(output_size, requires_grad=True),
    }


def get_parameters(model: Model) -> List[torch.Tensor]:
    """Convenience accessor for optimiser."""
    return [model["w1"], model["b1"], model["w2"], model["b2"]]


def forward_pass(model: Model, x: torch.Tensor) -> torch.Tensor:
    """Forward pass producing raw logits."""
    x = torch.mm(x, model["w1"].t()) + model["b1"]  # Linear
    x = torch.max(torch.tensor(0.0), x)  # ReLU
    x = torch.mm(x, model["w2"].t()) + model["b2"]  # Linear
    return x


def primitive_cross_entropy_loss(
    outputs: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy implemented with log-softmax."""
    logp = torch.log_softmax(outputs, dim=1)
    loss = -logp[range(outputs.size(0)), targets].mean()
    return loss


def train_model(
    model: Model,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 1000,
    lr: float = 0.1,
    record_interval: int = 100,
) -> Tuple[List[float], List[int]]:
    """SGD training loop."""
    optim = torch.optim.SGD(get_parameters(model), lr=lr)
    losses, steps = [], []

    for epoch in range(epochs):
        out = forward_pass(model, X)
        loss = primitive_cross_entropy_loss(out, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if (epoch + 1) % record_interval == 0:
            losses.append(loss.item())
            steps.append(epoch + 1)
            print(f"Epoch {epoch+1:4d}/{epochs}, loss {loss.item():.4f}")

    return losses, steps


def plot_results(X: torch.Tensor, y: torch.Tensor, model: Model) -> None:
    """Plot decision boundary after training."""
    Xl, yl = X.detach().tolist(), y.detach().tolist()

    x_min, x_max = min(p[0] for p in Xl) - 1, max(p[0] for p in Xl) + 1
    y_min, y_max = min(p[1] for p in Xl) - 1, max(p[1] for p in Xl) + 1
    xs, ys = torch.arange(x_min, x_max, 0.1), torch.arange(y_min, y_max, 0.1)
    xx, yy = torch.meshgrid(xs, ys, indexing="xy")
    mesh = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    with torch.no_grad():
        logits = forward_pass(model, mesh)
        _, Z = torch.max(logits, dim=1)
        Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4, cmap="viridis")
    plt.scatter([p[0] for p in Xl], [p[1] for p in Xl], c=yl, alpha=0.8, cmap="viridis")
    plt.title("Two-Layer NN Decision Boundary (After Training)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(label="Class")
    plt.show()


if __name__ == "__main__":
    X, y = generate_data()
    plot_raw_data(X, y)

    model = init_model()
    print("Initial parameters require_grad status:")
    for i, p in enumerate(get_parameters(model)):
        print(f"  Param {i}: shape {tuple(p.shape)}, requires_grad={p.requires_grad}")

    losses, steps = train_model(model, X, y, epochs=3000, lr=0.1, record_interval=100)
    plt.plot(steps, losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    plot_results(X, y, model)
