import torch
import numpy as np
from typing import List, Union, Optional


def median_bandwidth(X: torch.Tensor) -> float:
    """Compute the median heuristic bandwidth for Gaussian kernel.
    Matches R's implementation by using the same sampling approach.

    Args:
        X: Input tensor of shape (n_samples, n_features)

    Returns:
        float: Computed bandwidth
    """
    n_samples = X.shape[0]
    # Match R's sample(1:len) behavior - shuffle entire sequence
    indices = torch.randperm(n_samples)[:1000]
    X_sampled = X[indices]

    # Compute pairwise distances using R's approach
    X_norm = torch.sum(X_sampled**2, dim=1).view(-1, 1)
    dist = X_norm + X_norm.t() - 2 * torch.mm(X_sampled, X_sampled.t())

    # Get upper triangular elements (excluding diagonal) as in R
    triu_indices = torch.triu_indices(n_samples, n_samples, offset=1)
    dist = dist[triu_indices]

    # Compute median and ensure non-zero bandwidth
    bandwidth = np.sqrt(0.5 * torch.median(dist).item())
    return max(bandwidth, 0.001)  # Match R's minimum bandwidth value


def gaussian_gram_matrix(
    X: torch.Tensor, bandwidth: Optional[float] = None
) -> torch.Tensor:
    """Compute the Gaussian kernel Gram matrix.

    Args:
        X: Input tensor of shape (n_samples, n_features)
        bandwidth: Bandwidth parameter. If None, median heuristic is used.

    Returns:
        torch.Tensor: Gram matrix of shape (n_samples, n_samples)
    """
    if bandwidth is None:
        bandwidth = median_bandwidth(X)

    # Match R's implementation for numerical stability
    X_norm = torch.sum(X**2, dim=1).view(-1, 1)
    K = torch.exp(-(X_norm + X_norm.t() - 2 * torch.mm(X, X.t())) / (2 * bandwidth**2))
    return K


def discrete_gram_matrix(X: torch.Tensor) -> torch.Tensor:
    """Compute the discrete kernel Gram matrix.

    Args:
        X: Input tensor of shape (n_samples, n_features)

    Returns:
        torch.Tensor: Gram matrix of shape (n_samples, n_samples)
    """
    n_samples = X.shape[0]
    K = torch.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = torch.all(X[i] == X[j]).float()
    return K


def linear_gram_matrix(X: torch.Tensor) -> torch.Tensor:
    """Compute the linear kernel Gram matrix.

    Args:
        X: Input tensor of shape (n_samples, n_features)

    Returns:
        torch.Tensor: Gram matrix of shape (n_samples, n_samples)
    """
    return torch.mm(X, X.t())


def pairwise_rbf_gram_matrix(
    X: torch.Tensor, bandwidth: Optional[float] = None
) -> torch.Tensor:
    """Compute the pairwise RBF kernel Gram matrix.

    Args:
        X: Input tensor of shape (n_samples, n_features)
        bandwidth: Bandwidth parameter. If None, median heuristic is used.

    Returns:
        torch.Tensor: Gram matrix of shape (n_samples, n_samples)
    """
    n_features = X.shape[1]
    K = torch.zeros((X.shape[0], X.shape[0]), device=X.device)

    for i in range(n_features):
        X_i = X[:, i:i + 1]
        K += gaussian_gram_matrix(X_i, bandwidth)

    return K / n_features


def dhsic(
    X: Union[List[torch.Tensor], torch.Tensor],
    Y: Optional[torch.Tensor] = None,
    K: Optional[List[torch.Tensor]] = None,
    kernel: Union[str, List[str]] = "gaussian",
    bandwidth: Union[float, List[float]] = 1.0,
    matrix_input: bool = False,
) -> dict:
    """Compute the d-variable Hilbert Schmidt Independence Criterion (dHSIC).
    Implementation matches R's dHSIC package.

    Args:
        X: Either a list of tensors or a single tensor. If a single tensor and Y is provided,
           X and Y are treated as two variables. If matrix_input is True, columns of X are
           treated as different variables.
        Y: Optional second tensor if X is a single tensor
        K: Optional list of pre-computed Gram matrices
        kernel: Kernel type(s) to use. Can be "gaussian", "gaussian.fixed", "discrete", "linear", or "pairwise"
        bandwidth: Bandwidth parameter(s) for Gaussian kernel
        matrix_input: If True, treat columns of X as different variables

    Returns:
        dict: Dictionary containing dHSIC value, computation times, and bandwidths
    """
    import time

    original_K = K
    # Handle input preprocessing
    if K is None:
        if Y is not None:
            X = [X, Y]

        if matrix_input:
            if not isinstance(X, torch.Tensor):
                raise ValueError("X must be a tensor when matrix_input=True")
            X = [X[:, i : i + 1] for i in range(X.shape[1])]

        # Convert to list of tensors if not already
        if not isinstance(X, list):
            X = [X]

        # Ensure all inputs are 2D tensors
        X = [x if x.dim() == 2 else x.view(-1, 1) for x in X]

        d = len(X)
        n_samples = X[0].shape[0]

        if n_samples < 2 * d:
            print(
                "Warning: Sample size is smaller than twice the number of variables. dHSIC is trivial."
            )
            return {
                "dHSIC": 0.0,
                "time": {"GramMat": 0.0, "HSIC": 0.0},
                "bandwidth": None,
            }

        # Handle kernel and bandwidth parameters
        if isinstance(kernel, str):
            kernel = [kernel] * d
        if isinstance(bandwidth, (int, float)):
            bandwidth = [bandwidth] * d

        # Compute Gram matrices
        start_time = time.time()
        K = []
        for j in range(d):
            if kernel[j] == "gaussian":
                bandwidth[j] = median_bandwidth(X[j])
                K.append(gaussian_gram_matrix(X[j], bandwidth[j]))
            elif kernel[j] == "gaussian.fixed":
                K.append(gaussian_gram_matrix(X[j], bandwidth[j]))
            elif kernel[j] == "discrete":
                bandwidth[j] = None
                K.append(discrete_gram_matrix(X[j]))
            elif kernel[j] == "linear":
                bandwidth[j] = None
                K.append(linear_gram_matrix(X[j]))
            elif kernel[j] == "pairwise":
                bandwidth[j] = median_bandwidth(X[j])
                K.append(pairwise_rbf_gram_matrix(X[j], bandwidth[j]))
            else:
                raise ValueError(f"Unknown kernel type: {kernel[j]}")

        time_gram_mat = time.time() - start_time

    else:
        if not isinstance(K, list):
            raise ValueError("K must be a list of tensors")
        d = len(K)
        n_samples = K[0].shape[0]
        time_gram_mat = None

    # Compute dHSIC using R's normalization approach
    start_time = time.time()

    # Initialize terms as in R's implementation
    term1 = torch.ones_like(K[0])
    term2 = 1.0
    term3 = 2.0 / n_samples

    # Compute terms with improved numerical stability
    for j in range(d):
        term1 = term1 * K[j]
        term2 = (1.0 / n_samples**2) * term2 * torch.sum(K[j])
        term3 = (1.0 / n_samples) * term3 * torch.sum(K[j], dim=0)

    # Final computation with improved numerical stability
    term1 = torch.sum(term1)
    term3 = torch.sum(term3)

    dhsic_value = (1.0 / n_samples**2) * term1 + term2 - term3
    time_hsic = time.time() - start_time

    return {
        "dHSIC": dhsic_value.item(),
        "time": {"GramMat": time_gram_mat, "HSIC": time_hsic},
        "bandwidth": bandwidth if original_K is None else None,
    }
