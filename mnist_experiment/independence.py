import torch
from dhsic import dhsic


def _linear_kernel(x):
    return x @ x.T


def _rbf_kernel(x, sigma=None):
    """x: (B,d).  Returns (B,B) kernel matrix."""
    pairwise_d2 = torch.cdist(x, x, p=2.0) ** 2
    if sigma is None:
        # median heuristic
        sigma = torch.median(pairwise_d2.detach())
        sigma = torch.sqrt(0.5 * sigma) + 1e-8
    k = torch.exp(-pairwise_d2 / (2 * sigma ** 2))
    return k


def _pairwise_rbf_kernel(x, sigma=None):
    """x: (B,d).  Returns (B,B) kernel matrix."""
    return sum([
        _rbf_kernel(x[:, [i]], sigma) for i in range(x.size(1))
    ]) / x.size(1)


def hsic(x, y, sigma_x=None, sigma_y=None, pairwise=False, linear=False):
    """
    HSIC estimator (Gretton et al. 2005).
    x, y : (B,d1), (B,d2)
    returns scalar HSIC value.
    """
    B = x.size(0)
    if linear:
        K = _linear_kernel(x)
        L = _linear_kernel(y)
    elif pairwise:
        K = _pairwise_rbf_kernel(x, sigma_x)
        L = _pairwise_rbf_kernel(y, sigma_y)
    else:
        K = _rbf_kernel(x, sigma_x)
        L = _rbf_kernel(y, sigma_y)

    H = torch.eye(B, device=x.device) - 1.0 / B
    Kc = H @ K @ H
    Lc = H @ L @ H
    hsic_val = (Kc * Lc).sum() / (B - 1) ** 2
    # hsic_val = torch.trace(K @ H @ L @ H) / (B - 1) ** 2
    return hsic_val


def linear_hsic(x, y):
    return hsic(x, y, linear=True)


def pairwise_hsic(x, y, sigma_x=None, sigma_y=None):
    return hsic(x, y, sigma_x=sigma_x, sigma_y=sigma_y, pairwise=True)


def unbiased_hsic(x, y, sigma_x=None, sigma_y=None, pairwise=False):
    """
    unbiased HSIC estimator (Song, Le, et al. "Feature selection via dependence maximization." 2012.).
    x, y : (B,d1), (B,d2)
    returns scalar HSIC value.
    """
    N = x.size(0)
    if pairwise:
        kernel_XX = _pairwise_rbf_kernel(x, sigma_x)
        kernel_YY = _pairwise_rbf_kernel(y, sigma_y)
    else:
        kernel_XX = _rbf_kernel(x, sigma_x)
        kernel_YY = _rbf_kernel(y, sigma_y)

    tK = kernel_XX - torch.diag(kernel_XX)
    tL = kernel_YY - torch.diag(kernel_YY)

    hsic = (
        torch.trace(tK @ tL)
        + (torch.sum(tK) * torch.sum(tL) / (N - 1) / (N - 2))
        - (2 * torch.sum(tK, 0).dot(torch.sum(tL, 0)) / (N - 2))
    )

    return hsic / (N * (N - 3))


def pairwise_unbiased_hsic(x, y, sigma_x=None, sigma_y=None):
    return unbiased_hsic(x, y, sigma_x=sigma_x, sigma_y=sigma_y, pairwise=True)


def joint_hsic(x, pairwise=False, linear=False):
    """
    Joint HSIC estimator using dHSIC implementation.
    x: List of (B,di)
    returns scalar dHSIC value.

    This function provides the same API as hsic() but uses the dHSIC implementation
    internally for potentially better numerical stability and consistency with R.
    """
    if linear:
        kernel = "linear"
    elif pairwise:
        kernel = "pairwise"
    else:
        kernel = "gaussian"

    result = dhsic(x, kernel=kernel)
    return result['dHSIC']


def linear_joint_hsic(x):
    return joint_hsic(x, linear=True)


def pairwise_joint_hsic(x):
    return joint_hsic(x, pairwise=True)
