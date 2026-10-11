import os
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from reward import get_y, get_y_max
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist


def set_global_seeds(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class SubsetProxy(Subset):
    """
    Behaves exactly like torch.utils.data.Subset, but any attribute lookup that
    Subset doesn't implement is forwarded to the underlying dataset.

    Example
    -------
    cifar = torchvision.datasets.CIFAR10(...)
    car_idx = [i for i,t in enumerate(cifar.targets) if t == cifar.class_to_idx['car']]
    car_set = SubsetProxy(cifar, car_idx)
    >>> len(car_set)          # subset length
    5000
    >>> car_set.classes       # forwarded attribute
    ['airplane', 'automobile', ...]
    >>> car_set.targets[:5]   # forwarded, but still full-dataset order
    [1, 1, 1, 1, 1]
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)          # keep Subset internals safe
        except AttributeError:
            return getattr(self.dataset, name)


def get_stratified_subset_dataset(ds, nsamples, seed):
    samples_per_digit = nsamples // 10
    g = torch.Generator().manual_seed(seed)        # reproducibility
    subset_indices = []
    for d in range(10):
        class_idxs = (ds.targets == d).nonzero(as_tuple=False).view(-1)
        perm = class_idxs[torch.randperm(len(class_idxs), generator=g)]
        subset_indices.extend(perm[:samples_per_digit].tolist())

    return SubsetProxy(ds, subset_indices)


def plot_examples(
    test_ds, model, direction, alpha1=0.2, alpha2=1.0, variant='kmeans',
    figname=None, showfig=True
):
    '''
    alpha1 : weight for small intervention
    alpha2 : weight for large intervention
    variant : how to calculate the rgb values from the image to be used in reward calculation
        if kmeans, we find the color by kmeans with 2 clusters
        if max_intensity, we find the color with the maximum l2 norm
        across rgb and then we use the rgb values for the maximum intensity pixel.
    '''
    device = next(model.parameters()).device
    test_ld = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    model.eval()

    # Collect one sample per digit 0-9
    gray_imgs, col_imgs, recon_imgs, interv_imgs, interv2_imgs, labels = [], [], [], [], [], []
    seen = set()
    true_ys = []
    true_rgbs = []
    with torch.no_grad():
        for gray, colour, label, rgb, z, R in test_ld:
            d = label.item()
            if d in seen:
                continue
            seen.add(d)
            true_ys.append(np.float64(R.detach().cpu().numpy()[0]))
            true_rgbs.append(rgb.detach().cpu().numpy())
            gray_imgs.append(gray[0])
            col_imgs.append(colour[0])
            labels.append(d)
            flat = colour.view(1, -1).to(device)
            recon = model(flat)[0].cpu().view(3, 28, 28)
            recon_imgs.append(recon)

            # intervene
            latents = model.encode(flat)
            latents[:, :len(direction)] += alpha1 * torch.Tensor(direction).to(device)
            interv = model.decode(latents)
            interv_imgs.append(interv.cpu().view(3, 28, 28))

            latents = model.encode(flat)
            latents[:, :len(direction)] += alpha2 * torch.Tensor(direction).to(device)
            interv = model.decode(latents)
            interv2_imgs.append(interv.cpu().view(3, 28, 28))

            if len(seen) == 10:
                break

    def reward_fn_kmeans(x):
        return get_y(x, test_ds.dgp)

    def reward_fn_max_intensity(x):
        return get_y_max(x, test_ds.dgp)

    if variant == 'kmeans':
        reward_fn = reward_fn_kmeans
    elif variant == 'max_intensity':
        reward_fn = reward_fn_max_intensity
    else:
        raise AttributeError("Unknown option")

    y_true = np.array(true_ys)
    y_original = reward_fn(col_imgs)[0]
    y_recon = reward_fn(recon_imgs)[0]
    y_interv = reward_fn(interv_imgs)[0]
    y_interv2 = reward_fn(interv2_imgs)[0]

    fig, ax = plt.subplots(5, 10, figsize=(27, 10))
    for i in range(10):
        # Grayscale
        ax[0, i].imshow(gray_imgs[i][0], cmap="gray")
        ax[0, i].set_title(f"Digit {labels[i]}")
        ax[0, i].axis("off")

        # Colored input
        ax[1, i].imshow(col_imgs[i].permute(1, 2, 0))
        ax[1, i].set_title(f"Reward: {y_original[i]:.2f} \n (true reward: {y_true[i]:.2f})")
        ax[1, i].axis("off")

        # Reconstruction
        ax[2, i].imshow(recon_imgs[i].permute(1, 2, 0))
        ax[2, i].set_title(f"Reward: {y_recon[i]:.2f}")
        ax[2, i].axis("off")

        # Intervention
        ax[3, i].imshow(interv_imgs[i].permute(1, 2, 0))
        ax[3, i].set_title(f"Reward: {y_interv[i]:.2f}")
        ax[3, i].axis("off")

        ax[4, i].imshow(interv2_imgs[i].permute(1, 2, 0))
        ax[4, i].set_title(f"Reward: {y_interv2[i]:.2f}")
        ax[4, i].axis("off")

    for r in range(5):
        ax[r, 0].axis("on")        # or ax[r, 0].set_axis_on()
        ax[r, 0].tick_params(      # hide ticks only
            left=False, bottom=False, labelleft=False, labelbottom=False)

    ax[0, 0].set_ylabel("Gray",          rotation=90, size="large")
    ax[1, 0].set_ylabel("Original",       rotation=90, size="large")
    ax[2, 0].set_ylabel("Reconstructed", rotation=90, size="large")
    ax[3, 0].set_ylabel(f"Interv({alpha1:.1f})", rotation=90, size="large")
    ax[4, 0].set_ylabel(f"Interv({alpha2:.1f})", rotation=90, size="large")
    plt.tight_layout()
    if figname is not None:
        plt.savefig(figname, dpi=300)

    if showfig:
        plt.show()
    else:
        plt.close(fig)


def get_performance(
    test_ds, model, direction, alpha1=0.2, alpha2=1.0, variant='kmeans', batch_size=256
):
    '''
    alpha1 : weight for small intervention
    alpha2 : weight for large intervention
    variant : how to calculate the rgb values from the image to be used in reward calculation
        if kmeans, we find the color by kmeans with 2 clusters
        if max_intensity, we find the color with the maximum l2 norm
        across rgb and then we use the rgb values for the maximum intensity pixel.
    '''
    device = next(model.parameters()).device
    test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()

    # collections ---------------------------------------------------------
    gray_imgs, col_imgs = [], []
    recon_imgs, interv_imgs, interv2_imgs = [], [], []
    labels = []                      # kept in case you use it elsewhere
    true_ys, true_rgbs = [], []
    dir_tensor = torch.tensor(direction, device=device).unsqueeze(0)
    print("Generating interventional image data")
    with torch.no_grad():
        for gray, colour, label, rgb, z, R in test_ld:
            b = gray.size(0)
            true_ys.extend(R.cpu().numpy().astype(np.float64).tolist())
            true_rgbs.extend(rgb.cpu().numpy())
            labels.extend(label.cpu().tolist())

            gray_imgs.extend(gray.cpu().numpy())        # each is (1,28,28)
            col_imgs.extend(colour.cpu().numpy())       # each is (3,28,28)

            # 2. reconstructions -------------------------------------------
            flat = colour.view(b, -1).to(device)
            recon = model(flat)[0].detach().cpu().view(b, 3, 28, 28)
            recon_imgs.extend(recon.numpy())

            # 3. first-step intervention -----------------------------------
            latents = model.encode(flat)
            lat1 = latents.clone()
            lat1[:, :dir_tensor.size(1)] += alpha1 * dir_tensor
            interv = model.decode(lat1).detach().cpu().view(b, 3, 28, 28)
            interv_imgs.extend(interv.numpy())

            # 4. second-step intervention ----------------------------------
            lat2 = latents.clone()
            lat2[:, :dir_tensor.size(1)] += alpha2 * dir_tensor
            interv2 = model.decode(lat2).detach().cpu().view(b, 3, 28, 28)
            interv2_imgs.extend(interv2.numpy())

    # ---------- reward computation (unchanged) ----------------------------
    if isinstance(variant, list):
        results = []
        for v in variant:
            reward_fn = _pick_reward_fn(v, test_ds.dgp)
            results.append(_compute_rewards(reward_fn, true_ys,
                                            col_imgs, recon_imgs,
                                            interv_imgs, interv2_imgs))
        return tuple(results)

    reward_fn = _pick_reward_fn(variant, test_ds.dgp)
    return _compute_rewards(reward_fn, true_ys,
                            col_imgs, recon_imgs,
                            interv_imgs, interv2_imgs)


# helper ---------------------------------------------------------------
def _pick_reward_fn(v, dgp):

    def reward_fn_kmeans(x):
        return get_y(x, dgp)

    def reward_fn_max_intensity(x):
        return get_y_max(x, dgp)

    if v == 'kmeans':
        return reward_fn_kmeans
    elif v == 'max_intensity':
        return reward_fn_max_intensity
    raise ValueError("Unknown variant")


def _compute_rewards(
    reward_fn, true_ys, col_imgs, recon_imgs, interv_imgs, interv2_imgs
):
    y_true = np.asarray(true_ys)
    y_original = reward_fn(col_imgs)[0]
    y_recon = reward_fn(recon_imgs)[0]
    y_interv = reward_fn(interv_imgs)[0]
    y_interv2 = reward_fn(interv2_imgs)[0]
    return y_true, y_original, y_recon, y_interv, y_interv2


def plot_latent_space_from_dataset(dataset, autoencoder, batch_size=256, figname=None, showfig=True,
                                   color_variable='instrument'):
    """
    Visualize the latent space of a trained autoencoder on a ColoredMNIST dataset.
    Plots a scatter plot using the first ndim(z) latent variables, colored by the selected color variable.

    Parameters
    ----------
    dataset : ColoredMNIST
        The dataset instance to visualize.
    autoencoder : Autoencoder or ConvAutoencoder
        The trained autoencoder model.
    batch_size : int
        Batch size for DataLoader.
    figname : str or None
        If provided, save the figure to this filename.
    showfig : bool
        Whether to display the figure.
    color_variable : str or list of str
        Which variable(s) to use for coloring. Each must be one of {'instrument', 'rgb', 'digit', 'reward'}.
        If a list, a subplot is created for each variable.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

    device = next(autoencoder.parameters()).device
    autoencoder.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    latents_list = []
    z_list = []
    rgb_list = []
    label_list = []
    reward_list = []
    with torch.no_grad():
        for batch in loader:
            # batch: gray, color, label, rgb, z, R
            color = batch[1].to(device)  # (B, 3, 28, 28)
            z = batch[4]                 # (B, ndim_z)
            rgb = batch[3]               # (B, 3)
            label = batch[2]             # (B,)
            reward = batch[5]            # (B,)
            b = color.size(0)
            flat = color.view(b, -1)
            # Encode
            lat = autoencoder.encode(flat)
            latents_list.append(lat.detach().cpu())
            z_list.append(z.detach().cpu())
            rgb_list.append(rgb.detach().cpu())
            label_list.append(label.detach().cpu())
            reward_list.append(reward.detach().cpu())

    latents = torch.cat(latents_list, dim=0)
    z_all = torch.cat(z_list, dim=0)
    rgb_all = torch.cat(rgb_list, dim=0)
    label_all = torch.cat(label_list, dim=0)
    reward_all = torch.cat(reward_list, dim=0)
    ndim_z = z_all.shape[1]
    # Use only the first ndim_z latent dims
    latents_slice = latents[:, :ndim_z]

    X = latents_slice.numpy()
    Y_z = z_all.numpy()
    Y_rgb = rgb_all.numpy()
    Y_label = label_all.numpy()
    Y_reward = reward_all.numpy()
    slice_indices = tuple(range(ndim_z))

    # Support for multiple color variables (subplots)
    color_vars = color_variable if isinstance(color_variable, (list, tuple)) else [color_variable]
    nplots = len(color_vars)
    is_3d = len(slice_indices) == 3
    fig, axes = plt.subplots(1, nplots, figsize=(8 * nplots, 6),
                             subplot_kw={'projection': '3d'} if is_3d else {})
    if nplots == 1:
        axes = [axes]
    for idx, color_var in enumerate(color_vars):
        if color_var == 'instrument':
            Y = Y_z
        elif color_var == 'rgb':
            Y = Y_rgb
        elif color_var == 'digit':
            Y = Y_label.reshape(-1, 1)
        elif color_var == 'reward':
            Y = Y_reward.reshape(-1, 1)
        else:
            raise ValueError("color_variable must be one of {'instrument', 'rgb', 'digit', 'reward'}")

        if Y.shape[1] == 2:
            Y = np.hstack((Y, np.zeros((Y.shape[0], 1))))

        ax = axes[idx]
        if not is_3d:
            # 2D scatter
            if Y.shape[1] == 1:
                scatter = ax.scatter(X[:, 0], X[:, 1], c=Y[:, 0], cmap='viridis', s=20)
                fig.colorbar(scatter, ax=ax, label=color_var)
            elif Y.shape[1] == 3:
                norm = Y - Y.min(0)
                ptp = np.ptp(Y, axis=0) + 1e-8
                norm = norm / ptp
                ax.scatter(
                    X[:, 0],
                    X[:, 1],
                    c=norm,
                    s=20
                )
                ax.annotate(f'Color: {color_var}', xy=(0.99, 0.01), xycoords='axes fraction',
                            fontsize=12, ha='right', va='bottom',
                            bbox=dict(boxstyle='round', fc='w', alpha=0.7))
            else:
                scatter = ax.scatter(X[:, 0], X[:, 1], c=Y[:, 0], cmap='viridis', s=20)
                fig.colorbar(scatter, ax=ax, label=f'{color_var} 0')
            ax.set_xlabel(f'Latent {slice_indices[0]}')
            ax.set_ylabel(f'Latent {slice_indices[1]}')
            ax.set_title(f'Latent Space Scatter (2D)\nColor: {color_var}')
        else:
            # 3D scatter
            if Y.shape[1] == 1:
                p = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y[:, 0], cmap='viridis', s=20)
                fig.colorbar(p, ax=ax, label=color_var)
            elif Y.shape[1] == 3:
                norm = Y - Y.min(0)
                ptp = np.ptp(Y, axis=0) + 1e-8
                norm = norm / ptp
                ax.scatter(
                    X[:, 0],
                    X[:, 1],
                    X[:, 2],
                    c=norm,
                    s=20
                )
                ax.text2D(0.99, 0.01, f'Color: {color_var}', transform=ax.transAxes,
                          fontsize=12, ha='right', va='bottom',
                          bbox=dict(boxstyle='round', fc='w', alpha=0.7))
            else:
                p = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y[:, 0], cmap='viridis', s=20)
                fig.colorbar(p, ax=ax, label=f'{color_var} 0')
            ax.set_xlabel(f'Latent {slice_indices[0]}')
            ax.set_ylabel(f'Latent {slice_indices[1]}')
            ax.set_zlabel(f'Latent {slice_indices[2]}')
            ax.set_title(f'Latent Space Scatter (3D)\nColor: {color_var}')

    plt.tight_layout()
    if figname is not None:
        plt.savefig(figname, dpi=300)
    if showfig:
        plt.show()
    else:
        plt.close(fig)


def plot_linear_alignment_from_dataset(dataset, autoencoder, batch_size=256, figname=None, showfig=True):
    """
    Visualize linear alignment between the first ndim(z) latent factors and the rgb variable in ColoredMNIST.
    Fits a linear model from latents to rgb and plots:
      - Scatter plots of predicted vs. true RGB for each dimension (R, G, B), each in its own subplot.
      - Heatmap of the learned transformation matrix as the last subplot.

    Parameters
    ----------
    dataset : ColoredMNIST
        The dataset instance to visualize.
    autoencoder : Autoencoder or ConvAutoencoder
        The trained autoencoder model.
    batch_size : int
        Batch size for DataLoader.
    figname : str or None
        If provided, save the figure to this filename.
    showfig : bool
        Whether to display the figure.
    """
    device = next(autoencoder.parameters()).device
    autoencoder.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    latents_list = []
    rgb_list = []
    z_list = []
    with torch.no_grad():
        for batch in loader:
            color = batch[1].to(device)  # (B, 3, 28, 28)
            rgb = batch[3]               # (B, 3)
            z = batch[4]                 # (B, ndim_z)
            b = color.size(0)
            flat = color.view(b, -1)
            lat = autoencoder.encode(flat)
            latents_list.append(lat.detach().cpu())
            rgb_list.append(rgb.detach().cpu())
            z_list.append(z.detach().cpu())

    latents = torch.cat(latents_list, dim=0)
    rgb_all = torch.cat(rgb_list, dim=0)
    z_all = torch.cat(z_list, dim=0)
    ndim_z = z_all.shape[1]
    latents_slice = latents[:, :ndim_z].numpy()
    rgbs = rgb_all.numpy()

    # Fit linear model: latents -> rgb
    reg = LinearRegression().fit(latents_slice, rgbs)
    rgb_pred = reg.predict(latents_slice)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    colors = ['R', 'G', 'B']
    for i in range(3):
        ax = axes[i]
        ax.scatter(rgb_pred[:, i], rgbs[:, i], alpha=0.5, label=colors[i], color=colors[i].lower())
        minv = min(rgb_pred[:, i].min(), rgbs[:, i].min())
        maxv = max(rgb_pred[:, i].max(), rgbs[:, i].max())
        ax.plot([minv, maxv], [minv, maxv], 'r--', lw=1)
        ax.set_xlabel(f'Predicted {colors[i]}')
        ax.set_ylabel(f'True {colors[i]}')
        ax.set_title(f'{colors[i]} Channel')
        ax.legend()

    # 4. Heatmap of the transformation matrix
    ax = axes[3]
    im = ax.imshow(reg.coef_, cmap='bwr', aspect='auto')
    ax.set_title('Linear transformation (Latents → RGB)')
    ax.set_xlabel('Latent dim')
    ax.set_ylabel('RGB dim')
    ax.set_xticks(np.arange(ndim_z))
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(['R', 'G', 'B'])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if figname is not None:
        plt.savefig(figname, dpi=300)
    if showfig:
        plt.show()
    else:
        plt.close(fig)


def safe_plot_examples(
    test_ds, model, direction, train_ds=None, safety='clip', k=3, n_components=5,
    alpha1=0.2, alpha2=1.0, variant='kmeans',
    figname=None, showfig=True
):
    """
    Like plot_examples, but after intervention in latent space, applies a safety mechanism to keep the latent
    on the data manifold.
    Supports:
      - 'clip': Clip each latent dim to mean ± k*std (fit on train_ds)
      - 'pca': Project intervened latent to PCA subspace (fit on train_ds)
      - 'nearest': Use nearest neighbor latent from train_ds

    Parameters
    ----------
    test_ds : Dataset
        Test dataset.
    model : Autoencoder or ConvAutoencoder
        Trained model.
    direction : np.ndarray or list
        Intervention direction.
    train_ds : Dataset, optional
        Training dataset (required for 'clip', 'pca', 'nearest').
    safety : str
        One of {'clip', 'pca', 'nearest'}.
    k : float
        Number of stds for clipping (used if safety='clip').
    alpha1, alpha2 : float
        Intervention strengths.
    variant : str
        Reward variant.
    figname : str or None
        If provided, save the figure to this filename.
    showfig : bool
        Whether to display the figure.
    """
    device = next(model.parameters()).device
    test_ld = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    model.eval()

    # If needed, fit statistics/PCA/NN on train_ds
    latents_train = None
    pca = None
    if safety in ['clip', 'pca', 'nearest']:
        assert train_ds is not None, "train_ds must be provided for safety='clip', 'pca', or 'nearest'"
        train_ld = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=0)
        latents_list = []
        with torch.no_grad():
            for batch in train_ld:
                color = batch[1].to(device)
                b = color.size(0)
                flat = color.view(b, -1)
                lat = model.encode(flat)
                latents_list.append(lat.detach().cpu())
        latents_train = torch.cat(latents_list, dim=0).numpy()
        if safety == 'pca':
            ndim = latents_train.shape[1]
            pca = PCA(n_components=min(n_components, ndim))
            pca.fit(latents_train)

    # Collect one sample per digit 0-9
    gray_imgs, col_imgs, recon_imgs, interv_imgs, interv2_imgs, labels = [], [], [], [], [], []
    seen = set()
    true_ys = []
    true_rgbs = []
    with torch.no_grad():
        for gray, colour, label, rgb, z, R in test_ld:
            d = label.item()
            if d in seen:
                continue
            seen.add(d)
            true_ys.append(np.float64(R.detach().cpu().numpy()[0]))
            true_rgbs.append(rgb.detach().cpu().numpy())
            gray_imgs.append(gray[0])
            col_imgs.append(colour[0])
            labels.append(d)
            flat = colour.view(1, -1).to(device)
            recon = model(flat)[0].cpu().view(3, 28, 28)
            recon_imgs.append(recon)

            # intervene (alpha1)
            latents = model.encode(flat)
            latents[:, :len(direction)] += alpha1 * torch.Tensor(direction).to(device)
            lat_np = latents.detach().cpu().numpy()
            if safety == 'clip':
                mean = latents_train.mean(axis=0)
                std = latents_train.std(axis=0)
                lat_np = np.clip(lat_np, mean - k * std, mean + k * std)
            elif safety == 'pca':
                lat_np = pca.inverse_transform(pca.transform(lat_np))
            elif safety == 'nearest':
                dists = cdist(lat_np, latents_train)
                idx = np.argmin(dists, axis=1)
                lat_np = latents_train[idx]
            lat_safe = torch.from_numpy(lat_np).to(device).type_as(latents)
            interv = model.decode(lat_safe)
            interv_imgs.append(interv.cpu().view(3, 28, 28))

            # intervene (alpha2)
            latents = model.encode(flat)
            latents[:, :len(direction)] += alpha2 * torch.Tensor(direction).to(device)
            lat_np = latents.detach().cpu().numpy()
            if safety == 'clip':
                mean = latents_train.mean(axis=0)
                std = latents_train.std(axis=0)
                lat_np = np.clip(lat_np, mean - k * std, mean + k * std)
            elif safety == 'pca':
                lat_np = pca.inverse_transform(pca.transform(lat_np))
            elif safety == 'nearest':
                dists = cdist(lat_np, latents_train)
                idx = np.argmin(dists, axis=1)
                lat_np = latents_train[idx]
            lat_safe = torch.from_numpy(lat_np).to(device).type_as(latents)
            interv = model.decode(lat_safe)
            interv2_imgs.append(interv.cpu().view(3, 28, 28))

            if len(seen) == 10:
                break

    def reward_fn_kmeans(x):
        return get_y(x, test_ds.dgp)

    def reward_fn_max_intensity(x):
        return get_y_max(x, test_ds.dgp)

    if variant == 'kmeans':
        reward_fn = reward_fn_kmeans
    elif variant == 'max_intensity':
        reward_fn = reward_fn_max_intensity
    else:
        raise AttributeError("Unknown option")

    y_true = np.array(true_ys)
    y_original = reward_fn(col_imgs)[0]
    y_recon = reward_fn(recon_imgs)[0]
    y_interv = reward_fn(interv_imgs)[0]
    y_interv2 = reward_fn(interv2_imgs)[0]

    fig, ax = plt.subplots(5, 10, figsize=(27, 10))
    for i in range(10):
        # Grayscale
        ax[0, i].imshow(gray_imgs[i][0], cmap="gray")
        ax[0, i].set_title(f"Digit {labels[i]}")
        ax[0, i].axis("off")

        # Colored input
        ax[1, i].imshow(col_imgs[i].permute(1, 2, 0))
        ax[1, i].set_title(
            f"Reward: {y_original[i]:.2f} "
            f"\n (true reward: {y_true[i]:.2f})"
        )
        ax[1, i].axis("off")

        # Reconstruction
        ax[2, i].imshow(recon_imgs[i].permute(1, 2, 0))
        ax[2, i].set_title(f"Reward: {y_recon[i]:.2f}")
        ax[2, i].axis("off")

        # Intervention
        ax[3, i].imshow(interv_imgs[i].permute(1, 2, 0))
        ax[3, i].set_title(f"Reward: {y_interv[i]:.2f}")
        ax[3, i].axis("off")

        ax[4, i].imshow(interv2_imgs[i].permute(1, 2, 0))
        ax[4, i].set_title(f"Reward: {y_interv2[i]:.2f}")
        ax[4, i].axis("off")

    for r in range(5):
        ax[r, 0].axis("on")
        ax[r, 0].tick_params(
            left=False, bottom=False, labelleft=False, labelbottom=False)

    ax[0, 0].set_ylabel("Gray",          rotation=90, size="large")
    ax[1, 0].set_ylabel("Original",       rotation=90, size="large")
    ax[2, 0].set_ylabel("Reconstructed", rotation=90, size="large")
    ax[3, 0].set_ylabel(f"Interv({alpha1:.1f})", rotation=90, size="large")
    ax[4, 0].set_ylabel(f"Interv({alpha2:.1f})", rotation=90, size="large")
    plt.tight_layout()
    if figname is not None:
        plt.savefig(figname, dpi=300)
    if showfig:
        plt.show()
    else:
        plt.close(fig)
