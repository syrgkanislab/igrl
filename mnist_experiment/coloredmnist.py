import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms.functional import to_tensor
import matplotlib.pyplot as plt


class ColoredMNIST(datasets.MNIST):
    """
    Returns
    -------
    gray   : torch.FloatTensor, shape (1,28,28)
        Original grayscale image in [0,1].
    color  : torch.FloatTensor, shape (3,28,28)
        Colored version where white → [r,g,b].
    label  : int
        Digit class.
    rgb    : torch.FloatTensor, shape (3,)
        [r,g,b] for this sample.
    z      : torch.FloatTensor, shape (3,)
        The Z vector used in the formula.
    R      : torch.FloatTensor, shape ()     (scalar)
        Sum(r,g,b).
    """

    def __init__(
        self, *args, dgp='dgp2', alpha=0.5, beta=0.2, gray_threshold=None, seed=None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        print(f"Initiating Colored Mnist with dgp={dgp}, alpha={alpha}, beta={beta}")
        self.dgp = dgp
        self.alpha, self.beta = alpha, beta
        self.gray_threshold = gray_threshold

        # Set a deterministic seed for Z_ and U_ generation
        if seed is None:
            seed = 42  # Default seed

        # Save the current random state
        rng_state = torch.get_rng_state()

        # Set temporary seed for Z_ and U_ generation
        torch.manual_seed(seed)
        self.Z_ = torch.randn((self.__len__(), 3))
        self.U_ = torch.randn((self.__len__(), 3))

        # Restore the random state
        torch.set_rng_state(rng_state)

    @staticmethod
    def _clip01(x):
        return torch.clamp(x, 0., 1.)

    def _sample_rgb(self, idx):
        if self.dgp == 'dgp1':
            z = self.Z_[idx, :2]
            u = self.U_[idx, :2]
            r = self._clip01(0.5 + self.alpha * z[0] + self.beta * u[0])
            g = self._clip01(0.5 + self.alpha * z[1] + self.beta * u[1])
            b = self._clip01(0.5 + self.alpha * (z[0] - z[1]) / 2)
            rgb = torch.stack([r, g, b])
            R = rgb.sum()
        elif self.dgp == 'dgp2':
            z = self.Z_[idx]
            u = self.U_[idx]
            r = self._clip01(0.5 + self.alpha * z[0] + self.beta * u[0])
            g = self._clip01(0.5 + self.alpha * z[1] + self.beta * u[1])
            b = self._clip01(0.5 + self.alpha * z[2] + self.beta * u[2])
            rgb = torch.stack([r, g, b])
            R = rgb.sum()
        elif self.dgp == 'dgp3':
            z = self.Z_[idx]
            u = self.U_[idx]
            r = self._clip01(0.5 + self.alpha * z[0] + self.beta * u[0])
            g = self._clip01(0.5 + self.alpha * z[1] + self.beta * u[1])
            b = self._clip01(0.5 + self.alpha * z[2] + self.beta * u[2])
            rgb = torch.stack([r, g, b])
            R = rgb.sum() - u.sum()   # confounding in reward
        elif self.dgp == 'dgp4':
            z = self.Z_[idx]
            u = self.U_[idx]
            r = self._clip01(0.5 + self.alpha * (z[0] - z[1]) + self.beta * u[0])
            g = self._clip01(0.5 + self.alpha * (z[1] - z[2]) + self.beta * u[1])
            b = self._clip01(0.5 + self.alpha * (z[2] - z[0]) + self.beta * u[2])
            rgb = torch.stack([r, g, b])
            R = rgb[0] - u[0]
        else:
            raise AttributeError(f"Unknown option {self.dgp}")

        return rgb, z, R

    def __getitem__(self, idx):
        pil_img, label = super().__getitem__(idx)
        if self.gray_threshold is not None:
            gray = 1.0 * (to_tensor(pil_img) > self.gray_threshold)
        else:
            gray = to_tensor(pil_img)
        rgb, z, R = self._sample_rgb(idx)
        color = gray.repeat(3, 1, 1) * rgb.view(3, 1, 1)
        return gray, color, label, rgb, z, R


# ----------------------------------------------------------------------
# Display 10 originals & colored counterparts
# ----------------------------------------------------------------------
def show_examples(dgp, alpha=0.5, beta=0.2):
    ds = ColoredMNIST(dgp=dgp, root=".", train=False, download=True)
    loader = DataLoader(ds, batch_size=10, shuffle=True)
    gray, color, labels, _, _, _ = next(iter(loader))
    fig, ax = plt.subplots(2, 10, figsize=(15, 3))
    for i in range(10):
        # originals (grayscale)
        ax[0, i].imshow(gray[i, 0].cpu(), cmap="gray")
        ax[0, i].set_title(f"{labels[i].item()}")
        ax[0, i].axis("off")

        # colored
        img = color[i].permute(1, 2, 0).cpu()                 # (28,28,3)
        ax[1, i].imshow(img)
        ax[1, i].axis("off")
    ax[0, 0].set_ylabel("Original", rotation=90, size="large")
    ax[1, 0].set_ylabel("Colored", rotation=90, size="large")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_examples()
