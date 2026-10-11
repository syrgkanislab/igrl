import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from independence import pairwise_hsic, pairwise_joint_hsic


"""
Auto-encoder on *Colored*-MNIST
with auxiliary Z-to-latent linear loss
-------------------------------------

Loss =  recon_loss  +  λ_aux · latent_loss
  where
    recon_loss  = MSE( decoder(encoder(x)), x )
    latent_loss = MSE( W·Z + b , first-3‐coords(encoder(x)) )

Dataset supplies Z for every sample.
"""


# ----------------------------------------------------------------------
# 2.  Auto-encoder definition
# ----------------------------------------------------------------------
class Autoencoder(nn.Module):
    def __init__(self, in_dim=3 * 28 * 28, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, in_dim),
            nn.Sigmoid()                # keep output in [0,1]
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


class ConvAutoencoder(nn.Module):
    """
    A simple convolutional auto-encoder tailored to 28×28 MNIST images.

    Args
    ----
    in_channels : int
        Number of channels in the input image (1 for grayscale MNIST).
    latent_dim : int
        Dimensionality of the bottleneck representation.
    """
    def __init__(self, in_channels: int = 3, latent_dim: int = 128):
        super().__init__()

        # ---------- Encoder ----------
        # 28×28  → 14×14  →  7×7 spatial size reduction
        self.encoder = nn.Sequential(
            nn.Unflatten(1, (in_channels, 28, 28)),
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),   # (B,16,14,14)
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),   # (B,32,14,14)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),            # (B,64, 7, 7)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),                                                     # (B, 64*7*7)
            nn.Linear(64 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, latent_dim)
        )

        # ---------- Decoder ----------

        # 7×7  → 14×14  → 28×28 spatial size restoration
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64 * 7 * 7),
            nn.BatchNorm1d(64 * 7 * 7),
            nn.Unflatten(1, (64, 7, 7)),                                      # (B,64,7,7)
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),                 # (B,32,7,7)
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),                 # (B,16,14,14)
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, in_channels, kernel_size=3, stride=2,
                               padding=1, output_padding=1),                 # (B,3,28,28)
            nn.Sigmoid(),                                                     # keep output in [0,1]
            nn.Flatten()
        )

    # ---- Public helpers (same API as your dense version) ----
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


class ZPredictor(nn.Module):
    """3×3 linear map:   Z  →  prediction of first-3 latent coords."""
    def __init__(self, dimz):
        super().__init__()
        self.map = nn.Linear(dimz, dimz, bias=True)

    def forward(self, Z):
        return self.map(Z)


# ----------------------------------------------------------------------
# 3.  Training/Evaluation Function
# ----------------------------------------------------------------------
def run_epoch(
    ae, zp, loader, optimizer, device, criterion, indep_reg, jindep_reg, vanilla,
    lambda_aux, lambda_reg1, lambda_reg2, lambda_reg3, train=True, noise_std=0.0
):
    """
    Run one epoch of training or evaluation.
    If train==True and noise_std > 0, adds Gaussian noise to the latent code (denoising autoencoder).
    """
    epoch_loss = 0.0
    ae.train(mode=train)
    if not vanilla:
        zp.train(mode=train)

    for _, x, _, _, Z, *_ in loader:
        x = x.to(device).view(x.size(0), -1)  # (B, 2352)
        Z = Z.to(device).float()  # (B, 3)

        if train:
            optimizer.zero_grad()

        x_rec, lat = ae(x)                     # lat shape (B, latent_dim)
        # Denoising: add noise to latent during training
        if train and noise_std > 0.0:
            lat = lat + torch.randn_like(lat) * noise_std
        recon_loss = criterion(x_rec, x)

        loss = recon_loss

        if not vanilla:
            z_pred = zp(Z)
            dimz = z_pred.size(1)
            resid = lat[:, :dimz] - z_pred              # R = D - AZ - c

            if lambda_aux is not None:
                aux_loss = criterion(z_pred, lat[:, :dimz])
                loss += lambda_aux * aux_loss

            if lambda_reg1 is not None:
                reg1_loss = indep_reg(resid, Z)    # independence penalty
                loss += lambda_reg1 * reg1_loss

            if lambda_reg2 is not None:
                reg2_loss = indep_reg(lat[:, dimz:], Z)
                loss += lambda_reg2 * reg2_loss

            if lambda_reg3 is not None:
                reg3_loss = 0.85 * indep_reg(lat[:, :dimz], lat[:, dimz:])    # D ci V
                reg3_loss += 0.15 * indep_reg(resid, lat[:, dimz:])    # U ci V
                reg3_loss += 0.05 * jindep_reg([Z, resid, lat[:, dimz:]])    # Z ci U ci V
                loss += lambda_reg3 * reg3_loss

        if train:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item() * x.size(0)

    return epoch_loss / len(loader.dataset)


def train_ae(
    train_ld, test_ld, device, *, arch='dense', dimz=3, latent_dim=10, epochs=50,
    lr=1e-3, patience=5, vanilla=False, indep_reg=pairwise_hsic, jindep_reg=pairwise_joint_hsic,
    lambda_aux=1.0, lambda_reg1=1.0, lambda_reg2=1.0, lambda_reg3=1.0, name='ae_checkpoint.pt',
    warm_start=None, noise_std=0.0
):
    '''
    warm_start : either None or an autoencoder instance to be used as a warm start
    noise_std : standard deviation of Gaussian noise added to latent code during training (denoising autoencoder)
    jindep_reg : function for joint independence regularization
    '''
    print(
        f"Training model with parameters: arch={arch}, dimz={dimz}, "
        f"latent_dim={latent_dim}, epochs={epochs}, lr={lr}, vanilla={vanilla}, "
        f"indep_reg={indep_reg}, jindep_reg={jindep_reg}, lambda_aux={lambda_aux}, "
        f"lambda_reg1={lambda_reg1}, lambda_reg2={lambda_reg2}, lambda_reg3={lambda_reg3}, "
        f"noise_std={noise_std}"
    )

    # models
    if warm_start is not None:
        print("Using pre-trained autoencoder")
        ae = warm_start
    else:
        if arch == 'dense':
            print("Using a dense autoencoder")
            ae = Autoencoder(in_dim=3*28*28, latent_dim=latent_dim).to(device)
        else:
            print("Using a convolutional autoencoder")
            ae = ConvAutoencoder(latent_dim=latent_dim).to(device)

    criterion = nn.MSELoss()
    if not vanilla:
        zp = ZPredictor(dimz).to(device)
        optimizer = optim.Adam(list(ae.parameters()) + list(zp.parameters()), lr=lr)
    else:
        zp = None
        optimizer = optim.Adam(list(ae.parameters()), lr=lr)

    best_va = np.inf
    best_ae_state = copy.deepcopy(ae.state_dict())  # keep full copy, not a reference
    best_zp_state = copy.deepcopy(zp.state_dict()) if zp else None
    patience, epochs_no_improve = patience, 0  # stop after x epochs w/out gain

    for ep in range(1, epochs + 1):
        tr = run_epoch(
            ae, zp, train_ld, optimizer, device, criterion, indep_reg, jindep_reg,
            vanilla, lambda_aux, lambda_reg1, lambda_reg2, lambda_reg3, train=True,
            noise_std=noise_std
        )
        va = run_epoch(
            ae, zp, test_ld, optimizer, device, criterion, indep_reg, jindep_reg,
            vanilla, lambda_aux, lambda_reg1, lambda_reg2, lambda_reg3, train=False,
            noise_std=0.0
        )
        print(f"Ep {ep:2d}/{epochs}  train {tr:.6f}  val {va:.6f}")

        if va < best_va:                                    # improvement
            best_va = va
            best_ae_state = copy.deepcopy(ae.state_dict())
            best_zp_state = copy.deepcopy(zp.state_dict()) if zp else None
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve > patience:
                print(f"No val-loss improvement for {patience} epochs → early stop.")
                break

    # restore best weights
    ae.load_state_dict(best_ae_state)
    if zp and best_zp_state is not None:
        zp.load_state_dict(best_zp_state)

    if vanilla:
        torch.save({
            "ae_state": ae.state_dict(),
        }, name)
    else:
        torch.save({
            "ae_state": ae.state_dict(),
            "zp_state": zp.state_dict()
        }, name)

    print(f"saved → {name}")

    return ae, zp


def init_big_from_small(small: nn.Module,
                        big:   nn.Module,
                        shared_latent: int = 3):
    """
    Initialise `big` so its first `shared_latent` latent dimensions reproduce the
    behaviour of the trained `small` auto-encoder.

    Assumes identical hidden architecture aside from latent size.
    """
    with torch.no_grad():

        # ---------- Encoder ----------
        # Copy all hidden Linear layers wholesale
        for layer_s, layer_b in zip(small.encoder[:-1], big.encoder[:-1]):
            if isinstance(layer_s, nn.Linear):
                layer_b.weight.copy_(layer_s.weight)
                layer_b.bias.copy_(layer_s.bias)

        # Copy the (hidden -> latent) mapping for the first `shared_latent` rows
        enc_s = small.encoder[-1]          # nn.Linear
        enc_b = big.encoder[-1]
        enc_b.weight[:shared_latent].copy_(enc_s.weight)
        enc_b.bias[:shared_latent].copy_(enc_s.bias)
        # enc_b.weight[shared_latent:] left at random initialisation

        # ---------- Decoder ----------
        # Copy the (latent -> hidden) mapping for the matching columns
        dec_s0 = small.decoder[0]          # nn.Linear
        dec_b0 = big.decoder[0]
        dec_b0.weight[:, :shared_latent].copy_(dec_s0.weight)
        # Bias vector is same size in both models → copy entirely
        dec_b0.bias.copy_(dec_s0.bias)

        # Copy all remaining hidden Linear layers wholesale
        for layer_s, layer_b in zip(small.decoder[1:], big.decoder[1:]):
            if isinstance(layer_s, nn.Linear):
                layer_b.weight.copy_(layer_s.weight)
                layer_b.bias.copy_(layer_s.bias)

    print(
        "Initialisation complete: shared layers cloned, "
        f"latent dims 0–{shared_latent-1} aligned."
    )


def _copy_identical(src: nn.Module, tgt: nn.Module) -> None:
    """Clone everything when the layer shapes match exactly."""
    if isinstance(src, (nn.Linear,
                        nn.Conv2d,
                        nn.ConvTranspose2d)):
        tgt.weight.copy_(src.weight)
        if src.bias is not None:
            tgt.bias.copy_(src.bias)

    elif isinstance(src, (nn.BatchNorm1d, nn.BatchNorm2d)):
        tgt.weight.copy_(src.weight)
        tgt.bias.copy_(src.bias)
        tgt.running_mean.copy_(src.running_mean)
        tgt.running_var.copy_(src.running_var)


def init_big_from_small_conv(small: nn.Module,
                             big:   nn.Module,
                             shared_latent: int = 3) -> None:
    """
    Make the first `shared_latent` latent dimensions of *big* behave exactly
    like the (only) latent dimensions of *small*.

    ▸ Works for fully-connected or convolutional auto-encoders;
      provided the two models are architecturally identical except that
      `big` uses a larger latent dimension.

    ▸ Copies every layer whose shape is unchanged;
      partially copies the layers that touch the latent space.
    """
    with torch.no_grad():

        # ---------- ENCODER (everything except the final layer) ----------
        for layer_s, layer_b in zip(small.encoder[:-1], big.encoder[:-1]):
            _copy_identical(layer_s, layer_b)

        # ---------- ENCODER → LATENT ----------
        enc_s = small.encoder[-1]
        enc_b = big.encoder[-1]

        if isinstance(enc_s, nn.Linear):                 # FC bottleneck
            enc_b.weight[:shared_latent].copy_(enc_s.weight)
            enc_b.bias[:shared_latent].copy_(enc_s.bias)

        elif isinstance(enc_s, nn.Conv2d):               # Conv bottleneck
            # Conv2d weight: (out_ch, in_ch, kH, kW)
            enc_b.weight[:shared_latent].copy_(enc_s.weight)
            if enc_s.bias is not None:
                enc_b.bias[:shared_latent].copy_(enc_s.bias)

        else:
            raise TypeError("Unsupported encoder–to–latent layer type "
                            f"{type(enc_s).__name__}")

        # ---------- LATENT → DECODER ----------
        dec_s0 = small.decoder[0]
        dec_b0 = big.decoder[0]

        if isinstance(dec_s0, nn.Linear):                # FC decoder stem
            # weight: (out, in) ; copy the shared columns
            dec_b0.weight[:, :shared_latent].copy_(dec_s0.weight)
            dec_b0.bias.copy_(dec_s0.bias)

        elif isinstance(dec_s0, nn.ConvTranspose2d):     # Conv-T stem
            # weight: (in_ch, out_ch, kH, kW)  (n.b. transpose conv!)
            dec_b0.weight[:shared_latent].copy_(dec_s0.weight)
            if dec_s0.bias is not None:
                dec_b0.bias.copy_(dec_s0.bias)

        else:
            raise TypeError("Unsupported latent–to–decoder layer type "
                            f"{type(dec_s0).__name__}")

        # ---------- REMAINING DECODER LAYERS ----------
        for layer_s, layer_b in zip(small.decoder[1:], big.decoder[1:]):
            _copy_identical(layer_s, layer_b)

    print(
        f"[\u2713] Copied all matching layers and aligned latent dims "
        f"0\u2013{shared_latent-1}."
    )
