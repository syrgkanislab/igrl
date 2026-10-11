import torch
import numpy as np
from sklearn.linear_model import LinearRegression


def tsls(Z, D, Y, debug=False):
    first = LinearRegression().fit(Z, D)
    if debug:
        print(first.coef_)
    second = LinearRegression().fit(first.predict(Z), Y)
    u = second.coef_
    unorm = np.linalg.norm(u, ord=2)
    if unorm > 0:  # if u is all zeros then return all zeros else normalize
        u = u / unorm
    return u


def get_iv_dataset(train_ld, ae, ndim_latent):
    '''
    Construct an IV dataset based on a trained autoencoder ae
    '''
    device = next(ae.parameters()).device
    Z, D, Y = [], [], []
    with torch.no_grad():
        for _, colour, _, rgb, z, R in train_ld:
            flat = colour.to(device).view(colour.size(0), -1)
            latents = ae.encode(flat)[:, :ndim_latent]
            Z.append(z.detach().cpu().numpy())
            Y.append(R.detach().cpu().numpy())
            D.append(latents.detach().cpu().numpy())
    Z = np.vstack(Z)
    D = np.vstack(D)
    Y = np.concatenate(Y)
    return Z, D, Y
