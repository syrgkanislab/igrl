import os
import random
import numpy as np
import torch
from coloredmnist import ColoredMNIST
from utilities import set_global_seeds, get_stratified_subset_dataset
from ivreg import tsls, get_iv_dataset
from independence import pairwise_hsic, hsic, linear_hsic, pairwise_joint_hsic, joint_hsic, linear_joint_hsic
from autoencoder import (
    train_ae, Autoencoder, ConvAutoencoder,
    init_big_from_small, init_big_from_small_conv
)
import joblib
import argparse
import warnings
warnings.simplefilter("ignore")


def worker_init_fn(worker_id):
    """Function to make DataLoader workers deterministic"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def igrl(
    train_ds, test_ds, device, arch, epochs, batch_size, patience, dimz, latent_dim, vanilla,
    indep_reg, jindep_reg, lambda_aux, lambda_reg1, lambda_reg2, lambda_reg3,
    modelname, figname, resname, warm_start, rerun
):
    if os.path.exists(modelname) and os.path.exists(resname) and not rerun:
        return
    print(f"Training size: {len(train_ds)}, Test size: {len(test_ds)}")
    train_ld = torch.utils.data.DataLoader(
        train_ds, batch_size, shuffle=True, num_workers=10, pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    test_ld = torch.utils.data.DataLoader(
        test_ds, batch_size, shuffle=False, num_workers=10, pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    print(f"Training model {modelname}")
    ae, _ = train_ae(
        train_ld, test_ld, device, arch=arch, epochs=epochs, patience=patience,
        dimz=dimz, latent_dim=latent_dim, vanilla=vanilla, indep_reg=indep_reg,
        jindep_reg=jindep_reg, lambda_aux=lambda_aux, lambda_reg1=lambda_reg1,
        lambda_reg2=lambda_reg2, lambda_reg3=lambda_reg3, name=modelname,
        warm_start=warm_start
    )
    print(f"Finished training model {modelname}")
    print(f"Training IV from model {modelname}")
    Z, D, Y = get_iv_dataset(train_ld, ae, dimz)
    u = tsls(Z, D, Y)
    joblib.dump(u, resname)
    print(f"Saved intervention direction in {resname}")
    return


def igrl_experiment(
    it, dgp, arch, latent_dim, reg_weight, indep_reg, warm_start,
    epochs1, epochs2, subsample, batch_size, patience, rerun
):
    conf_str = (
        f'config_{dgp}_{arch}_{latent_dim}_{reg_weight}_{indep_reg}_'
        f'{epochs1}_{batch_size}_{patience}'
    )
    if subsample > 0:
        conf_str = f'{conf_str}_{subsample}'
    conf_str_ws = f'{conf_str}_{warm_start}_{epochs2}'
    print(f"Training iteration {it} of configuration: {conf_str_ws}")

    # Set global seeds at the start of each experiment iteration
    set_global_seeds(it)

    alpha = np.random.uniform(0.1, 0.7)
    beta = np.random.uniform(0.1, 0.7)
    print(f"{it}: alpha={alpha}, beta={beta}")

    train_ds = ColoredMNIST(
        dgp=dgp, alpha=alpha, beta=beta, root=".", train=True, download=True, seed=it
    )
    test_ds = ColoredMNIST(
        dgp=dgp, alpha=alpha, beta=beta, root=".", train=False, download=True, seed=it
    )
    if subsample > 0:
        train_ds = get_stratified_subset_dataset(train_ds, int(subsample * 0.8), it)
        test_ds = get_stratified_subset_dataset(test_ds, int(subsample * 0.2), it)

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"{n_gpus} CUDA device(s) visible to PyTorch. Using device {it % n_gpus}")
        device = torch.device(f"cuda:{it % n_gpus}")
    else:
        device = torch.device("cpu")

    result_dir = 'result_dir'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    image_dir = 'image_dir'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    model_dir = 'model_dir'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    epochs = epochs1
    if dgp == 'dgp1':
        dimz = 2
    elif dgp in ['dgp2', 'dgp3', 'dgp4']:
        dimz = 3
    else:
        raise AttributeError(f"Unknown option {dgp}")

    if indep_reg == 'pairwise_hsic':
        indep_reg = pairwise_hsic
        jindep_reg = pairwise_joint_hsic
    elif indep_reg == 'hsic':
        indep_reg = hsic
        jindep_reg = joint_hsic
    elif indep_reg == 'linear_hsic':
        indep_reg = linear_hsic
        jindep_reg = linear_joint_hsic
    else:
        raise AttributeError(f"Unknown option {indep_reg}")

    # vanilla AE
    igrl(
        train_ds, test_ds, device, arch, epochs, batch_size, patience,
        dimz=dimz, latent_dim=dimz, vanilla=True, indep_reg=indep_reg,
        jindep_reg=jindep_reg, lambda_aux=None, lambda_reg1=None,
        lambda_reg2=None, lambda_reg3=None,
        modelname=os.path.join(model_dir, f"vanilla_{it}_{conf_str}.pt"),
        figname=os.path.join(image_dir, f"vanilla_{it}_{conf_str}_examples.png"),
        resname=os.path.join(result_dir, f"vanilla_{it}_{conf_str}_res.jbl"),
        warm_start=None, rerun=rerun
    )

    # just Z
    igrl(
        train_ds, test_ds, device, arch, epochs, batch_size, patience,
        dimz=dimz, latent_dim=dimz, vanilla=False, indep_reg=indep_reg,
        jindep_reg=jindep_reg, lambda_aux=reg_weight, lambda_reg1=None,
        lambda_reg2=None, lambda_reg3=None,
        modelname=os.path.join(model_dir, f"justZ_{it}_{conf_str}.pt"),
        figname=os.path.join(image_dir, f"justZ_{it}_{conf_str}_examples.png"),
        resname=os.path.join(result_dir, f"justZ_{it}_{conf_str}_res.jbl"),
        warm_start=None, rerun=rerun
    )

    # IRAE1: adding conditional independence of D - AZ and Z
    igrl(
        train_ds, test_ds, device, arch, epochs, batch_size, patience,
        dimz=dimz, latent_dim=dimz, vanilla=False, indep_reg=indep_reg,
        jindep_reg=jindep_reg, lambda_aux=reg_weight, lambda_reg1=reg_weight,
        lambda_reg2=None, lambda_reg3=None,
        modelname=os.path.join(model_dir, f"irae1_{it}_{conf_str}.pt"),
        figname=os.path.join(image_dir, f"irae1_{it}_{conf_str}_examples.png"),
        resname=os.path.join(result_dir, f"irae1_{it}_{conf_str}_res.jbl"),
        warm_start=None, rerun=rerun
    )

    # IRAE2
    if warm_start:
        if arch == 'dense':
            irae1 = Autoencoder(latent_dim=dimz).to(device)
            irae2 = Autoencoder(latent_dim=latent_dim).to(device)
            irae1.load_state_dict(
                torch.load(
                    os.path.join(model_dir, f"irae1_{it}_{conf_str}.pt"),
                    map_location=device
                )['ae_state']
            )
            init_big_from_small(irae1, irae2, dimz)
        elif arch == 'conv':
            irae1 = ConvAutoencoder(latent_dim=dimz).to(device)
            irae2 = ConvAutoencoder(latent_dim=latent_dim).to(device)
            irae1.load_state_dict(
                torch.load(
                    os.path.join(model_dir, f"irae1_{it}_{conf_str}.pt"),
                    map_location=device
                )['ae_state']
            )
            init_big_from_small_conv(irae1, irae2, dimz)
        else:
            raise AttributeError(f"Unknown option {arch}")
        epochs = epochs2
    else:
        irae2 = None
        epochs = epochs1

    igrl(
        train_ds, test_ds, device, arch, epochs, batch_size, patience,
        dimz=dimz, latent_dim=latent_dim, vanilla=False, indep_reg=indep_reg,
        jindep_reg=jindep_reg, lambda_aux=reg_weight, lambda_reg1=reg_weight,
        lambda_reg2=reg_weight, lambda_reg3=None,
        modelname=os.path.join(model_dir, f"irae2_{it}_{conf_str_ws}.pt"),
        figname=os.path.join(image_dir, f"irae2_{it}_{conf_str_ws}_examples.png"),
        resname=os.path.join(result_dir, f"irae2_{it}_{conf_str_ws}_res.jbl"),
        warm_start=irae2, rerun=rerun
    )

    # IRAE3
    if warm_start:
        if arch == 'dense':
            irae1 = Autoencoder(latent_dim=dimz).to(device)
            irae3 = Autoencoder(latent_dim=latent_dim).to(device)
            irae1.load_state_dict(
                torch.load(
                    os.path.join(model_dir, f"irae1_{it}_{conf_str}.pt"),
                    map_location=device
                )['ae_state']
            )
            init_big_from_small(irae1, irae3, dimz)
        elif arch == 'conv':
            irae1 = ConvAutoencoder(latent_dim=dimz).to(device)
            irae3 = ConvAutoencoder(latent_dim=latent_dim).to(device)
            irae1.load_state_dict(
                torch.load(
                    os.path.join(model_dir, f"irae1_{it}_{conf_str}.pt"),
                    map_location=device
                )['ae_state']
            )
            init_big_from_small_conv(irae1, irae3, dimz)
        else:
            raise AttributeError(f"Unknown option {arch}")
        epochs = epochs2
    else:
        irae3 = None
        epochs = epochs1

    igrl(
        train_ds, test_ds, device, arch, epochs, batch_size, patience,
        dimz=dimz, latent_dim=latent_dim, vanilla=False, indep_reg=indep_reg,
        jindep_reg=jindep_reg, lambda_aux=reg_weight, lambda_reg1=reg_weight,
        lambda_reg2=reg_weight, lambda_reg3=reg_weight,
        modelname=os.path.join(model_dir, f"irae3_{it}_{conf_str_ws}.pt"),
        figname=os.path.join(image_dir, f"irae3_{it}_{conf_str_ws}_examples.png"),
        resname=os.path.join(result_dir, f"irae3_{it}_{conf_str_ws}_res.jbl"),
        warm_start=irae3, rerun=True
    )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo script that processes an input file."
    )
    parser.add_argument(
        "it",
        type=int,
        help="Experiment Iteration."
    )
    parser.add_argument(
        "-d", "--dgp",
        type=str,
        default='dgp2',
        help="DGP for latent generation. One of {dgp1, dgp2, dgp3, dgp4}."
    )
    parser.add_argument(
        "-a", "--arch",
        type=str,
        default='dense',
        help="Autoencoder architecture type one of {dense, conv}."
    )
    parser.add_argument(
        "-l", "--ldim",
        type=int,
        default=10,
        help="Autoencoder latent dimension."
    )
    parser.add_argument(
        "-rw", "--regweight",
        type=float,
        default=1.0,
        help="Weight on regularization components beyond the reconstruction loss."
    )
    parser.add_argument(
        "-ir", "--ireg",
        type=str,
        default='pairwise_hsic',
        help="Independence regularizer. One of {pairwise_hsic, hsic, linear_hsic}."
    )
    parser.add_argument(
        "--warm_start",
        action="store_true",
        default=False,
        help="Whether to warm start IRAE2,3 from IRAE1",
    )
    parser.add_argument(
        "-e1", "--epochs1",
        type=int,
        default=50,
        help="How many epochs to train."
    )
    parser.add_argument(
        "-e2", "--epochs2",
        type=int,
        default=50,
        help="If warm_start, then this is used for how many epochs to train for warm-started models."
    )
    parser.add_argument(
        "-ss", "--subsample",
        type=int,
        default=-1,
        help="Subsample the data."
    )
    parser.add_argument(
        "-bs", "--batch_size",
        type=int,
        default=256,
        help="Batch size."
    )
    parser.add_argument(
        "-pt", "--patience",
        type=int,
        default=5,
        help="Early stopping patience."
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        default=False,
        help="Rerun existing results if found in dir",
    )
    args = parser.parse_args()
    igrl_experiment(
        args.it, args.dgp, args.arch, args.ldim, args.regweight, args.ireg, args.warm_start,
        args.epochs1, args.epochs2, args.subsample, args.batch_size, args.patience, args.rerun
    )
