import os
import numpy as np
import torch
from autoencoder import Autoencoder, ConvAutoencoder
from coloredmnist import ColoredMNIST
from utilities import plot_examples, get_performance, plot_latent_space_from_dataset, plot_linear_alignment_from_dataset
from utilities import set_global_seeds, get_stratified_subset_dataset
import joblib
import argparse
import warnings
warnings.simplefilter("ignore")


def igrl(train_ds, test_ds, device, arch, latent_dim,
         modelname, figname, scatterplot_figname, linear_alignment_figname, resname, perfname,
         rerun):
    if os.path.exists(perfname) and not rerun:
        existing = joblib.load(perfname)
        perf = existing[0]
        perf_mi = existing[1]
        return perf, perf_mi

    print(f"Training size: {len(train_ds)}, Test size: {len(test_ds)}")
    print(f"Loading model {modelname}")
    if arch == 'dense':
        ae = Autoencoder(latent_dim=latent_dim).to(device)
    elif arch == 'conv':
        ae = ConvAutoencoder(latent_dim=latent_dim).to(device)
    else:
        raise AttributeError(f"Unknown architecture {arch}")
    ae.load_state_dict(torch.load(modelname, map_location=device)['ae_state'])
    ae.eval()

    print(f"Loading intervention direction {resname}")
    u = joblib.load(resname)

    print(f"Preparing image {figname}")
    plot_examples(test_ds, ae, u, figname=figname, showfig=False)
    print(f"Saved image {figname}")
    # Create and save latent space scatter plot
    print(f"Preparing latent space scatter plot {scatterplot_figname}")
    plot_latent_space_from_dataset(
        test_ds, ae, figname=scatterplot_figname, showfig=False,
        color_variable=['instrument', 'rgb', 'reward', 'digit']
    )
    print(f"Saved scatter plot {scatterplot_figname}")
    # Create and save linear alignment plot
    print(f"Preparing linear alignment plot {linear_alignment_figname}")
    plot_linear_alignment_from_dataset(
        test_ds, ae, figname=linear_alignment_figname, showfig=False
    )
    print(f"Saved linear alignment plot {linear_alignment_figname}")

    print(f"Preparing results {perfname}")
    perf_results, perf_mi_results = get_performance(
        test_ds, ae, u, variant=['kmeans', 'max_intensity']
    )

    y_true, y_original, y_recon, y_interv, y_interv2 = perf_results
    perf = [
        np.mean(y_true), np.mean(y_original), np.mean(y_recon),
        np.mean(y_interv), np.mean(y_interv2)
    ]

    y_true, y_original, y_recon, y_interv, y_interv2 = perf_mi_results
    perf_mi = [
        np.mean(y_true), np.mean(y_original), np.mean(y_recon),
        np.mean(y_interv), np.mean(y_interv2)
    ]

    joblib.dump([perf, perf_mi], perfname)
    print(f"Saved results {perfname}")

    return perf, perf_mi


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
    print(f"Evaluating iteration {it} of configuration: {conf_str_ws}")

    # Set global seeds at the start of each experiment iteration
    set_global_seeds(it)

    # initializing alpha and beta right after the random seed to ensure consistency
    # between train and evaluation so that it is the same as the training alpha and beta
    alpha = np.random.uniform(0.1, 0.7)
    beta = np.random.uniform(0.1, 0.7)
    print(f"{it}: alpha={alpha}, beta={beta}")

    # data
    train_ds = ColoredMNIST(
        dgp=dgp, alpha=alpha, beta=beta, root=".", train=True,
        download=True
    )
    test_ds = ColoredMNIST(
        dgp=dgp, alpha=alpha, beta=beta, root=".", train=False,
        download=True
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
    perf_dir = 'perf_dir'
    if not os.path.exists(perf_dir):
        os.makedirs(perf_dir)

    if dgp == 'dgp1':
        dimz = 2
    elif dgp in ['dgp2', 'dgp3', 'dgp4']:
        dimz = 3
    else:
        raise AttributeError(f"Unknown option {dgp}")

    # vanilla AE
    vanilla_model_path = os.path.join(model_dir, f"vanilla_{it}_{conf_str}.pt")
    vanilla_fig_path = os.path.join(image_dir, f"vanilla_{it}_{conf_str}_examples.png")
    vanilla_scatter_path = vanilla_fig_path.replace('_examples.png', '_scatterplot.png')
    vanilla_linear_alignment_path = vanilla_fig_path.replace('_examples.png', '_linear_alignment.png')
    vanilla_res = igrl(
        train_ds, test_ds, device, arch, latent_dim=dimz,
        modelname=vanilla_model_path,
        figname=vanilla_fig_path,
        scatterplot_figname=vanilla_scatter_path,
        linear_alignment_figname=vanilla_linear_alignment_path,
        resname=os.path.join(result_dir, f"vanilla_{it}_{conf_str}_res.jbl"),
        perfname=os.path.join(perf_dir, f"vanilla_{it}_{conf_str}_perf.jbl"),
        rerun=rerun
    )

    # just Z prediction
    justZ_model_path = os.path.join(model_dir, f"justZ_{it}_{conf_str}.pt")
    justZ_fig_path = os.path.join(image_dir, f"justZ_{it}_{conf_str}_examples.png")
    justZ_scatter_path = justZ_fig_path.replace('_examples.png', '_scatterplot.png')
    justZ_linear_alignment_path = justZ_fig_path.replace('_examples.png', '_linear_alignment.png')
    justZ_res = igrl(
        train_ds, test_ds, device, arch, latent_dim=dimz,
        modelname=justZ_model_path,
        figname=justZ_fig_path,
        scatterplot_figname=justZ_scatter_path,
        linear_alignment_figname=justZ_linear_alignment_path,
        resname=os.path.join(result_dir, f"justZ_{it}_{conf_str}_res.jbl"),
        perfname=os.path.join(perf_dir, f"justZ_{it}_{conf_str}_perf.jbl"),
        rerun=rerun
    )

    # Z prediction and D-AZ independent of Z; IRAE1
    irae1_model_path = os.path.join(model_dir, f"irae1_{it}_{conf_str}.pt")
    irae1_fig_path = os.path.join(image_dir, f"irae1_{it}_{conf_str}_examples.png")
    irae1_scatter_path = irae1_fig_path.replace('_examples.png', '_scatterplot.png')
    irae1_linear_alignment_path = irae1_fig_path.replace('_examples.png', '_linear_alignment.png')
    irae1_res = igrl(
        train_ds, test_ds, device, arch, latent_dim=dimz,
        modelname=irae1_model_path,
        figname=irae1_fig_path,
        scatterplot_figname=irae1_scatter_path,
        linear_alignment_figname=irae1_linear_alignment_path,
        resname=os.path.join(result_dir, f"irae1_{it}_{conf_str}_res.jbl"),
        perfname=os.path.join(perf_dir, f"irae1_{it}_{conf_str}_perf.jbl"),
        rerun=rerun
    )

    # IRAE2
    irae2_model_path = os.path.join(model_dir, f"irae2_{it}_{conf_str_ws}.pt")
    irae2_fig_path = os.path.join(image_dir, f"irae2_{it}_{conf_str_ws}_examples.png")
    irae2_scatter_path = irae2_fig_path.replace('_examples.png', '_scatterplot.png')
    irae2_linear_alignment_path = irae2_fig_path.replace('_examples.png', '_linear_alignment.png')
    irae2_res = igrl(
        train_ds, test_ds, device, arch, latent_dim=latent_dim,
        modelname=irae2_model_path,
        figname=irae2_fig_path,
        scatterplot_figname=irae2_scatter_path,
        linear_alignment_figname=irae2_linear_alignment_path,
        resname=os.path.join(result_dir, f"irae2_{it}_{conf_str_ws}_res.jbl"),
        perfname=os.path.join(perf_dir, f"irae2_{it}_{conf_str_ws}_perf.jbl"),
        rerun=rerun
    )

    # IRAE3
    irae3_model_path = os.path.join(model_dir, f"irae3_{it}_{conf_str_ws}.pt")
    irae3_fig_path = os.path.join(image_dir, f"irae3_{it}_{conf_str_ws}_examples.png")
    irae3_scatter_path = irae3_fig_path.replace('_examples.png', '_scatterplot.png')
    irae3_linear_alignment_path = irae3_fig_path.replace('_examples.png', '_linear_alignment.png')
    irae3_res = igrl(
        train_ds, test_ds, device, arch, latent_dim=latent_dim,
        modelname=irae3_model_path,
        figname=irae3_fig_path,
        scatterplot_figname=irae3_scatter_path,
        linear_alignment_figname=irae3_linear_alignment_path,
        resname=os.path.join(result_dir, f"irae3_{it}_{conf_str_ws}_res.jbl"),
        perfname=os.path.join(perf_dir, f"irae3_{it}_{conf_str_ws}_perf.jbl"),
        rerun=True
    )

    return {'vanilla': vanilla_res, 'justZ': justZ_res,
            'irae1': irae1_res, 'irae2': irae2_res, 'irae3': irae3_res}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo script that processes an input file."
    )
    parser.add_argument(
        "start",
        type=int,
        help="Experiment Iteration."
    )
    parser.add_argument(
        "finish",
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
        help=(
            "If warm_start, then this is used for how many epochs to train "
            "for warm-started models."
        )
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
    start = args.start
    finish = args.finish
    rerun = args.rerun
    dgp = args.dgp
    arch = args.arch
    latent_dim = args.ldim
    reg_weight = args.regweight
    indep_reg = args.ireg
    epochs1 = args.epochs1
    epochs2 = args.epochs2
    warm_start = args.warm_start
    subsample = args.subsample
    batch_size = args.batch_size
    patience = args.patience
    results = [
        igrl_experiment(
            it, dgp, arch, latent_dim, reg_weight, indep_reg, warm_start,
            epochs1, epochs2, subsample, batch_size, patience, rerun
        ) for it in range(start, finish + 1)
    ]
    if subsample > 0:
        conf_str = (
            f'config_{dgp}_{arch}_{latent_dim}_{reg_weight}_{indep_reg}_'
            f'{epochs1}_{batch_size}_{patience}_{subsample}_{warm_start}_{epochs2}'
        )
    else:
        conf_str = (
            f'config_{dgp}_{arch}_{latent_dim}_{reg_weight}_{indep_reg}_'
            f'{epochs1}_{batch_size}_{patience}_{warm_start}_{epochs2}'
        )
    performance_dir = 'performance_results'
    if not os.path.exists(performance_dir):
        os.makedirs(performance_dir)
    joblib.dump(
        results,
        os.path.join(
            performance_dir,
            f"performance_results_{start}_{finish}_{conf_str}.jbl"
        )
    )
