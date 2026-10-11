# Learning Treatment Representations for Downstream Instrumental Variable Regression

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.8-blue.svg)](https://www.python.org/)

## Overview

This repository provides code to reproduce all experiments in the paper **Learning Treatment Representations for Downstream Instrumental Variable Regression**. We propose a method for learning low-dimensional representations of treatments that preserve downstream performance under instrumental variable (IV) regression.

We include three sets of experiments:

- Linear settings
- Quadratic (nonlinear) settings
- MNIST (nonlinear) settings

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
    - [Linear and Quadratic Experiments](#linear-and-quadratic-experiments)
    - [MNIST Experiments](#mnist-experiments)
- [License](#license)
- [Contact](#contact)

## Installation

Clone the repository and install the required packages:


## Usage

### Linear and Quadratic Experiments

We include two synthetic setups. Each setup loops over 3 data-generating processes (DGPs).

#### Linear Experiments

Run:

```
cd linear_quadratic_experiment/
python linearexperiments.py
```

This will:

- Run linear DGPs 1–3 with treatment sample sizes \( m = 50, 100, 500 \)
- Save result plots to `./plots/`
- Save summary statistics to `linear_result.txt`

#### Quadratic Experiments

Run:

```
cd linear_quadratic_experiment/
python quadraticexperiments.py
```

This will:

- Run quadratic DGPs 1–3
- Save plots to `./plots/`
- Save statistics to `quadratic_result.txt`
- Save average improvements as `.npz` files

### MNIST Experiments

This large-scale experiment evaluates learned treatment representations on a deep learning task.

#### Step 1: Run Training Jobs

Launch the full sweep:

```
cd mnist_experiment/
./all_exps_many_hparams.sh
```

You may adjust `MAX_PARALLEL` in the script to control the number of concurrent jobs.

#### Step 2: Postprocess Results

After all runs finish, generate summary tables by running the jupyter notebook `Postprocess.ipynb`,

On a node with 2× H100 GPUs, the experiment completes in ~2 days. On smaller GPUs, it may take up to a week.

## License

This project is licensed under the MIT License.