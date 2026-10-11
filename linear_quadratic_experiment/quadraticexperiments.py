import numpy as np
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import optim
from torch.utils.data import DataLoader

import gendata
import methods
import rffautoencoder3
import rffvae
import rffivae

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device (Apple Silicon)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

def trainae(train_loader,test_loader, m, r, k, modeltype, referencemodel=None):
    input_dim = m
    encoder_hidden_dims = [100, 50, 20]  # List of hidden dimensions for encoder
    decoder_hidden_dims = [20, 50, 100]  # List of hidden dimensions for decoder
    predictor_hidden_dims = []           # List of hidden dimensions for predictor
    latent_dim = r    # D dimension
    z_dim = k         # Z dimension
    use_rff = True    # Use Random Fourier Features
    rff_sigma = 20    # Sigma parameter for RFF
    
    # Create model
    model = rffautoencoder3.AutoencoderSystem(
        input_dim=input_dim,
        encoder_hidden_dims=encoder_hidden_dims,
        decoder_hidden_dims=decoder_hidden_dims,
        predictor_hidden_dims=predictor_hidden_dims,
        latent_dim=latent_dim,
        z_dim=z_dim,
        use_rff=use_rff,
        rff_sigma=rff_sigma,
        rff_dim = 10,
        model_type = modeltype
    )

    if referencemodel is not None:
        model = rffautoencoder3.transfer_weights(referencemodel, model)
        
    # Define optimizer
    learning_rate = 5e-4
    optimizer = optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            alpha=0.9,           # Smoothing constant
            eps=1e-8,
            weight_decay=1e-6,
            momentum=0.0          # Can add momentum to RMSprop too
        )
    
    # regularizer weight
    if modeltype == -1:
        lambda_pred = 0
        lambda_corr = 0
        lambda_corrv = 0
        lambda_corrd = 0
    elif modeltype == 0:
        lambda_pred = 1
        lambda_corr = 0
        lambda_corrv = 0
        lambda_corrd = 0
    elif modeltype == 1:
        lambda_pred = 1
        lambda_corr = 1
        lambda_corrv = 0
        lambda_corrd = 0
    elif modeltype == 2:
        lambda_pred = 1
        lambda_corr = 1
        lambda_corrv = 1
        lambda_corrd = 0
    elif modeltype == 3:
        lambda_pred = 1
        lambda_corr = 1
        lambda_corrv = 1
        lambda_corrd = 1
    
    # history
    history = rffautoencoder3.train(
        model=model, 
        train_dataloader=train_loader,
        val_dataloader=test_loader,
        optimizer=optimizer,
        scheduler=None,                    # Learning rate scheduler
        scheduler_metric='val_loss',       # Use validation loss for scheduler decisions
        epochs=1000,                       # Maximum epochs
        patience=20,                       # Early stopping patience
        device=device,
        lambda_rec=1.0,                   
        lambda_pred = lambda_pred,           
        lambda_corr = lambda_corr,           
        lambda_corrv = lambda_corrv,
        lambda_corrd = lambda_corrd,
        save_best=False,                   # Save best model
        model_path=None,
        printevery=100
    )
    return model

def perturbae(standardized_data, train_loader, val_loader, m, r, k, modeltype, scalers, B, theta, referencemodel=None):
    model = trainae(train_loader, val_loader, m, r, k, modeltype=modeltype, referencemodel=referencemodel)
    with torch.no_grad():
        input_tensor = torch.FloatTensor(standardized_data["X"]).to(device)
        est_D = model.encoder(input_tensor).cpu().numpy()
        input_tensor_test = torch.FloatTensor(standardized_data["X_test"]).to(device)
        est_D_test = model.encoder(input_tensor_test).cpu().numpy()
    iv = methods.IVRegression()
    iv.fit(standardized_data["Z"], est_D[:,:k], standardized_data["Y"])
    ate_values = np.array(iv.theta)
    if est_D.shape[1] - k > 0:
        print("Add padding")
        for _ in range(est_D.shape[1] - k):
            ate_values = np.append(ate_values, 0)
    
    Dprime = est_D_test + 1*ate_values.T/np.linalg.norm(ate_values)
    with torch.no_grad():
        Xprime = model.decoder(torch.tensor(Dprime, dtype=torch.float32).to(device)).cpu().numpy()
    Xprime_orgspace = scalers["X"].inverse_transform(Xprime)
    Yprime = theta((Xprime_orgspace @ np.linalg.pinv(B.T))[:, :4])
   
    Dprime = est_D + 1*ate_values.T/np.linalg.norm(ate_values)
    with torch.no_grad():
        Xprime = model.decoder(torch.tensor(Dprime, dtype=torch.float32).to(device)).cpu().numpy()
    Xprime_orgspace = scalers["X"].inverse_transform(Xprime)
    Yprime_train = theta((Xprime_orgspace @ np.linalg.pinv(B.T))[:, :4])

    return Yprime, Yprime_train, model

def trainvae(train_loader,test_loader, m, r, k):
    input_dim = m
    encoder_hidden_dims = [100, 50, 20]  # List of hidden dimensions for encoder
    decoder_hidden_dims = [20, 50, 100]  # List of hidden dimensions for decoder
    latent_dim = r    # D dimension
    z_dim = k         # Z dimension
    use_rff = True    # Use Random Fourier Features
    rff_sigma = 20    # Sigma parameter for RFF
    
    # Create model
    model = rffvae.VAESystem(
        input_dim=input_dim,
        encoder_hidden_dims=encoder_hidden_dims,
        decoder_hidden_dims=decoder_hidden_dims,
        latent_dim=latent_dim,
        z_dim=z_dim,
        use_rff=use_rff,
        rff_sigma=rff_sigma,
        rff_dim = 10
    )
        
    # Define optimizer
    learning_rate = 1e-4
    optimizer = optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            alpha=0.9,           # Smoothing constant
            eps=1e-8,
            weight_decay=1e-6,
            momentum=0.0          # Can add momentum to RMSprop too
        )
        
    # Train model with validation, early stopping, and learning rate scheduling
    history = rffvae.train_vae(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=test_loader,
            optimizer=optimizer,
            scheduler=None,
            scheduler_metric=None,
            epochs=1000,
            patience=20,
            device=device,
            lambda_kl=3,
            save_best=False,
            model_path=None,
            printevery=100,
        )
    return model

def train_ivae_model(train_loader, test_loader, m, r, k):
    input_dim = m
    encoder_hidden_dims = [100, 50, 20]  # List of hidden dimensions for encoder
    decoder_hidden_dims = [20, 50, 100]  # List of hidden dimensions for decoder
    prior_hidden_dims = []               # List of hidden dimensions for conditional prior
    latent_dim = r                       # Latent (d) dimension
    z_dim = k                            # Auxiliary (z/u) dimension
    use_rff = True                       # Use Random Fourier Features
    rff_sigma = 20                       # Sigma parameter for RFF
    
    # Create iVAE model
    model = rffivae.iVAESystem(
        input_dim=input_dim,
        z_dim=z_dim,
        encoder_hidden_dims=encoder_hidden_dims,
        decoder_hidden_dims=decoder_hidden_dims,
        prior_hidden_dims=prior_hidden_dims,
        latent_dim=latent_dim,
        use_rff=use_rff,
        rff_sigma=rff_sigma,
        rff_dim=10
    )
    
    # Define optimizer - using same settings as your autoencoder
    learning_rate = 5e-4
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=learning_rate,
        alpha=0.9,           # Smoothing constant
        eps=1e-8,
        weight_decay=1e-6,
        momentum=0.0          # Can add momentum to RMSprop too
    )
    
    # Train the model
    history = rffivae.train_ivae(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=test_loader,
        optimizer=optimizer,
        scheduler=None,               # Learning rate scheduler
        scheduler_metric='val_loss',  # Use validation loss for scheduler decisions
        epochs=1000  ,                # Maximum epochs
        patience=20,                  # Early stopping patience
        device=device,
        lambda_kl=3,                  # Weight for KL divergence term
        save_best=False,              # Save best model
        model_path=None,
        printevery=100                # Print every X epochs (1001 to effectively disable)
    )
    
    return model

def iter(i, type, returnfull=False):
    rs = i
    np.random.seed(rs)
    torch.manual_seed(rs)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except:
        pass  # Older PyTorch versions
    
    n = 10000
    m = 50
    k = 4
    r = 4

    improvement = dict()
    improvementmed = dict()
    improvementcount = dict()
    
    if type == 1:
        # A is full rank
        A = np.random.randn(r, k)
        assert np.linalg.matrix_rank(A) == A.shape[0]

        # B
        B = np.random.randn(m, int(2*r + r*(r-1)/2))

        # t is vector
        t = np.random.randn(r, 1)

        g = lambda Z : Z @ A.T
        poly = PolynomialFeatures(degree=2, include_bias=False)
        f = lambda D: poly.fit_transform(D) @ B.T
        theta = lambda D : D @ t
        U = lambda n, r: 0.2*np.random.randn(n, r)
        V = lambda n, m: 0.2*np.random.randn(n, m)
        Q = lambda u, v, n: np.sum(u, axis=1)[:, None] + 0.2 * np.random.randn(n, 1)
    
    elif type == 2:
        # A is full rank
        A = np.random.randn(r, k)
        assert np.linalg.matrix_rank(A) == A.shape[0]

        # B
        B = np.random.randn(m, int(2*r + r*(r-1)/2))

        # t is vector
        t = np.random.randn(r, 1)

        g = lambda Z : Z @ A.T
        poly = PolynomialFeatures(degree=2, include_bias=False)
        f = lambda D: poly.fit_transform(D) @ B.T
        theta = lambda D : D @ t
        h = 3
        E = np.random.randn(h, r)
        U = lambda n, r: 0.2 * np.random.uniform(-1, 1, (n, h)) @ E
        V = lambda n, m: 0.2*np.random.randn(n, m)
        Q = lambda u, v, n: np.sum(u, axis=1)[:, None] + 0.2 * np.random.randn(n, 1)    
    
    elif type == 3:
        # A is full rank
        A = np.random.randn(r, k)
        assert np.linalg.matrix_rank(A) == A.shape[0]

        # B
        B = np.random.randn(m, int(2*r + r*(r-1)/2))

        # t is vector
        t = np.random.randn(r, 1)

        h=3
        E=np.random.randn(h, r)
        #print(E)

        h2=5
        F=np.random.randn(h2, m)
        #print(F)

        g = lambda Z : Z @ A.T
        poly = PolynomialFeatures(degree=2, include_bias=False)
        f = lambda D: poly.fit_transform(D) @ B.T
        theta = lambda D : D @ t
        h = 3
        E = np.random.randn(h, r)
        U = lambda n, r: 0.2 * np.random.uniform(-1, 1, (n, h)) @ E
        h2 = 5
        E2 = np.random.randn(h2, m)
        V = lambda n, m: 0.05 * np.random.randn(n, h2) @ E2
        Q = lambda u, v, n: np.sum(u, axis=1)[:, None] + 0.2 * np.random.randn(n, 1)
    else:
        print("Not supported")
        return
    
    data = gendata.gendata(g, f, theta, U, V, Q, n, m, k, r, rs, testsize=0.2)
    standardized_data, scalers = gendata.standardize_numpy_datasets(data)

    # Create the dataset
    val_idx = np.random.choice(len(standardized_data["X"]), int(len(standardized_data["X"])*0.125), replace=False)
    val_mask = np.zeros(len(standardized_data["X"]), dtype=bool)
    val_mask[val_idx] = True


    dataset = gendata.Dataset(standardized_data["X"][~val_mask], standardized_data["Z"][~val_mask])
    dataset_val = gendata.Dataset(standardized_data["X"][val_mask], standardized_data["Z"][val_mask])
    
    # Create the DataLoader
    train_loader = DataLoader(dataset, batch_size=500, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=500, shuffle=True)

    ##########################################################################################
    # LIRR
    mymodel = methods.LIRR(r)
    mymodel.fit(standardized_data["Z"], standardized_data["X"], standardized_data["Y"])
    D = mymodel.encode(standardized_data["X_test"])
    Dprime = D + 1*mymodel.gettheta().T/np.linalg.norm(mymodel.gettheta())
    Xprime = gendata.unstandardize_numpy_datasets({"X": mymodel.decode(Dprime)}, scalers)["X"]
    Yprime = theta((Xprime @ np.linalg.pinv(B.T))[:, :4])
    improvement["lirr_test"] = np.mean(Yprime - data["Y_test"])
    improvementmed["lirr_test"] = np.median(Yprime - data["Y_test"])
    improvementcount["lirr_test"] = np.mean((Yprime - data["Y_test"]) >= 0)
    D = mymodel.encode(standardized_data["X"])
    Dprime = D + 1*mymodel.gettheta().T/np.linalg.norm(mymodel.gettheta())
    Xprime = gendata.unstandardize_numpy_datasets({"X": mymodel.decode(Dprime)}, scalers)["X"]
    Yprime = theta((Xprime @ np.linalg.pinv(B.T))[:, :4])
    improvement["lirr"] = np.mean(Yprime - data["Y"])
    improvementmed["lirr"] = np.median(Yprime - data["Y"])
    improvementcount["lirr"] = np.mean((Yprime - data["Y"]) >= 0)

    ##########################################################################################
    # PCA
    pcamodel = methods.PCAMethod(r)
    pcamodel.fit(standardized_data["Z"], standardized_data["X"], standardized_data["Y"])
    D = pcamodel.encode(standardized_data["X_test"])
    Dprime = D + 1*pcamodel.gettheta().T/np.linalg.norm(pcamodel.gettheta())
    Xprime = gendata.unstandardize_numpy_datasets({"X": pcamodel.decode(Dprime)}, scalers)["X"]
    Yprime_pca = theta((Xprime @ np.linalg.pinv(B.T))[:, :4])
    improvement["pca_test"] = np.mean(Yprime_pca - data["Y_test"])
    improvementmed["pca_test"] = np.median(Yprime_pca - data["Y_test"])
    improvementcount["pca_test"] = np.mean((Yprime_pca - data["Y_test"]) >= 0)
    D = pcamodel.encode(standardized_data["X"])
    Dprime = D + 1*pcamodel.gettheta().T/np.linalg.norm(pcamodel.gettheta())
    Xprime = gendata.unstandardize_numpy_datasets({"X": pcamodel.decode(Dprime)}, scalers)["X"]
    Yprime_pca = theta((Xprime @ np.linalg.pinv(B.T))[:, :4])
    improvement["pca"] = np.mean(Yprime_pca - data["Y"])
    improvementmed["pca"] = np.median(Yprime_pca - data["Y"])
    improvementcount["pca"] = np.mean((Yprime_pca - data["Y"]) >= 0)

    ##########################################################################################
    # Vanilla AE
    print("Training Vanilla AE")
    Yprime, Yprime_train, _ = perturbae(standardized_data, train_loader, val_loader, m, k, k, -1, scalers, B, theta)
    improvement["ae_test"] = np.mean(Yprime - data["Y_test"])
    improvementmed["ae_test"] = np.median(Yprime - data["Y_test"])
    improvementcount["ae_test"] = np.mean((Yprime - data["Y_test"]) >= 0)
    improvement["ae"] = np.mean(Yprime_train - data["Y"])
    improvementmed["ae"] = np.median(Yprime_train - data["Y"])
    improvementcount["ae"] = np.mean((Yprime_train - data["Y"]) >= 0)

    ##########################################################################################
    # IRAE[0]
    print("Training AE just Z")
    Yprime, Yprime_train, _ = perturbae(standardized_data, train_loader, val_loader, m, k, k, 0, scalers, B, theta)
    improvement["ae0_test"] = np.mean(Yprime - data["Y_test"])
    improvementmed["ae0_test"] = np.median(Yprime - data["Y_test"])
    improvementcount["ae0_test"] = np.mean((Yprime - data["Y_test"]) >= 0)
    improvement["ae0"] = np.mean(Yprime_train - data["Y"])
    improvementmed["ae0"] = np.median(Yprime_train - data["Y"])
    improvementcount["ae0"] = np.mean((Yprime_train - data["Y"]) >= 0)

    ##########################################################################################
    # IRAE[1]
    print("Training IRAE[1]")
    Yprime, Yprime_train, referencemodel = perturbae(standardized_data, train_loader, val_loader, m, k, k, 1, scalers, B, theta)
    improvement["ae1_test"] = np.mean(Yprime - data["Y_test"])
    improvementmed["ae1_test"] = np.median(Yprime - data["Y_test"])
    improvementcount["ae1_test"] = np.mean((Yprime - data["Y_test"]) >= 0)
    improvement["ae1"] = np.mean(Yprime_train - data["Y"])
    improvementmed["ae1"] = np.median(Yprime_train - data["Y"])
    improvementcount["ae1"] = np.mean((Yprime_train - data["Y"]) >= 0)

    ##########################################################################################
    # IRAE[2]
    print("Training IRAE[2]")
    Yprime, Yprime_train, _ = perturbae(standardized_data, train_loader, val_loader, m, 10, k, 2, scalers, B, theta)#, referencemodel)
    improvement["ae2_test"] = np.mean(Yprime - data["Y_test"])
    improvementmed["ae2_test"] = np.median(Yprime - data["Y_test"])
    improvementcount["ae2_test"] = np.mean((Yprime - data["Y_test"]) >= 0)
    improvement["ae2"] = np.mean(Yprime_train - data["Y"])
    improvementmed["ae2"] = np.median(Yprime_train - data["Y"])
    improvementcount["ae2"] = np.mean((Yprime_train - data["Y"]) >= 0)

    ##########################################################################################
    # IRAE[3]
    print("Training IRAE[3]")
    Yprime, Yprime_train, _ = perturbae(standardized_data, train_loader, val_loader, m, 10, k, 3, scalers, B, theta)#, referencemodel)
    improvement["ae3_test"] = np.mean(Yprime - data["Y_test"])
    improvementmed["ae3_test"] = np.median(Yprime - data["Y_test"])
    improvementcount["ae3_test"] = np.mean((Yprime - data["Y_test"]) >= 0)
    improvement["ae3"] = np.mean(Yprime_train - data["Y"])
    improvementmed["ae3"] = np.median(Yprime_train - data["Y"])
    improvementcount["ae3"] = np.mean((Yprime_train - data["Y"]) >= 0)

    ##########################################################################################
    # VAE
    print("Training vanilla VAE")
    model = trainvae(train_loader, val_loader, m, r, k)

    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(standardized_data["X_test"]).to(device)
        X_tensor = torch.FloatTensor(standardized_data["X"]).to(device)
        mean_test, logvar_test = model.encoder(X_test_tensor)
        mean, logvar = model.encoder(X_tensor)
        # Use mean for IV regression (no sampling for this part)
        est_D_test = mean_test.cpu().numpy()
        est_D = mean.cpu().numpy()
        
    iv = methods.IVRegression()
    iv.fit(standardized_data["Z"], est_D, standardized_data["Y"])
    ate_values = iv.theta
    
    # For test data - sample 10 times from latent distribution
    Dprime_base = est_D_test + 1*ate_values.T/np.linalg.norm(ate_values)
    Dprime_base_tensor = torch.tensor(Dprime_base, dtype=torch.float32).to(device)
    logvar_test_tensor = torch.tensor(logvar_test, dtype=torch.float32).to(device)
    
    # Create 10 samples by sampling from the latent distribution
    all_diffs_test = []
    with torch.no_grad():
        for _ in range(10):
            # Sample from latent space using reparameterization trick
            std = torch.exp(0.5 * logvar_test_tensor)
            eps = torch.randn_like(std)
            Dprime_sample = Dprime_base_tensor + eps * std
            # Decode the sample
            Xprime_sample = model.decoder(Dprime_sample).cpu().numpy()
            Xprime_orgspace = scalers["X"].inverse_transform(Xprime_sample)
            Yprime = theta((Xprime_orgspace @ np.linalg.pinv(B.T))[:, :4])
            all_diffs_test.append(Yprime - data["Y_test"])
    
    # Concatenate all differences for statistics calculation
    all_diffs_test = np.vstack(all_diffs_test)
    improvement["vae_test"] = np.mean(all_diffs_test)
    improvementmed["vae_test"] = np.median(all_diffs_test)
    improvementcount["vae_test"] = np.mean(all_diffs_test >= 0)
    
    # For training data - sample 10 times from latent distribution
    Dprime_base = est_D + 1*ate_values.T/np.linalg.norm(ate_values)
    
    # Convert to torch tensors
    Dprime_base_tensor = torch.tensor(Dprime_base, dtype=torch.float32).to(device)
    logvar_tensor = torch.tensor(logvar, dtype=torch.float32).to(device)
    
    # Create 10 samples by sampling from the latent distribution
    all_diffs_train = []
    with torch.no_grad():        
        for _ in range(10):
            # Sample from latent space using reparameterization trick
            std = torch.exp(0.5 * logvar_tensor)
            eps = torch.randn_like(std)  # This will be on the same device as std
            Dprime_sample = Dprime_base_tensor + eps * std
            # Decode
            Xprime_sample = model.decoder(Dprime_sample).cpu().numpy()
            Xprime_orgspace = scalers["X"].inverse_transform(Xprime_sample)
            Yprime = theta((Xprime_orgspace @ np.linalg.pinv(B.T))[:, :4])
            all_diffs_train.append(Yprime - data["Y"])
    
    # Concatenate all differences for statistics calculation
    all_diffs_train = np.vstack(all_diffs_train)
    improvement["vae"] = np.mean(all_diffs_train)
    improvementmed["vae"] = np.median(all_diffs_train)
    improvementcount["vae"] = np.mean(all_diffs_train >= 0)

    ##########################################################################################
    # ADD IVAE
    print("Training iVAE")
    model = train_ivae_model(train_loader, val_loader, m, r, k)
    
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(standardized_data["X_test"]).to(device)
        Z_test_tensor = torch.FloatTensor(standardized_data["Z_test"]).to(device)
        X_tensor = torch.FloatTensor(standardized_data["X"]).to(device)
        Z_tensor = torch.FloatTensor(standardized_data["Z"]).to(device)
    
        # Get mean and logvar from encoder - Note that iVAE encoder requires both X and Z inputs
        mean_test, logvar_test = model.encoder(X_test_tensor, Z_test_tensor)
        mean, logvar = model.encoder(X_tensor, Z_tensor)
        
        # Use mean for IV regression (no sampling for this part)
        est_D = mean.cpu().numpy()
        est_D_test = mean_test.cpu().numpy()
    
    iv = methods.IVRegression()
    iv.fit(standardized_data["Z"], est_D, standardized_data["Y"])
    ate_values = iv.theta
    
    # For test data - sample 10 times from latent distribution
    Dprime_base = est_D_test + 1*ate_values.T/np.linalg.norm(ate_values)
    
    # Convert to torch tensors
    Dprime_base_tensor = torch.tensor(Dprime_base, dtype=torch.float32).to(device)
    logvar_test_tensor = torch.tensor(logvar_test, dtype=torch.float32).to(device)
    
    # Create 10 samples by sampling from the latent distribution
    all_diffs_test = []
    with torch.no_grad():
        for _ in range(10):
            # Sample from latent space using reparameterization trick
            std = torch.exp(0.5 * logvar_test_tensor)
            eps = torch.randn_like(std) 
            Dprime_sample = Dprime_base_tensor + eps * std
            # Decode the sample and move back to CPU for numpy operations
            Xprime_sample = model.decoder(Dprime_sample).cpu().numpy()
            Xprime_orgspace = scalers["X"].inverse_transform(Xprime_sample)
            Yprime = theta((Xprime_orgspace @ np.linalg.pinv(B.T))[:, :4])
            all_diffs_test.append(Yprime - data["Y_test"])
    
    # Concatenate all differences for statistics calculation
    all_diffs_test = np.vstack(all_diffs_test)
    improvement["ivae_test"] = np.mean(all_diffs_test)
    improvementmed["ivae_test"] = np.median(all_diffs_test)
    improvementcount["ivae_test"] = np.mean(all_diffs_test >= 0)

    # For test data - sample 10 times from latent distribution
    Dprime_base = est_D + 1*ate_values.T/np.linalg.norm(ate_values)
    
    # Convert to torch tensors
    Dprime_base_tensor = torch.tensor(Dprime_base, dtype=torch.float32).to(device)
    logvar_tensor = torch.tensor(logvar, dtype=torch.float32).to(device)

    all_diffs_train = []
    with torch.no_grad():
        for _ in range(10):
            # Sample from latent space using reparameterization trick
            std = torch.exp(0.5 * logvar_tensor)
            eps = torch.randn_like(std)
            Dprime_sample = Dprime_base_tensor + eps * std
            # Decode the sample and move back to CPU for numpy operations
            Xprime_sample = model.decoder(Dprime_sample).cpu().numpy()
            Xprime_orgspace = scalers["X"].inverse_transform(Xprime_sample)
            Yprime = theta((Xprime_orgspace @ np.linalg.pinv(B.T))[:, :4])
            all_diffs_train.append(Yprime - data["Y"])
    
    # Concatenate all differences for statistics calculation
    all_diffs_train = np.vstack(all_diffs_train)
    improvement["ivae"] = np.mean(all_diffs_train)
    improvementmed["ivae"] = np.median(all_diffs_train)
    improvementcount["ivae"] = np.mean(all_diffs_train >= 0)
    
    return improvement, improvementmed, improvementcount  # Return the new dictionary

def experiment(dgp, file):
    improvement = dict()
    improvementmed = dict()
    improvementcount = dict()
    repeats = 30
    for i in tqdm(range(repeats)):
        improv, improvmed, improvcount = iter(i, dgp)
        for k in improv.keys():
            if k in improvement:
                improvement[k].append(improv[k])
                improvementmed[k].append(improvmed[k])
                improvementcount[k].append(improvcount[k])
            else:
                improvement[k] = [improv[k]]
                improvementmed[k] = [improvmed[k]]
                improvementcount[k] = [improvcount[k]]
    np.savez("quadratic_avgimrpove_dgp="+str(dgp), **improvement)        
    lst = ['pca_test', 'pca', \
        'lirr_test', 'lirr', \
        'ae_test', 'ae', \
        'ae0_test', 'ae0',
       'ae1_test', 'ae1', \
       'ae2_test', 'ae2', \
       'ae3_test', 'ae3', \
       'vae_test', 'vae', \
       'ivae_test', 'ivae']
    
    title = {
    'pca': 'PCA',
    'lirr': "LIRR",
    'ae': "Vanilla AE",
    'ae0': "IRAE[0]",
    'ae1': "IRAE[1]",
    'ae2': 'IRAE[2]',
    'ae3': 'IRAE',
    'vae': "Vanilla VAE",
    'ivae': "iVAE"
    }

    file.write("\tavg improvement across of "+str(repeats)+" runs and its std\n")
    plt.figure(figsize=(15, 7.5))
    for i, k in enumerate(lst):
        y = improvement[k][:repeats]

        subplot_idx = i//2
        orgrow = subplot_idx // 3
        orgcol = subplot_idx % 3
        subplot_idx = orgrow + orgcol*3 +1

        if "test" in k:
            file.write(f"\t\t{title[k.replace("_test", "")]}\t{np.mean(np.array(y))}\t{np.std(np.array(y))}\n")
            plt.subplot(3, 3, subplot_idx)
            plt.title(title[k.replace("_test", "")])
            plt.hist(y, alpha=0.7, label="Test", density=True, bins=np.linspace(-15, 15, 31))#, color="red")
            plt.plot([0, 0], [0, plt.ylim()[1]], 'r--')
            plt.xlabel("Average Test Improvements")
            plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig("./plots/quadratic_avgimprove_dgp="+str(dgp)+".png")


def main():
    with open("quadratic_result.txt", "w") as file:
        file.write("Running experiment for DGP 1\n")
        print("Running experiment for DGP 1\n")
        experiment(1, file)
        file.write("Running experiment for DGP 2\n")
        print("Running experiment for DGP 2\n")
        experiment(2, file)
        file.write("Running experiment for DGP 3\n")
        print("Running experiment for DGP 3\n")
        experiment(3, file)

if __name__ == "__main__":
    main()