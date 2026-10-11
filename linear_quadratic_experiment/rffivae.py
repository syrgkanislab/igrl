import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

class RandomFourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, sigma=1.0):
        super(RandomFourierFeatures, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = sigma
        
        # Random weights with Gaussian distribution
        self.register_buffer('weight', torch.randn(in_features, out_features // 2) / sigma)
        self.register_buffer('bias', torch.rand(out_features // 2) * 2 * np.pi)
        
    def forward(self, x):
        # Project input
        projection = x @ self.weight + self.bias
        
        # Apply sin and cos transformations
        feature_sin = torch.sin(projection)
        feature_cos = torch.cos(projection)
        
        # Concatenate features and normalize
        features = torch.cat([feature_sin, feature_cos], dim=-1) * np.sqrt(2.0 / self.out_features)
        return features

class iVAEEncoder(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dims, latent_dim, use_rff=True, rff_sigma=1.0, rff_dim=2):
        super(iVAEEncoder, self).__init__()
        
        self.use_rff = use_rff
        self.latent_dim = latent_dim
        
        # Modified to take both x and z as inputs
        self.x_dim = input_dim
        self.z_dim = z_dim
        combined_input_dim = input_dim + z_dim
        
        # Apply RFF as the first step if specified
        if use_rff:
            rff_out_dim = combined_input_dim * rff_dim
            self.rff_layer = RandomFourierFeatures(combined_input_dim, rff_out_dim, sigma=rff_sigma)
            current_dim = rff_out_dim
        else:
            self.rff_layer = None
            current_dim = combined_input_dim
        
        # Create layers list for the encoder network
        layers = []
        
        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        
        # Create the encoder network
        self.encoder_layers = nn.Sequential(*layers)
        
        # For VAE, we output both mean and log variance of the latent distribution
        self.mean_projection = nn.Linear(current_dim, latent_dim)
        self.logvar_projection = nn.Linear(current_dim, latent_dim)
    
    def forward(self, x, z):
        # Combine inputs
        combined_input = torch.cat([x, z], dim=1)
        
        # Apply RFF if specified
        if self.use_rff and self.rff_layer is not None:
            combined_input = self.rff_layer(combined_input)
        
        # Apply encoder layers
        h = self.encoder_layers(combined_input)
        
        # Project to latent space parameters (mean and logvar)
        mean = self.mean_projection(h)
        logvar = self.logvar_projection(h)
        
        # Return the distribution parameters
        return mean, logvar
    
    def sample(self, x, z, n_samples=1):
        """Sample from the encoder distribution given x and z"""
        mean, logvar = self.forward(x, z)
        
        # Reparameterization trick for multiple samples if needed
        if n_samples > 1:
            # Expand mean and logvar for multiple samples
            batch_size = mean.size(0)
            mean = mean.unsqueeze(1).expand(batch_size, n_samples, self.latent_dim)
            logvar = logvar.unsqueeze(1).expand(batch_size, n_samples, self.latent_dim)
            
            # Reshape for easier computation
            mean = mean.reshape(batch_size * n_samples, self.latent_dim)
            logvar = logvar.reshape(batch_size * n_samples, self.latent_dim)
        
        # Reparameterization trick: d = mean + std * epsilon
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        d = mean + eps * std
        
        return d, mean, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super(Decoder, self).__init__()
        
        # Create a list of layers
        layers = []
        
        # Input layer
        current_dim = latent_dim
        
        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, d):
        return self.decoder(d)

class ConditionalPrior(nn.Module):
    def __init__(self, z_dim, hidden_dims, latent_dim):
        super(ConditionalPrior, self).__init__()
        
        # Create a list of layers
        layers = []
        
        # Input layer
        current_dim = z_dim
        
        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        
        # For conditional prior, output both mean and log variance
        self.prior_layers = nn.Sequential(*layers)
        self.mean_projection = nn.Linear(current_dim, latent_dim)
        self.logvar_projection = nn.Linear(current_dim, latent_dim)
    
    def forward(self, z):
        h = self.prior_layers(z)
        mean = self.mean_projection(h)
        logvar = self.logvar_projection(h)
        return mean, logvar
    
    def sample(self, z, n_samples=1):
        """Sample from the conditional prior distribution p(D|Z)"""
        mean, logvar = self.forward(z)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        d = mean + eps * std
        
        return d, mean, logvar

class iVAESystem(nn.Module):
    def __init__(self, input_dim, z_dim, encoder_hidden_dims, decoder_hidden_dims, prior_hidden_dims, 
                 latent_dim, use_rff=True, rff_sigma=1.0, rff_dim=2):
        super(iVAESystem, self).__init__()
        
        self.encoder = iVAEEncoder(
            input_dim=input_dim,
            z_dim=z_dim,
            hidden_dims=encoder_hidden_dims, 
            latent_dim=latent_dim, 
            use_rff=use_rff, 
            rff_sigma=rff_sigma,
            rff_dim=rff_dim
        )
        self.decoder = Decoder(latent_dim, decoder_hidden_dims, input_dim)
        self.prior = ConditionalPrior(z_dim, prior_hidden_dims, latent_dim)
        
        # Beta weighting for the KL divergence term
        self.beta = 1.0
        
    def forward(self, x, z):
        # Encode input to latent space D (mean and logvar of q(D|X,Z))
        d_mean, d_logvar = self.encoder(x, z)
        
        # Sample from q(D|X,Z) using reparameterization trick
        d_sample, _, _ = self.encoder.sample(x, z)
        
        # Reconstruct input from latent sample
        x_reconstructed = self.decoder(d_sample)
        
        # Get conditional prior parameters
        prior_mean, prior_logvar = self.prior(z)
        
        return x_reconstructed, d_sample, d_mean, d_logvar, prior_mean, prior_logvar
    
    def compute_losses(self, x, z):
        """Compute iVAE loss - standard ELBO with conditional prior"""
        # Forward pass
        x_reconstructed, d_sample, d_mean, d_logvar, prior_mean, prior_logvar = self.forward(x, z)
        
        # 1. Reconstruction loss (negative log-likelihood of p(X|D))
        # Using mean squared error as a proxy for Gaussian negative log-likelihood
        reconstruction_loss = F.mse_loss(x_reconstructed, x, reduction='sum') / x.size(0)
        
        # 2. KL divergence between q(D|X,Z) and p(D|Z)
        # KL(N(μ1,σ1²) || N(μ2,σ2²)) = log(σ2/σ1) + (σ1² + (μ1-μ2)²)/(2σ2²) - 1/2
        kl_divergence = 0.5 * torch.mean(
            prior_logvar - d_logvar 
            + (torch.exp(d_logvar) + (d_mean - prior_mean).pow(2)) / torch.exp(prior_logvar)
            - 1
        )
        
        # 3. Total iVAE loss = reconstruction_loss + kl_divergence
        # This is the standard ELBO with conditional prior
        total_loss = reconstruction_loss + kl_divergence
        
        return {
            'reconstruction_loss': reconstruction_loss,
            'kl_divergence': kl_divergence,
            'total_loss': total_loss
        }

# Training function for iVAE
def train_ivae(model, train_dataloader, val_dataloader=None, 
              optimizer=None, scheduler=None, scheduler_metric='val_loss',
              epochs=100, patience=10, device='cpu', 
              lambda_kl=1.0,    # Weight for KL divergence term
              save_best=True, model_path=None,
              printevery=10):
    """
    Train the iVAE system with validation and early stopping
    
    Parameters:
    - model: The iVAE model
    - train_dataloader: DataLoader for training data
    - val_dataloader: DataLoader for validation data (optional)
    - optimizer: Optimizer for training (default: Adam with lr=1e-3)
    - scheduler: Learning rate scheduler (optional)
    - scheduler_metric: Metric to use for scheduler stepping ('val_loss' or 'train_loss')
    - epochs: Maximum number of epochs to train
    - patience: Number of epochs to wait for improvement before early stopping
    - device: Device to train on ('cpu' or 'cuda')
    - lambda_kl: Weight for KL divergence term
    - save_best: Whether to save the best model
    - model_path: Path to save the best model
    - printevery: How often to print progress
    
    Returns:
    - history: Dictionary containing training and validation metrics
    """
    
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    model.to(device)
    
    # For early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    # For tracking metrics
    history = {
        'train_loss': [], 'train_rec_loss': [], 'train_kl_loss': [],
        'val_loss': [], 'val_rec_loss': [], 'val_kl_loss': [],
        'learning_rates': []
    }
    
    for epoch in range(epochs):
        # Track current learning rate
        if scheduler is not None:
            current_lr = scheduler.get_last_lr()[0]
            history['learning_rates'].append(current_lr)
            if epoch % printevery == 0:
                print(f"Current learning rate: {current_lr:.6f}")
        
        # Training phase
        model.train()
        total_loss = 0
        total_rec_loss = 0
        total_kl_loss = 0
        
        for batch_idx, (x, z) in enumerate(train_dataloader):
            x, z = x.to(device), z.to(device)
            
            optimizer.zero_grad()
            
            losses = model.compute_losses(x, z)
            
            # Total iVAE loss
            loss = losses['total_loss']
            
            loss.backward()
            
            # Optional: Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            total_rec_loss += losses['reconstruction_loss'].item()
            total_kl_loss += losses['kl_divergence'].item() * lambda_kl
        
        # Calculate average training metrics
        avg_loss = total_loss / len(train_dataloader)
        avg_rec_loss = total_rec_loss / len(train_dataloader)
        avg_kl_loss = total_kl_loss / len(train_dataloader)
        
        # Store training metrics
        history['train_loss'].append(avg_loss)
        history['train_rec_loss'].append(avg_rec_loss)
        history['train_kl_loss'].append(avg_kl_loss)
        
        # Update scheduler based on training loss if specified
        if scheduler is not None and scheduler_metric == 'train_loss':
            if isinstance(scheduler, (optim.lr_scheduler.ReduceLROnPlateau)):
                scheduler.step(avg_loss)
            else:
                scheduler.step()
                
        # Validation phase
        if val_dataloader is not None:
            model.eval()
            val_total_loss = 0
            val_total_rec_loss = 0
            val_total_kl_loss = 0
            
            with torch.no_grad():
                for val_x, val_z in val_dataloader:
                    val_x, val_z = val_x.to(device), val_z.to(device)
                    
                    val_losses = model.compute_losses(val_x, val_z)
                    
                    # Calculate validation loss
                    val_loss = val_losses['total_loss']
                    
                    val_total_loss += val_loss.item()
                    val_total_rec_loss += val_losses['reconstruction_loss'].item()
                    val_total_kl_loss += val_losses['kl_divergence'].item() * lambda_kl
            
            # Calculate average validation metrics
            avg_val_loss = val_total_loss / len(val_dataloader)
            avg_val_rec_loss = val_total_rec_loss / len(val_dataloader)
            avg_val_kl_loss = val_total_kl_loss / len(val_dataloader)
            
            # Store validation metrics
            history['val_loss'].append(avg_val_loss)
            history['val_rec_loss'].append(avg_val_rec_loss)
            history['val_kl_loss'].append(avg_val_kl_loss)
            
            # Update scheduler based on validation loss if specified
            if scheduler is not None and scheduler_metric == 'val_loss':
                if isinstance(scheduler, (optim.lr_scheduler.ReduceLROnPlateau)):
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()
            
            # Early stopping logic
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save the best model
                if model_path is not None and save_best:
                    torch.save(model.state_dict(), model_path)
                    print(f"Model saved to {model_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
                    
            # Print epoch stats with validation (only on specified intervals)
            if epoch % printevery == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.8f}, Val Loss: {avg_val_loss:.8f}')
                print(f'  Train: Rec={avg_rec_loss:.8f}, KL={avg_kl_loss:.8f}')
                print(f'  Val: Rec={avg_val_rec_loss:.8f}, KL={avg_val_kl_loss:.8f}')
        else:
            # For validation-less training, update scheduler if not using validation metric
            if scheduler is not None and scheduler_metric == 'train_loss':
                if isinstance(scheduler, (optim.lr_scheduler.ReduceLROnPlateau)):
                    scheduler.step(avg_loss)
                else:
                    scheduler.step()
            
            # Print epoch stats without validation (only on specified intervals)
            if epoch % printevery == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.8f}, Rec={avg_rec_loss:.8f}, KL={avg_kl_loss:.8f}')
    
    # Load the best model if early stopping occurred and best model was saved
    if val_dataloader is not None and save_best and model_path is not None and patience_counter < patience:
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded best model from {model_path}")

    print(f'FINAL Epoch {epoch+1}/{epochs}, Loss: {history["train_loss"][-1]:.8f}, Val Loss: {history["val_loss"][-1]:.8f}')
    print(f'  Train: Rec={history["train_rec_loss"][-1]:.8f}, KL={history["train_kl_loss"][-1]:.8f}')
    print(f'  VAL: Rec={history["val_rec_loss"][-1]:.8f}, KL={history["val_kl_loss"][-1]:.8f}')    
    
    return history