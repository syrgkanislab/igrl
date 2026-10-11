import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

def linearhsic(z, e):
    # 3. Correlation between z and e using Linear kernel HSIC
    batch_size = z.size(0)
    K_z = torch.matmul(z, z.T)
    K_e = torch.matmul(e, e.T)  # Kernel matrix for errors
    H = torch.eye(batch_size, device=z.device) - (1.0/batch_size) * torch.ones(batch_size, batch_size, device=z.device)
    K_z_centered = torch.matmul(torch.matmul(H, K_z), H)
    K_e_centered = torch.matmul(torch.matmul(H, K_e), H)
    hsic = torch.trace(torch.matmul(K_z_centered, K_e_centered)) / (batch_size**2)
    return hsic

def transfer_weights(source_model, target_model, initialization_strategy='xavier'):
    """
    Transfer weights from a model with smaller bottleneck to one with larger bottleneck.
    
    Args:
        source_model: AutoencoderSystem with smaller latent_dim
        target_model: AutoencoderSystem with larger latent_dim
        initialization_strategy: How to initialize new weights ('xavier', 'normal', 'zeros')
    
    Returns:
        target_model: Model with transferred weights
    """
    
    def copy_layer_weights(source_layer, target_layer):
        """Copy weights between layers of the same type"""
        if isinstance(source_layer, nn.Linear) and isinstance(target_layer, nn.Linear):
            # Copy weights and biases, handling dimension mismatches
            source_weight = source_layer.weight.data
            source_bias = source_layer.bias.data if source_layer.bias is not None else None
            
            target_weight = target_layer.weight.data
            target_bias = target_layer.bias.data if target_layer.bias is not None else None
            
            # Determine copy dimensions
            copy_out_dim = min(source_weight.shape[0], target_weight.shape[0])
            copy_in_dim = min(source_weight.shape[1], target_weight.shape[1])
            
            # Copy the overlapping portion
            target_weight[:copy_out_dim, :copy_in_dim] = source_weight[:copy_out_dim, :copy_in_dim]
            
            if source_bias is not None and target_bias is not None:
                target_bias[:copy_out_dim] = source_bias[:copy_out_dim]
                
        elif isinstance(source_layer, nn.BatchNorm1d) and isinstance(target_layer, nn.BatchNorm1d):
            # Copy batch norm parameters
            copy_dim = min(source_layer.num_features, target_layer.num_features)
            
            target_layer.weight.data[:copy_dim] = source_layer.weight.data[:copy_dim]
            target_layer.bias.data[:copy_dim] = source_layer.bias.data[:copy_dim]
            target_layer.running_mean.data[:copy_dim] = source_layer.running_mean.data[:copy_dim]
            target_layer.running_var.data[:copy_dim] = source_layer.running_var.data[:copy_dim]
    
    # Transfer RandomFourierFeatures (these should be identical)
    if hasattr(source_model.encoder, 'rff_layer') and hasattr(target_model.encoder, 'rff_layer'):
        if source_model.encoder.rff_layer is not None and target_model.encoder.rff_layer is not None:
            target_model.encoder.rff_layer.weight.data = source_model.encoder.rff_layer.weight.data.clone()
            target_model.encoder.rff_layer.bias.data = source_model.encoder.rff_layer.bias.data.clone()
    
    # Transfer encoder layers (except the final latent projection)
    source_encoder_layers = list(source_model.encoder.encoder_layers.children())
    target_encoder_layers = list(target_model.encoder.encoder_layers.children())
    
    for source_layer, target_layer in zip(source_encoder_layers, target_encoder_layers):
        copy_layer_weights(source_layer, target_layer)
    
    # Handle encoder's latent projection layer (this connects to bottleneck)
    source_latent_proj = source_model.encoder.latent_proj
    target_latent_proj = target_model.encoder.latent_proj
    
    # Copy what we can from the latent projection
    copy_in_dim = min(source_latent_proj.weight.shape[1], target_latent_proj.weight.shape[1])
    copy_out_dim = min(source_latent_proj.weight.shape[0], target_latent_proj.weight.shape[0])
    
    target_latent_proj.weight.data[:copy_out_dim, :copy_in_dim] = \
        source_latent_proj.weight.data[:copy_out_dim, :copy_in_dim]
    
    if source_latent_proj.bias is not None:
        target_latent_proj.bias.data[:copy_out_dim] = source_latent_proj.bias.data[:copy_out_dim]
    
    # Initialize remaining weights in latent projection
    if target_latent_proj.weight.shape[0] > copy_out_dim:
        remaining_weights = target_latent_proj.weight.data[copy_out_dim:, :]
        if initialization_strategy == 'xavier':
            nn.init.xavier_uniform_(remaining_weights)
        elif initialization_strategy == 'normal':
            nn.init.normal_(remaining_weights, 0, 0.02)
        
        if target_latent_proj.bias is not None:
            remaining_bias = target_latent_proj.bias.data[copy_out_dim:]
            nn.init.zeros_(remaining_bias)
    
    # Handle decoder (first layer takes latent_dim as input)
    source_decoder_layers = list(source_model.decoder.decoder.children())
    target_decoder_layers = list(target_model.decoder.decoder.children())
    
    # First decoder layer needs special handling due to latent dimension change
    first_source_layer = source_decoder_layers[0]
    first_target_layer = target_decoder_layers[0]
    
    if isinstance(first_source_layer, nn.Linear) and isinstance(first_target_layer, nn.Linear):
        # Copy weights for the overlapping input dimensions (latent_dim)
        copy_in_dim = min(first_source_layer.weight.shape[1], first_target_layer.weight.shape[1])
        copy_out_dim = min(first_source_layer.weight.shape[0], first_target_layer.weight.shape[0])
        
        first_target_layer.weight.data[:copy_out_dim, :copy_in_dim] = \
            first_source_layer.weight.data[:copy_out_dim, :copy_in_dim]
        
        if first_source_layer.bias is not None:
            first_target_layer.bias.data[:copy_out_dim] = first_source_layer.bias.data[:copy_out_dim]
        
        # Initialize remaining input connections
        if first_target_layer.weight.shape[1] > copy_in_dim:
            remaining_weights = first_target_layer.weight.data[:, copy_in_dim:]
            if initialization_strategy == 'xavier':
                nn.init.xavier_uniform_(remaining_weights)
            elif initialization_strategy == 'normal':
                nn.init.normal_(remaining_weights, 0, 0.02)
    
    # Copy remaining decoder layers
    for source_layer, target_layer in zip(source_decoder_layers[1:], target_decoder_layers[1:]):
        copy_layer_weights(source_layer, target_layer)
    
    # Handle predictor if it exists
    # Predictor only predicts first z_dim columns of D, so architecture doesn't change
    if (source_model.predictor is not None and target_model.predictor is not None):
        source_predictor_layers = list(source_model.predictor.predictor.children())
        target_predictor_layers = list(target_model.predictor.predictor.children())
        
        # Copy all predictor layers directly - no dimension handling needed
        # since predictor architecture is independent of latent_dim
        for source_layer, target_layer in zip(source_predictor_layers, target_predictor_layers):
            copy_layer_weights(source_layer, target_layer)
    
    print(f"Successfully transferred weights from model with latent_dim={source_model.encoder.latent_proj.out_features} "
          f"to model with latent_dim={target_model.encoder.latent_proj.out_features}")
    print(f"Predictor outputs {target_model.predictor.predictor[-1].out_features if target_model.predictor else 'N/A'} "
          f"dimensions (first z_dim columns of D)")
    
    return target_model


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

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, use_rff=True, rff_sigma=1.0, rff_dim=10):
        super(Encoder, self).__init__()
        
        self.use_rff = use_rff
        
        # Apply RFF as the first step if specified
        if use_rff:
            # Multiply input dimension by rff_dim factor
            rff_out_dim = input_dim * rff_dim
            self.rff_layer = RandomFourierFeatures(input_dim, rff_out_dim, sigma=rff_sigma)
            # Update the current_dim to the RFF output dimension
            current_dim = rff_out_dim
        else:
            self.rff_layer = None
            current_dim = input_dim
        
        # Create layers list for the encoder network
        layers = []
        
        # Add hidden layers after potential RFF transformation
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        
        # Create the encoder network
        self.encoder_layers = nn.Sequential(*layers)
        
        # Final projection to latent space
        self.latent_proj = nn.Linear(current_dim, latent_dim)
    
    def forward(self, x):
        # Apply RFF if specified (as the first step)
        if self.use_rff and self.rff_layer is not None:
            x = self.rff_layer(x)
        
        # Apply encoder layers
        x = self.encoder_layers(x)
        
        # Project to latent space
        x = self.latent_proj(x)
            
        return x
        
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

class Predictor(nn.Module):
    def __init__(self, z_dim, hidden_dims, d_dim):
        super(Predictor, self).__init__()
        
        # Create a list of layers
        layers = []
        
        # Input layer
        current_dim = z_dim
        
        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, z_dim))
        
        self.predictor = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.predictor(z)

class AutoencoderSystem(nn.Module):
    def __init__(self, input_dim, encoder_hidden_dims, decoder_hidden_dims, predictor_hidden_dims, 
                 latent_dim, z_dim, use_rff=True, rff_sigma=1.0, rff_dim = 10, model_type = -1):
        super(AutoencoderSystem, self).__init__()
        
        self.encoder = Encoder(input_dim, encoder_hidden_dims, latent_dim, use_rff, rff_sigma, rff_dim = rff_dim)
        self.decoder = Decoder(latent_dim, decoder_hidden_dims, input_dim)
        self.model_type = model_type
        if model_type >= 0:
            self.predictor = Predictor(z_dim, predictor_hidden_dims, latent_dim)
        else:
            self.predictor = None
        
        
    def forward(self, x, z):
        # Encode input to latent space D
        d = self.encoder(x)
        
        # Reconstruct input from latent space
        x_reconstructed = self.decoder(d)
        
        if self.model_type >= 0:
            # Predict D from Z
            d_predicted = self.predictor(z)
        else:
            d_predicted = None
        
        return x_reconstructed, d, d_predicted
    
    def compute_losses(self, x, z):
        x_reconstructed, dv, d_predicted = self.forward(x, z)
        zdim = z.shape[1]
        d = dv[:,:zdim]
        v = dv[:,zdim:]
        
        # 1. Reconstruction loss
        reconstruction_loss = F.mse_loss(x_reconstructed, x)
        
        # 2. Prediction loss (Z predicting D)
        if self.model_type >= 0:
            prediction_loss = F.mse_loss(d_predicted, d)
        else:
            prediction_loss = torch.tensor(0.0, device=x.device)
        
        if self.model_type >= 1:
            correlation_loss = linearhsic(z, d - d_predicted)
        else:
            correlation_loss = torch.tensor(0.0, device=x.device)

        if self.model_type >= 2:
            correlation_lossv = linearhsic(z, v)
        else: 
            correlation_lossv = torch.tensor(0.0, device=x.device)
        
        if self.model_type >= 3:
            correlation_lossd = linearhsic(d, v)
        else: 
            correlation_lossd = torch.tensor(0.0, device=x.device)
        
        return {
            'reconstruction_loss': reconstruction_loss,
            'prediction_loss': prediction_loss,
            'correlation_loss': correlation_loss,
            'correlation_lossv': correlation_lossv,
            'correlation_lossd': correlation_lossd,
            'total_loss': reconstruction_loss + prediction_loss + correlation_loss
        }

# Training function
# Training function with validation, early stopping, and learning rate scheduler
def train(model, train_dataloader, val_dataloader=None, 
          optimizer=None, scheduler=None, scheduler_metric='val_loss',
          epochs=100, patience=10, device='cpu', 
          lambda_rec=1.0, lambda_pred=1.0, lambda_corr=1.0,
          lambda_corrv=1.0, lambda_corrd=1.0,
          save_best=True, model_path=None, printevery=10):
    """
    Train the autoencoder system with validation, early stopping, and learning rate scheduling
    
    Parameters:
    - model: The AutoencoderSystem model
    - train_dataloader: DataLoader for training data
    - val_dataloader: DataLoader for validation data (optional)
    - optimizer: Optimizer for training (default: Adam with lr=1e-3)
    - scheduler: Learning rate scheduler (optional)
    - scheduler_metric: Metric to use for scheduler stepping ('val_loss' or 'train_loss')
    - epochs: Maximum number of epochs to train
    - patience: Number of epochs to wait for improvement before early stopping
    - device: Device to train on ('cpu' or 'cuda')
    - lambda_rec, lambda_pred, lambda_corr: Weights for the different loss components
    - save_best: Whether to save the best model
    - model_path: Path to save the best model
    
    Returns:
    - history: Dictionary containing training and validation metrics
    """
    # sanity check
    if (lambda_corr == 0 or lambda_corrv==0 or lambda_corrd == 0):
        print("some corr coefficient is set to 0, skipping the  corresponding hsic calculation")

    if (lambda_pred == 0 and lambda_corr == 0 and lambda_corrv==0 and lambda_corrd == 0):
        assert model.model_type == -1
    elif (lambda_pred > 0 and lambda_corr == 0 and lambda_corrv==0 and lambda_corrd == 0):
        assert model.model_type == 0
    elif (lambda_pred > 0 and lambda_corr > 0 and lambda_corrv==0 and lambda_corrd == 0):
        assert model.model_type == 1
    elif (lambda_pred > 0 and lambda_corr > 0 and lambda_corrv > 0 and lambda_corrd == 0):
        assert model.model_type == 2
    elif (lambda_pred > 0 and lambda_corr > 0 and lambda_corrv > 0 and lambda_corrd > 0):
        assert model.model_type == 3
    else:
        print("not supported type, return")
        return
    

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    model.to(device)
    
    # For early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    # For tracking metrics
    history = {
        'train_loss': [], 'train_rec_loss': [], 'train_pred_loss': [], 
        'train_corr_loss': [], 'train_max_corr': [], 'train_corr_lossv': [], 'train_corr_lossd': [],
        'val_loss': [], 'val_rec_loss': [], 'val_pred_loss': [], 
        'val_corr_loss': [], 'val_max_corr': [], 'val_corr_lossv': [], 'val_corr_lossd': [],
        'learning_rates': []
    }

    for epoch in range(epochs):
        # Track current learning rate
        if scheduler is not None:
            current_lr = scheduler.get_last_lr()[0]
            history['learning_rates'].append(current_lr)
            print(f"Current learning rate: {current_lr:.6f}")
        
        # Training phase
        model.train()
        total_loss = 0
        total_rec_loss = 0
        total_pred_loss = 0
        total_corr_loss = 0
        total_corr_lossv = 0
        total_corr_lossd = 0
        
        for batch_idx, (x, z) in enumerate(train_dataloader):
            x, z = x.to(device), z.to(device)
            
            optimizer.zero_grad()
            
            losses = model.compute_losses(x, z)
            
            # Weighted sum of losses
            loss = lambda_rec * losses['reconstruction_loss'] + \
                   lambda_pred * losses['prediction_loss'] + \
                   lambda_corr * losses['correlation_loss'] + \
                   lambda_corrv * losses['correlation_lossv'] + \
                   lambda_corrd * losses['correlation_lossd']
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_rec_loss += losses['reconstruction_loss'].item()
            total_pred_loss += losses['prediction_loss'].item()
            total_corr_loss += losses['correlation_loss'].item()
            total_corr_lossv += losses['correlation_lossv'].item()
            total_corr_lossd += losses['correlation_lossd'].item()
        
        # Calculate average training metrics
        avg_loss = total_loss / len(train_dataloader)
        avg_rec_loss = total_rec_loss / len(train_dataloader)
        avg_pred_loss = total_pred_loss / len(train_dataloader)
        avg_corr_loss = total_corr_loss / len(train_dataloader)
        avg_corr_lossv = total_corr_lossv / len(train_dataloader)
        avg_corr_lossd = total_corr_lossd / len(train_dataloader)
        
        # Store training metrics
        history['train_loss'].append(avg_loss)
        history['train_rec_loss'].append(avg_rec_loss)
        history['train_pred_loss'].append(avg_pred_loss)
        history['train_corr_loss'].append(avg_corr_loss)
        history['train_corr_lossv'].append(avg_corr_lossv)
        history['train_corr_lossd'].append(avg_corr_lossd)
        
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
            val_total_pred_loss = 0
            val_total_corr_loss = 0
            val_total_corr_lossv = 0
            val_total_corr_lossd = 0
            
            with torch.no_grad():
                for val_x, val_z in val_dataloader:
                    val_x, val_z = val_x.to(device), val_z.to(device)
                    
                    val_losses = model.compute_losses(val_x, val_z)
                    
                    # Calculate weighted validation loss
                    val_loss = lambda_rec * val_losses['reconstruction_loss'] + \
                               lambda_pred * val_losses['prediction_loss'] + \
                               lambda_corr * val_losses['correlation_loss'] + \
                                lambda_corrd * val_losses['correlation_lossd'] + \
                                lambda_corrv * val_losses['correlation_lossv']
                                
                    val_total_loss += val_loss.item()
                    val_total_rec_loss += val_losses['reconstruction_loss'].item()
                    val_total_pred_loss += val_losses['prediction_loss'].item()
                    val_total_corr_loss += val_losses['correlation_loss'].item()
                    val_total_corr_lossv += val_losses['correlation_lossv'].item()
                    val_total_corr_lossd += val_losses['correlation_lossd'].item()
            
            # Calculate average validation metrics
            avg_val_loss = val_total_loss / len(val_dataloader)
            avg_val_rec_loss = val_total_rec_loss / len(val_dataloader)
            avg_val_pred_loss = val_total_pred_loss / len(val_dataloader)
            avg_val_corr_loss = val_total_corr_loss / len(val_dataloader)
            avg_val_corr_lossv = val_total_corr_lossv / len(val_dataloader)
            avg_val_corr_lossd = val_total_corr_lossd / len(val_dataloader)
            
            # Store validation metrics
            history['val_loss'].append(avg_val_loss)
            history['val_rec_loss'].append(avg_val_rec_loss)
            history['val_pred_loss'].append(avg_val_pred_loss)
            history['val_corr_loss'].append(avg_val_corr_loss)
            history['val_corr_lossv'].append(avg_val_corr_lossv)
            history['val_corr_lossd'].append(avg_val_corr_lossd)
            
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
            
            # Print epoch stats with validation
            if epoch%printevery == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
                print(f'  Train: Rec={avg_rec_loss:.4f}, Pred={avg_pred_loss:.4f}, HSIC={avg_corr_loss:.4f}, HSIC_V={avg_corr_lossv:.4f}, HSIC_D={avg_corr_lossd:.4f}')
                print(f'  Val: Rec={avg_val_rec_loss:.4f}, Pred={avg_val_pred_loss:.4f}, HSIC={avg_val_corr_loss:.4f}, HSIC_V={avg_val_corr_lossv:.4f}, HSIC_D={avg_val_corr_lossd:.4f}')
        
        else:
            # For validation-less training, update scheduler if not using validation metric
            if scheduler is not None and scheduler_metric == 'train_loss':
                if isinstance(scheduler, (optim.lr_scheduler.ReduceLROnPlateau)):
                    scheduler.step(avg_loss)
                else:
                    scheduler.step()
            
            # Print epoch stats without validation
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Rec={avg_rec_loss:.4f}, Pred={avg_pred_loss:.4f}, HSIC={avg_corr_loss:.4f}')
    
    # Load the best model if early stopping occurred and best model was saved
    if val_dataloader is not None and save_best and patience_counter < patience:
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded best model from {model_path}")


    print(f'FINAL Epoch {epoch+1}/{epochs}, Loss: {history['train_loss'][-1]:.8f}, Val Loss: {history['val_loss'][-1]:.8f}')
    print(f'  Train: Rec={history["train_rec_loss"][-1]:.8f}, PRED={history["train_pred_loss"][-1]:.8f}, HSIC={history["train_corr_loss"][-1]:.8f}, HSIC_V={history["train_corr_lossv"][-1]:.8f}, HSIC_D={history["train_corr_lossd"][-1]:.8f}')
    print(f'  VAL: Rec={history["val_rec_loss"][-1]:.8f}, PRED={history["val_pred_loss"][-1]:.8f}, HSIC={history["val_corr_loss"][-1]:.8f}, HSIC_V={history["val_corr_lossv"][-1]:.8f}, HSIC_D={history["val_corr_lossd"][-1]:.8f}')    
    
    return history