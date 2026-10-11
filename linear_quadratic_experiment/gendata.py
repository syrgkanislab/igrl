import numpy as np
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset, DataLoader

def gendata(g, f, theta, U, V, Q, n, m, k, r, rs=42, testsize=0.2):
    '''
    generate data
    n: number of sample
    m: number of features observed
    k: number of instrument
    r: true underlying latent dimension (assume k=r)
    rs: random seed for reproducibility
    g, f theta are models
    U, V, Q are functions of random noise

    Each sample generated according to the equation
    Z ~ N(0, I)
    D = g(Z) + u
    X = f(D) + v
    Y = theta(D) + Q(U, V)
    '''
    np.random.seed(rs)
    u = U(n, r)
    v = V(n, m)
    q = Q(u, v, n)

    # Z is dimension (n, k)
    Z = np.random.randn(n, k)
    # D is dimension (n, r)
    D = g(Z) + u
    #u = U(n, D.shape[1])
    #D += u
    # X is dimension (n, m)
    X = f(D) + v
    # Y is dimension (n, 1)
    q = Q(u, v, n)
    Y = theta(D) + q

    indices = np.arange(n)
    np.random.shuffle(indices)
    size = round(testsize*n)
    train_index = indices[size:]
    test_index = indices[:size]

    return {"Z":Z[train_index], "D":D[train_index], "X":X[train_index], "Y":Y[train_index],\
            "Z_test":Z[test_index], "D_test":D[test_index], "X_test":X[test_index], "Y_test":Y[test_index]}

# standarized data
def standardize_numpy_datasets(data):
    """
    Standardize multiple NumPy datasets in a dictionary format.
    Training data keys should be X, Y, Z
    Test data keys should be X_test, Y_test, Z_test
    """
    standardized_data = {}
    scalers = {}
    
    # First, standardize training data and save scalers
    for key in ['X', 'Y', 'Z']:
        if key in data:
            scaler = StandardScaler()
            standardized_data[key] = scaler.fit_transform(data[key])
            scalers[key] = scaler
    
    # Now standardize test data using the same scalers
    for key in ['X_test', 'Y_test', 'Z_test']:
        if key in data:
            base_key = key.replace('_test', '')
            if base_key in scalers:
                standardized_data[key] = scalers[base_key].transform(data[key])
            else:
                print(f"Warning: No scaler found for {key} (need {base_key} training data)")
    
    return standardized_data, scalers

def unstandardize_numpy_datasets(standardized_data, scalers):
    """
    Reverse the standardization process for multiple NumPy datasets.
    Uses the scalers created during standardization to transform data back to original scale.
    
    Parameters:
    - standardized_data: Dictionary containing standardized datasets
    - scalers: Dictionary containing the StandardScaler objects used for standardization
    
    Returns:
    - Dictionary containing unstandardized (original scale) datasets
    """
    unstandardized_data = {}
    
    # Process all keys in standardized_data
    for key in standardized_data:
        # Determine which scaler to use
        base_key = key.replace('_test', '')
        if base_key in scalers:
            unstandardized_data[key] = scalers[base_key].inverse_transform(standardized_data[key])
        else:
            print(f"Warning: No scaler found for {key}")
    return unstandardized_data

class Dataset(Dataset):
    def __init__(self, X, Z, Y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Z = torch.tensor(Z, dtype=torch.float32)
        if Y is not None:
            self.Y = torch.tensor(Y, dtype=torch.float32)
        else:
            self.Y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.Y is not None:
           return self.X[idx], self.Z[idx], self.Y[idx]
        else:
            return self.X[idx], self.Z[idx]
