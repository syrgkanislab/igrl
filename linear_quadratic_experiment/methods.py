import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from abc import ABC, abstractmethod

# Abstract base class that defines the common interface
class BaseModel(ABC):
    @abstractmethod
    def fit(self, Z, X, y, r):
        """
        Fit the model to the training data.
        
        Parameters:
        -----------
        Z : array-like of shape (n_samples, k)
            Instruments.
        X : array-like of shape (n_samples, m)
            Features.
        y : array-like of shape (n_samples, 1)
            Outcome.
        r : scaler
            latent dimension
            
        Returns:
        --------
        self : object
            Returns self.
        """
        pass


# IV regression model
class IVRegression(BaseModel):
    '''
    Run IV (Z^TX)^-1Z^TY
    Run TSLS
    
    Z: instrumental var
    X: endo var
    Y: outcome
    '''
    def __init__(self, intercept=False):
        self.theta = None
        self.intercept = intercept
        self.first = None
        self.second = None
    def fit(self, Z, X, Y):
        self.first = LinearRegression().fit(Z, X)
        self.second = LinearRegression().fit(self.first.predict(Z), Y)
        self.theta = self.second.coef_.T
        return self



# Proposed method via SVD
class LIRR(BaseModel):
    '''
    Solve C from X ~ CZ using least squares
    Perform SVD and take B = U, A = Sigma V.T
    Solve theta using IV regression on Y ~ B.TX

    Z: instrumental var
    X: features
    Y: outcome
    r: latent dimension

    Assume all vars are mean-centered and standardized.
    '''
    
    def __init__(self, r, intercept=False):
        self.r = r
        self.A = None
        self.B = None
        self.ivmodel = None
        self.intercept = intercept
        
    def fit(self, Z, X, y):
        # First least square to get C
        regr = LinearRegression(fit_intercept=False)
        regr.fit(Z, X)
        C = regr.coef_
        
        # then svd to get A and B
        U, S, Vh = np.linalg.svd(C, full_matrices=False)
        B_hat = U[:,:self.r]
        A_hat = np.diag(S[:self.r]) @ Vh[:self.r,]
        self.A = A_hat
        self.B = B_hat

        # fit iv regression on the estimated latent
        D = X @ B_hat
        
        self.ivmodel = IVRegression(self.intercept)
        self.ivmodel.fit(Z, D, y)
        
        return self

    def gettheta(self):
        return self.ivmodel.theta
        
    def encode(self, X):
        return X @ self.B

    def decode(self, D):
        return D @ (self.B).T


# Naive method PCA followed by IVregression
class PCAMethod(BaseModel):
    '''
    PCA followed by IVregression

    Z: instrumental var
    X: features
    Y: outcome
    r: latent dimension

    Assume all vars are mean-centered and standardized.
    '''
    
    def __init__(self, r, intercept=False):
        self.r = r
        self.PCA = None
        self.intercept = intercept
        
    def fit(self, Z, X, y):
        # First PCA, assume X is standarized
        pca = PCA(n_components=self.r)
        X_reduced = pca.fit_transform(X)
        self.PCA = pca
        
        # then ivregression
        self.ivmodel = IVRegression(self.intercept)
        self.ivmodel.fit(Z, X_reduced, y)
        
        return self

    def gettheta(self):
        return self.ivmodel.theta
        
    def encode(self, X):
        return self.PCA.transform(X)

    def decode(self, D):
        return self.PCA.inverse_transform(D)