import numpy as np
from scipy.linalg import cholesky_banded, cho_solve_banded
    
class Covariance:
     
    def __init__(self, err, beta=None, **kwargs): 
        """
        Covariance matrix handler for observational uncertainties.

        The class provides utilities for modifying the covariance matrix,
        computing its determinant, and solving linear systems involving the
        covariance matrix.

        Parameters
        ----------
        err : ndarray
            Data uncertainties. If one-dimensional, covariance matrix is diagonal.

        beta : float, optional
            Multiplicative scaling factor applied to the data uncertainties.
        """
        self.err = err
        self.cov_reset() # set up covariance matrix
        self.cov_cholesky = None # initialize
        self.beta=beta # uncertainty scaling factor 'beta' should be added to kwargs

    def __call__(self,params,**kwargs):
        """
        Update the covariance matrix given the current parameter set.
        """
        self.cov_reset() # reset covariance matrix

    def cov_reset(self): # make diagonal covariance matrix from uncertainties
        """
        Reset the covariance matrix to the observational uncertainties.
        """
        self.cov = self.err**2
        self.is_matrix = (self.cov.ndim == 2) # = True if is matrix

    def add_data_err_scaling(self, beta): # Scale uncertainty with factor beta
        """
        Scale the observational uncertainties by ``beta^2``. 
        This can account for underestimated/overestimated errors.

        Parameters
        ----------
        beta : float
            Multiplicative uncertainty scaling factor.
        """
        if not self.is_matrix:
            self.cov *= beta**2
        else:
            self.cov[np.diag_indices_from(self.cov)] *= beta**2

    def add_model_err(self, model_err): # Add a model uncertainty term
        """
        Add a model uncertainty term to the covariance matrix.

        Parameters
        ----------
        model_err : ndarray or float
            Model uncertainty to be added in quadrature to the variance.
        """
        if not self.is_matrix:
            self.cov += model_err**2
        else:
            self.cov += np.diag(model_err**2)

    def get_logdet(self): # log of determinant
        """
        Compute the logarithm of the determinant of the covariance matrix.
        """
        self.logdet = np.sum(np.log(self.cov)) 
        return self.logdet
 
    def solve(self, b):
        """
        Solve the linear system ``cov * x = b``.
        This method computes the vector ``x = cov^{-1} b``.

        Parameters
        ----------
        b : ndarray
            Right-hand side vector.

        Returns
        -------
        ndarray
            Solution vector.
        """
        if self.is_matrix:
            return np.linalg.solve(self.cov, b)
        return 1/self.cov * b # if diagonal matrix, only invert the diagonal
    
    def get_dense_cov(self): 
        """
        Return the full covariance matrix.
        """
        if self.is_matrix:
            return self.cov
        return np.diag(self.cov) # get the errors from the diagonal
    
class CovGauss: # covariance matrix suited for Gaussian processes

    def __init__(self, err, separation, err_eff=None, max_separation=None, **kwargs):
        
        """
        Covariance matrix handler using Gaussian-process (GP).

        The GP uses a radial basis function (RBF) kernel with parameters
        controlling the amplitude and correlation length scale.

        Parameters
        ----------
        err : ndarray
            Measurement uncertainties for each pixel.

        separation : ndarray
            Pairwise pixel separation matrix.

        err_eff : float or ndarray, optional
            Effective uncertainty used to scale the GP amplitude.

        max_separation : float, optional
            Maximum separation distance used when constructing the
            banded covariance matrix.
        """
        # Pre-computed average error and wavelength separation
        self.err=err
        _, n_pixels = self.err.shape
        self.separation = np.abs(separation) # separation between pixels
        self.err_eff  = err_eff # average squared error between pixels
        self.cov_reset() # set up covariance matrix

        # Convert to banded matrices
        self.separation = self.get_banded(self.separation,n_pixels=n_pixels,
                                          max_value=max_separation,
                                         pad_value=1000) # pad high number bc will be truncated

    def get_banded(cls, array, n_pixels, max_value=None, pad_value=0):
        """
        Convert a dense matrix into a banded matrix representation.

        Parameters
        ----------
        array : ndarray
            Dense matrix to convert.

        max_value : float, optional
            Maximum allowed value for entries in a diagonal. Diagonals
            exceeding this threshold are truncated.

        pad_value : float, optional
            Value used to pad shorter diagonals.

        n_pixels : int, optional
            Number of pixels in the spectrum.

        Returns
        -------
        ndarray
            Banded matrix representation.
        """

        banded_array = [] # Make banded covariance matrix
        for k in range(n_pixels):
            diag_k = np.diag(array, k=k) # Retrieve the k-th diagonal  
            diag_k = np.concatenate((diag_k, pad_value*np.ones(k))) # Pad the diagonals to the same sizes
            if (diag_k == 0).all() and (k != 0): # There are no more non-zero diagonals coming
                break
            if max_value is not None:
                if (diag_k > max_value).all():
                    break
            banded_array.append(diag_k)
        return np.asarray(banded_array) # Convert to array for scipy
    
    def __call__(self,params,**kwargs):
        """
        Update the covariance matrix given the current parameter set.
        """
        self.cov_reset() # Reset covariance matrix
        a = 10**(params.get('log_a'))
        l = 10**(params.get('log_l'))
        if (a is not None) and (l is not None): 
            self.add_RBF_kernel(a=a,l=l,variance=self.err_eff, **kwargs) # add radial-basis function kernel
            
    def cov_reset(self): # Create covariance matrix from uncertainties
        """
        Reset the covariance matrix to the observational uncertainties.
        """
        self.cov = np.zeros_like(self.separation)
        self.cov[0] = self.err**2
        self.is_matrix = True

    def add_RBF_kernel(self, a, l, variance, trunc_dist=5, scale_GP_amp=True, **kwargs):
        """
        Add a radial-basis function (RBF) kernel to the covariance matrix.
        The kernel introduces correlated noise with amplitude ``a`` and
        correlation length scale ``l``.

        Parameters
        ----------
        a : float
            Square root of the GP amplitude.

        l : float
            Correlation length scale.

        variance : float or ndarray
            Effective variance used to scale the GP amplitude.

        trunc_dist : float, optional
            Distance beyond which the kernel is truncated to maintain
            sparsity in the covariance matrix.

        scale_GP_amp : bool, optional
            If True, scale the GP amplitude relative to the observational
            uncertainties.
        """
        
        w_ij = (self.separation < trunc_dist*l) # Hann window function to ensure sparsity
        GP_amp = a**2 # GP amplitude
        if scale_GP_amp: # Use amplitude as fraction of flux uncertainty
            if isinstance(variance, float):
                GP_amp *= variance**2
            else:
                GP_amp *= variance[w_ij]**2
        self.cov[w_ij] += GP_amp * np.exp(-(self.separation[w_ij])**2/(2*l**2)) # Gaussian radial-basis function kernel

    def get_cholesky(self):
        """
        Compute the banded Cholesky decomposition of the covariance matrix.
        """
        self.cov = self.cov[(self.cov!=0).any(axis=1),:] # input banded cov (is banded bc separation is)
        self.cov_cholesky = cholesky_banded(self.cov, lower=True) # banded Cholesky decomposition w scipy 

    def get_logdet(self): # log of determinant of banded Cholesky
        """
        Compute the logarithm of the determinant of the covariance matrix.
        """
        self.logdet = 2*np.sum(np.log(self.cov_cholesky[0]))
        return self.logdet
    
    def solve(self, b): # solve cov*x = b, for x (x = cov^{-1}*b) for banded Cholesky
        """
        Solve ``cov * x = b`` using the banded Cholesky decomposition.

        Parameters
        ----------
        b : ndarray
            Right-hand side vector.

        Returns
        -------
        ndarray
            Solution vector ``x = cov^{-1} b``.
        """
        return cho_solve_banded((self.cov_cholesky, True), b)
    
    def get_dense_cov(self):
        """
        Reconstruct the full dense covariance matrix from the banded form.
        """
        cov_full = np.zeros((self.cov.shape[1], self.cov.shape[1])) # Full covariance matrix
        for i, diag_i in enumerate(self.cov):
            if i != 0:
                diag_i = diag_i[:-i]
            cov_full += np.diag(diag_i, k=i) # Fill upper diagonals
            if i != 0:
                cov_full += np.diag(diag_i, k=-i) # Fill lower diagonals
        return cov_full