import numpy as np
from scipy.stats import norm
import re

class Parameters:

    def __init__(self, free_params, constant_params):
        """
        Class for storing and handling free and constant parameters 
        used in an atmospheric retrieval.

        Parameters
        ----------
        free_params : dict
            Dictionary defining the free parameters of the retrieval.
            Each entry must have the form:

                key : (prior, mathtext_label)

            where:

            prior : list, tuple, or dict
                Prior specification. Supported formats are:

                - ``[min, max]`` for uniform priors
                - ``{'type': 'gaussian', 'mu': μ, 'sigma': σ, 'bounds': (...)}``
                for Gaussian priors optionally truncated to bounds.

            mathtext_label : str
                LaTeX-style label used for plotting and visualization.

        constant_params : dict
            Dictionary containing fixed parameters used in the retrieval.
            These parameters are stored but not sampled.
        """

        self.params = {} # all parameters + their values
            
        # Separate the prior range from the mathtext label
        self.param_priors, self.param_mathtext = {}, {}
        for key_i, (prior_i, mathtext_i) in free_params.items():
            self.param_priors[key_i]   = prior_i
            self.param_mathtext[key_i] = mathtext_i

        self.param_keys = np.array(list(self.param_priors.keys())) # keys of free parameters
        self.n_params = len(self.param_keys) # number of free parameters
        self.ndim = self.n_params
        self.free_params=free_params
        self.constant_params=constant_params
        self.params.update(constant_params) # dictionary with constant parameter values

        # get all T knot keys, start with "T" followed by digits, exclude T0
        self.t_keys = [key for key in self.param_keys if re.fullmatch(r"T\d+", key) and key!='T0'] 
        self.t_keys = sorted(self.t_keys, key=lambda x: int(x[1:]))
        self.t_grad_keys = [key for key in self.param_keys if re.fullmatch(r"dlnT_dlnP_\d+", key)] 

        # for varchem
        self.var_keys=[]
        if any('log_H2O_' in key for key in self.param_keys):
            n_knots = sum(1 for key in self.param_keys if re.fullmatch(r'log_H2O_\d+', key))
            self.var_keys = [f'log_H2O_{int(i)}' for i in range(n_knots)][1:]
        if any('log_p_H2O_' in key for key in self.param_keys):
            n_knots = sum(1 for key in self.param_keys if re.fullmatch(r'log_p_H2O_\d+', key))
            self.var_keys = [f'log_p_H2O_{int(i)}' for i in range(n_knots)][1:]

        # for partial pressures, log pressure keys
        self.log_p_keys = [key for key in self.param_keys if 'log_p_' in key]
        self.p_max = 10**self.constant_params['log_P_upper']
            
    @staticmethod
    def prior_function(prior_info):
        """
        Prior transformation function, returning a function 
        that maps a value from the unit hypercube ``x ∈ [0,1]`` 
        to a parameter value according to the
        specified prior distribution.

        Supported priors include uniform and Gaussian distributions.

        Parameters
        ----------
        prior_info : list, tuple, or dict
            Specification of the prior distribution:

            - ``[min, max]`` or ``(min, max)`` for a uniform prior
            - ``{'type': 'gaussian', 'mu': μ, 'sigma': σ, 'bounds': (...)}``
            for a Gaussian prior optionally truncated to bounds.

        Returns
        -------
        function
            Function mapping a unit hypercube variable ``x ∈ [0,1]`` to
            a physical parameter value.
        """
        
        # Uniform prior case
        if isinstance(prior_info, (list, tuple)) and len(prior_info) == 2:
            bounds = prior_info
            return lambda x: x * (bounds[1] - bounds[0]) + bounds[0]

        # Gaussian prior case
        elif isinstance(prior_info, dict) and prior_info.get('type') == 'gaussian':
            mu = prior_info['mu']
            sigma = prior_info['sigma']
            bounds = prior_info.get('bounds', None)

            def prior_fn(x):
                val = norm.ppf(np.clip(x, 1e-6, 1 - 1e-6), loc=mu, scale=sigma)
                if bounds is not None:
                    val = np.clip(val, bounds[0], bounds[1])
                return val

            return prior_fn

        else:
            raise ValueError(f"Unknown prior_info format: {prior_info}")
    
    def __call__(self, cube, ndim=None, nparams=None):
        """
        Transform a unit hypercube sample into physical parameter values.

        This method converts parameters sampled in the unit hypercube
        into physical parameter values according to the specified priors.

        Parameters
        ----------
        cube : ndarray
            Parameter vector in the unit hypercube with values in
            ``[0,1]``. Modified to contain physical parameter values.

        ndim : int, optional
            Dimensionality of the sampled parameter space.

        nparams : int, optional
            Total number of parameters in the sampler.

        Returns
        -------
        ndarray
            Copy of the original unit-cube sample before transformation.

        Notes
        -----
        The transformed parameter values are stored internally in
        ``self.params`` for later access.
        """

        self.p_sum = 0.0
        # cube is vector of length nparams, values [0,1]
        if (ndim is None) and (nparams is None):
            self.cube_copy = cube
        else:
            self.cube_copy = np.array(cube[:ndim])

        for i, key_i in enumerate(self.param_keys):
            
            # some params handled separately later, cube[i] must stay [0,1] for those
            if key_i not in self.t_keys + self.var_keys + self.log_p_keys:  
                cube[i] = self.prior_function(self.param_priors[key_i])(cube[i]) 
            
            # allow only minor temperature inversions
            if key_i in self.t_keys: # as long as order in dict T0,...
                cube[i]=self.prior_function([cube[i-1]*0.5,cube[i-1]*1.05])(cube[i]) # like in Zhang+2021 on 2M0355

            if key_i in self.var_keys: # abundance decrease to top, allow minor increase only
                cube[i]=self.prior_function([cube[i-1]*1.1,self.param_priors[key_i][-1]])(cube[i])

            if key_i in self.log_p_keys:
                cube[i] = self.prior_function(self.param_priors[key_i])(cube[i])
                self.p_sum += 10**cube[i]

            self.params[key_i] = cube[i] # add free parameter values to parameter dictionary

        return self.cube_copy
