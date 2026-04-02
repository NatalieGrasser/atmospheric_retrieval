import numpy as np
from scipy.special import loggamma # gamma function

class LogLikelihood:

    def __init__(self,retr_obj):
        """
        Log-likelihood evaluation based on Appendix D in Ruffio et al. (2019).
        DOI: https://doi.org/10.3847/1538-3881/ab4594

        Parameters
        ----------
        retr_obj : Retrieval class object with required attributes
        """

        inherit_attributes = ['n_parts','n_pixels','data_flux',
                                'mask_isfinite','data_wave','use_partial_pressure','pressure']
        for attr in inherit_attributes:  # list of attributes to pass down
            setattr(self, attr, getattr(retr_obj, attr))

        # if force photosphere to be in a certain pressure range
        if 'phot_fraction' in retr_obj.parameters.params:
            contr_attr = ['log_phot_upper','log_phot_lower',
                            'phot_fraction','phot_alpha']
            for attr in contr_attr:
                setattr(self, attr, retr_obj.parameters.params[attr])

        self.target_solar_metall = retr_obj.parameters.params['target_solar_metall']
        self.scale_flux   = retr_obj.parameters.params['scale_flux']
        self.scale_err    = retr_obj.parameters.params['scale_err']
        self.N_d_total    = self.mask_isfinite.sum() # number of degrees of freedom / valid datapoints
        self.alpha = 2 # from Ruffio+2019
        self.N_phi = 1 # number of linear scaling parameters
        self.sigma_p = 0.05 #bar
        
    def __call__(self, m_flux, Cov, **kwargs):
        """
        Calculate the total log-likelihood for a model spectrum.

        Parameters
        ----------
        m_flux : ndarray
            Model flux array with shape (n_parts, n_pixels).

        Cov : list
            List of covariance objects for each spectral segment,
            with required methods defined in covariance.py.

        **kwargs : dict
            Additional model quantities required for likelihood penalties:

            P_tot_surf : tuple
                Total and surface pressures for partial pressure mode.

            metall : float
                Atmospheric metallicity ([Fe/H]) of the model.

        Returns
        -------
        float
            Total log-likelihood value. Returns ``-np.inf`` for invalid models.
        """

        self.ln_L   = 0.0
        self.chi2_0 = 0.0
        self.phi = np.ones((self.n_parts, self.N_phi)) # store linear flux-scaling terms
        self.s2  = np.ones((self.n_parts)) # uncertainty-scaling
        self.m_flux_phi = m_flux # scaled model flux
        self.P_tot = kwargs['P_tot_surf'][0] # total pressure (for partial pressure retrievals)
        self.P_surf = kwargs['P_tot_surf'][1] # surface pressure (for partial pressure retrievals)
        self.FeH = kwargs['metall'] # metallicity

        for i in range(self.n_parts): # Loop over all segments

            mask_i = self.mask_isfinite[i,:] # mask out nans
            N_d = mask_i.sum() # Number of (valid) data points in this order/det pair
            if N_d == 0:
                continue
            data_flux_i = self.data_flux[i,mask_i] # data flux
            m_flux_i = m_flux[i,mask_i] # model flux

            if not np.all(np.isfinite(m_flux_i)): # unresolved issue, quick fix for now
                model_mask_i = np.isfinite(m_flux_i)
                m_flux_i[~model_mask_i] = 2.0 # to distinguish from normalized values
            
            if Cov[i].is_matrix:
                Cov[i].get_cholesky() # Retrieve a Cholesky decomposition
            if self.scale_flux: # Find the optimal phi-vector to match the observed spectrum
                self.m_flux_phi[i,mask_i],self.phi[i]=self.get_flux_scaling(data_flux_i, m_flux_i, Cov[i])

            residuals_phi = (self.data_flux[i] - self.m_flux_phi[i]) # Residuals wrt scaled model
            inv_cov_0_residuals_phi = Cov[i].solve(residuals_phi[mask_i])
            chi2_0 = np.dot(residuals_phi[mask_i].T, inv_cov_0_residuals_phi) # Chi-squared for the optimal linear scaling
            logdet_MT_inv_cov_0_M = 0

            inv_cov_0_M    = Cov[i].solve(m_flux_i) # Covariance matrix of phi
            MT_inv_cov_0_M = np.dot(m_flux_i.T, inv_cov_0_M)
            logdet_MT_inv_cov_0_M = np.log(MT_inv_cov_0_M) # (log)-determinant of the phi-covariance matrix

            if self.scale_err: 
                self.s2[i] = self.get_err_scaling(chi2_0, N_d) # Scale variance to maximize log-likelihood
            logdet_cov_0 = Cov[i].get_logdet()  # Get log of determinant (log prevents over/under-flow)

            # from Ruffio+2019
            self.ln_L += -1/2*(N_d-self.N_phi) * np.log(2*np.pi)+loggamma(1/2*(N_d-self.N_phi+self.alpha-1))

            # Add this order/detector to the total log-likelihood
            self.ln_L += -1/2*(logdet_cov_0+logdet_MT_inv_cov_0_M+(N_d-self.N_phi+self.alpha-1)*np.log(chi2_0))
            self.chi2_0 += chi2_0/self.s2[i]
        
        self.chi2_red = self.chi2_0/self.N_d_total # reduced chi^2

        if self.target_solar_metall:
            self.ln_L += self.penalty_metallicity(self.FeH)

        if self.use_partial_pressure:
            if self.P_tot > self.P_surf:
                self.ln_L -= np.inf # penalize unphysical solution

        if np.isfinite(self.ln_L)==False:
            return -np.inf
        else:
            return self.ln_L

    def get_flux_scaling(self, data_flux_i, m_flux_i, cov_i): 
        # Solve for linear scaling parameter phi: (M^T * cov^-1 * M) * phi = M^T * cov^-1 * d
        lhs = np.dot(m_flux_i.T, cov_i.solve(m_flux_i)) # Left-hand side
        rhs = np.dot(m_flux_i.T, cov_i.solve(data_flux_i)) # Right-hand side
        phi_i = rhs / lhs # Optimal linear scaling factor
        return np.dot(m_flux_i, phi_i), phi_i # Return scaled model flux + scaling factors

    def get_err_scaling(self, chi_squared_i_scaled, N_i):
        s2_i = np.sqrt(1/N_i * chi_squared_i_scaled)
        return s2_i # uncertainty scaling that maximizes log-likelihood

    def penalty_metallicity(self, metall_model, metall_target=0.0, metall_sigma= 0.1):
        """
        Gaussian prior penalizing deviations from target (solar) metallicity.
        (relevant for free chemistry, because we cannot set a prior on the metallicity)

        Parameters
        ----------
        metall_model : float
            Metallicity ([Fe/H]) predicted by the model.

        metall_target : float, optional
            Target metallicity value. Default is solar (0.0).

        metall_sigma : float, optional
            Standard deviation of the Gaussian prior.

        Returns
        -------
        float
            Log-likelihood penalty contribution.
        """
        
        deviation = metall_model - metall_target
        return -0.5 * (deviation / metall_sigma)**2