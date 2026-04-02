from astropy import units as u

setup_params = {        
    
        'color': 'deepskyblue', # color of retrieval output
        'instrument': 'JWST',
        'data_as_points': True, # show data as points, else as continuous curve

        #'input_spectrum': './spectra/Sing_2024_Fig2_WASP107b_transit_spec_data.csv', # in folder ./targetname/, .txt file with 3 columns: wavelength, flux, uncertainties
        'input_spectrum': './spectra/NIRCam_F322W2_Fiducial_Eureka_Spectrum.txt', # in folder ./targetname/, .txt file with 3 columns: wavelength, flux, uncertainties
        'wavelength_unit': u.um,
        'opa_mode': 'c-k', # lbl for high-res, c-k for low-res

        'emission_or_transmission': 'transmission',
        'ref_pressure': 0.01, # for transmission spectroscopy

        # the following must be given in astropy units to avoid issues
        'R_p_prior': {'type': 'gaussian', 'mu': 0.94, 'sigma': 0.01, 'bounds': (0.85, 1.15)}, # radius of planet, u.R_jup
        'M_p_prior': {'type': 'gaussian', 'mu': 0.1, 'sigma': 0.01, 'bounds': (0.08,0.12)}, # u.M_jup,
        'R_s': 0.67, # radius of star, u.R_sun
        
        # set priors. if param not used, set to None
        'rv_prior': None, # radial velocity in km/s
        'vsini_prior': None, # projected rotational velocity in km/s
        'epsilon_limb_prior': None, # limb darkening, [0,1]
        'log_g_prior': None,

        'log_P_lower': -5, # min log10 pressure
        'log_P_upper': 1, # max log10 pressure
        'pt_points': 3, # number of PT nodes
        'T_prior': [1000,4000], # bottom of atmosphere

        'species_names': ['H2O','CO','CH4','HCN','NH3','CO2','SO2','H2S'], # species to include
        'const_species': ['SO2','HCN'], # altitude-constant species in equchem, separate
        #'gaussian_species': ['SO2'],

        'cloud_species': None, # Mg2SiO4 / MgSiO3
        'cloud_mode': 'gray', # None / gray /  amorphous / crystalline (last two for physical clouds)
        'gray_cloud_slope': True, # wavelength dependence of cloud

        'normalize_spectrum': False, # normalize by median
        'scale_flux': False, # compute optimal linear scaling factor between model and data
        'scale_err': False,

        # implemented quench pressures for quenched equilibrium chemistry
        # quench_species1_species2_ ... can have multiple for groups of species
        'quench_CO': True,
        'quench_CH4': True,
        'quench_H2O': True,
        'quench_CO2': True,
        'quench_NH3': True,

        # comparison PT profile to plot
        'comparison_PT_path': './WASP-107b/Sing+2024_PT.txt',
        'comparison_PT_color': 'tab:purple',
        'comparison_PT_linestyle': 'dashdot',
        'comparison_PT_label': 'Sing+2024 $P$-$T$',
                
        }