from astropy import units as u

setup_params = {        
    
        'instrument': 'JWST',
        'const_efficiency_mode': True,

        #'input_spectrum': './spectra/Sing_2024_Fig2_WASP107b_transit_spec_data.csv', # in folder ./targetname/, .txt file with 3 columns: wavelength, flux, uncertainties
        #'input_spectrum': './spectra/NIRCam_F322W2_Fiducial_Eureka_Spectrum.txt', # in folder ./targetname/, .txt file with 3 columns: wavelength, flux, uncertainties
        #'input_spectrum':  './spectra/NIRCam_F444W_Fiducial_Eureka_Spectrum.txt', # in folder ./targetname/, .txt file with 3 columns: wavelength, flux, uncertainties
        
        # all JWST
        'input_spectrum': ['./spectra/native_NIRISS.dat', './spectra/native_NRS1.dat','./spectra/native_NRS2.dat'],
        'parts_labels': ['NIRISS','NRS1','NRS2'],
        
        # all data
        #'input_spectrum': ['./spectra/Sing_2024_Fig2_WASP107b_transit_spec_data.csv', './spectra/NIRCam_F322W2_Fiducial_Eureka_Spectrum.txt','./spectra/NIRCam_F444W_Fiducial_Eureka_Spectrum.txt','WFC3_G102_Fiducial_Pegasus_Spectrum.txt','WFC3_G141_Fiducial_Pegasus_Spectrum.txt','MIRI_LRS_Fiducial_Eureka_Spectrum.txt'],
        
        'wavelength_unit': u.um,
        'opa_mode': 'lbl', # lbl for high-res, c-k for low-res
        'lbl_opacity_sampling': 30, # only for high-res

        'emission_or_transmission': 'transmission',
        'ref_pressure': 0.01, # for transmission spectroscopy

        # the following must be given in astropy units to avoid issues
        'R_p_prior': {'type': 'gaussian', 'mu': 0.19, 'sigma': 0.01, 'bounds': (0.16, 0.21)}, # radius of planet, u.R_jup
        'M_p_prior': {'type': 'gaussian', 'mu': 0.015, 'sigma': 0.005, 'bounds': (0.01,0.02)}, # u.M_jup,
        'R_s': 0.37, # radius of star, u.R_sun
        
        # set priors. if param not used, set to None
        'rv_prior': [10,40], # radial velocity in km/s
        'vsini_prior': None, # projected rotational velocity in km/s
        'epsilon_limb_prior': None, # limb darkening, [0,1]
        'log_g_prior': None,

        'log_P_lower': -5, # min log10 pressure
        'log_P_upper': 1, # max log10 pressure
        'pt_points': 1, # number of PT nodes
        'T_prior': [200,800], # bottom of atmosphere

        'species_names': ['H2O','CO','CH4','NH3','CO2','SO2','CS2'], # species to include
        'const_species': ['SO2'], # altitude-constant species in equchem, separate
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
        #'quench_CH4_CO': True,
        'quench_H2O': True,
        'quench_CO2': True,
        'quench_NH3': True,

        # comparison PT profile to plot
        #'comparison_PT_path': './WASP-107b/Sing+2024_PT.txt',
        #'comparison_PT_color': 'tab:purple',
        #'comparison_PT_linestyle': 'dashdot',
        #'comparison_PT_label': 'Sing+2024 $P$-$T$',

        # plotting settings
        'data_as_points': True, # show data as points, else as continuous curve
        'data_markersize': 1,
        'data_elw': 0.005,
        #'data_alpha': 0.6,
        'model_lw': 1,
        'model_color': 'darken' # darken data color

        }

colors = ['teal','orange','salmon']

if isinstance(setup_params['input_spectrum'],list):
    setup_params.update({'offset_prior': [-0.1,0.1],
                         'num_offset_prior': 2,
                        'num_radtrans_objects': 1, # make just 1 radtrans obj for all bc continuous wave coverage
                         'data_color': colors})
    if 'parts_labels' not in setup_params:
        labels=[]
        for spec in setup_params['input_spectrum']:
            labels.append('_'.join(spec[10:].split('_')[:2]))
        setup_params.update({'parts_labels': labels})
        setup_params.update({'color': colors[0]})

else:
    data = setup_params['input_spectrum']
    if 'NIRISS' in data:
        setup_params.update({'color':colors[0]})
    elif 'S1' in data:
        setup_params.update({'color':colors[1]})
    elif 'S2' in data:
        setup_params.update({'color':colors[2]})