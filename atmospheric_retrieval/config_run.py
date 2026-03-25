#from petitRADTRANS.config import petitradtrans_config_parser
#petitradtrans_config_parser.set_input_data_path('/net/lem/data2/pRT3_formatted')

def init_retrieval(target,
                    PT_type,
                    chemistry,
                    Nlive,
                    evtol,
                    folder_suffix=''):
    
    """
    Initialize and configure a spectral retrieval object.

    This function prepares all components required to run an atmospheric
    retrieval, including the target definition, pressure-temperature (P-T)
    parameterization, chemical abundance model, cloud treatment, and sampling
    configuration. It constructs the parameter space, applies target-specific 
    settings, and returns a fully initialized `Retrieval` object.

    Uniform priors can be defined by [lower_limit, upper_limit]
    Gaussian priors can be defined by
        {'type': 'gaussian', 'mu': mean, 'sigma': std, 'bounds': (lower_bound, upper_bound)}

    Parameters
    ----------
    target : str
        Name of the target object, whose data should exist in a folder of the
        same name, used to initialize a `Target` instance containing 
        observational and instrument metadata.

    PT_type : str
        Parameterization of the atmospheric pressure-temperature profile.
        Supported options include:
        - 'PTknot' : temperature nodes at fixed pressure levels
        - 'PTgrad' : temperature gradients between layers
        - 'PTguillot' : Guillot (2010) analytic radiative equilibrium profile

    chemistry : str
        Chemical abundance model used in the retrieval. Supported options:
        - 'freechem'  : freely retrieved altitude-constant abundances
        - 'equchem'   : equilibrium chemistry
        - 'quequchem' : quenched equilibrium chemistry
        - 'flexequ'   : equilibrium chemistry with scaling factors
        - 'varchem'   : pressure-varying abundances for selected species

    Nlive : int
        Number of live points used in the nested sampling algorithm.

    evtol : float
        Evidence tolerance parameter controlling the stopping criterion
        of the nested sampler.

    cloud_mode : str or None, optional
        Cloud model used in the retrieval. Examples include:
        - 'gray' : gray cloud deck parameterization
        - 'MgSiO3' : physically motivated MgSiO3 cloud model
        - None : no clouds included.

    use_GP : bool, optional
        If True, include Gaussian Process hyperparameters to model correlated
        noise in the data.

    folder_suffix : str, optional
        Additional suffix appended to the retrieval output directory name.

    Returns
    -------
    Retrieval
        A fully configured `Retrieval` object containing the target,
        parameter definitions, species list, and retrieval configuration.
    """

    import os
    import numpy as np
    import pandas as pd
    import pathlib
    import importlib.util
    from astropy import units as u
    import warnings
    warnings.filterwarnings("ignore", message="Mean of empty slice") # ignore warning for empty orders
    warnings.filterwarnings("ignore", message="All-NaN slice encountered") # ignore warning for empty orders
    os.environ['OMP_NUM_THREADS'] = '1' # to avoid using too many CPUs, important for MPI
    
    from retrieval import Retrieval
    from parameters import Parameters
    from target import Target

    default_params = {

        'color': 'deepskyblue', # color of retrieval output
        'data_as_points': False, # show data as points, else as continuous curve

        'input_spectrum': None, # in folder ./targetname/, .txt file with 3 columns: wavelength, flux, uncertainties
        'wavelength_unit': u.nm,
        'flux_unit': None, # None if normalized, else set a unit with astropy quantities, or 'photons' for photons/s/m^3
        'opa_mode': 'lbl', # lbl for high-res, c-k for low-res
        'lbl_opacity_sampling': 3, # only for high-res
        'const_efficiency_mode': True, # constant efficiency mode
        'sampling_efficiency': 0.5,
        'use_GP': False, # use Gaussian processes for uncertainty modeling
        'emission_or_transmission': 'emission',
        'ref_pressure': 0.01, # for transmission spectroscopy

        # the following must be given in astropy units to avoid issues
        'R_p_prior': None, # radius of planet, u.R_jup
        'M_p_prior': None, # u.M_jup,
        'R_s': None, # radius of star, u.R_sun
        'distance': None, # distance in astropy units to avoid issues
        'obstime': None, # observing time in seconds (only relevant when not normalizing)
        
        # set priors. if param not used, set to None
        'rv_prior': [-20,50], # radial velocity in km/s
        'vsini_prior': [0,40], # projected rotational velocity in km/s
        'epsilon_limb_prior': [0,1], # limb darkening, [0,1]
        'log_g_prior': {'type': 'gaussian', 'mu': 4.1, 'sigma': 0.15, 'bounds': (3.6, 4.6)},
        'VMR_prior': [-15,-1], # VMR range for retrieved species
        'partial_P_prior': [-15,-1], # only used when retrieving partial pressures
        'T_prior': [1000,10000], # temperature range
        'metallicity_prior': [-1,1], # used in equilibrium chemistry
        'C_to_O_prior': [0.1,1], # used in equilibrium chemistry
        
        'n_atm_layers': 50, # number of atmosphere layers
        'Tgrad_lower': 0.0, # temperature gradient lower lim -> no temperatur inversions
        'log_P_lower': -6, # min log10 pressure
        'log_P_upper': 2, # max log10 pressure
        'pt_points': 5, # number of PT nodes
        'p_node_skew': 1, # if =1, pressure points equally distributed in log10-pressure-space

        'species_names': ['H2O','CO'], # species to include
        'vary_species': [], # pressure-dependent VMR for selected species
        'var_n': 3, # number of pressure nodes for pressure-dependent VMRs, chemistry='varchem'
        'const_species': [], # altitude-constant species in equchem, separate
        'gaussian_species': [],
        'offset_equ': 2, # VMR deviations (in dex) from equilibrium chemistry, for chemistry='flexequ'
        'use_partial_pressure': False, # retrieve partial pressure instead of VMRs if True

        'cloud_species': None, # Mg2SiO4 / MgSiO3
        'cloud_mode': None, # None / gray /  amorphous / crystalline (last two for physical clouds)
        'gray_cloud_slope': False, # wavelength dependence of cloud

        'disk': False, # retrieve a disk around the object (veiling)
        'remove_continuum': False, # remove continuum of spectrum with np.fft
        'normalize_spectrum': True, # normalize by median
        'scale_flux': True, # compute optimal linear scaling factor between model and data

        # if Guillot PT used, these priors must be defined (not used  other PT type)
        'T_int_prior': [20,200],  # internal temperature
        'T_equ_prior': [200,400], # equilibrium temperature
        'log_k_IR_prior': [-3,0.5],
        'log_gamma_prior': [-2,1],

        # implemented quench pressures for quenched equilibrium chemistry
        # quench_species1_species2_ ... can have multiple for groups of species
        'quench_H2O_CO_CH4': False,

        # additional retrievals with leaving out one species at a time from given list
        'evidence_retrievals_species': [],

        # comparison PT profile to plot
        'comparison_PT_path': None,
        'comparison_PT_color': 'cornflowerblue',
        'comparison_PT_linestyle': 'dashdot',
        'comparison_PT_label': None,
                
        ##### for running tests, have no effect by default ######
        'fix_solar_metall': False,
        'fix_solar_CO': False,
        'fix_logg': None,
        'target_solar_metall': False, # penalize metallicity in lnL for freechem
        'gaussian_prior_solar_metall': False, # prior for equchem
        'logg': None, # provide value
        'fix_PT': None, # provide (pressure, temperature)
        'force_cloud': False,
        }

    # Start with default params
    use_params = default_params.copy()

    project_path = f'{os.getcwd()}/{target}'
    custom_config_file = pathlib.Path(f'{project_path}/config.py')

    # overwrite default setup if custom setup defined in target folder
    if custom_config_file.exists():
        spec = importlib.util.spec_from_file_location("custom_config", str(custom_config_file))
        custom_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_config)
        if hasattr(custom_config, "setup_params"):
            use_params.update(custom_config.setup_params)

    target = Target(target, properties=use_params)
    species_info = target.species_info

    # initialize parameters dictionaries
    constant_params = use_params
    free_params = {}
    chemistry_params = {}
    pt_params = {}

    species_names = use_params['species_names']
    vary_species = use_params['vary_species']
    fixed_species = [s for s in species_names if s not in vary_species]
    log10_pressure_range = [use_params['log_P_lower'],use_params['log_P_upper']]   

    ########## create parameters dict ###############

    if use_params['rv_prior'] is not None:
        free_params.update({'rv': (use_params['rv_prior'],r'$v_{\rm rad}$')})

    if use_params['log_g_prior'] is not None and use_params['fix_logg'] is None:
        free_params.update({'log_g': (use_params['log_g_prior'],r'$\log g$')})
    else:
        constant_params['log_g'] = use_params['fix_logg']

    if use_params['epsilon_limb_prior'] is not None:
        free_params.update({'epsilon_limb': (use_params['epsilon_limb_prior'],r'$\epsilon_\mathrm{limb}$')})

    if use_params['vsini_prior'] is not None:
        free_params.update({'vsini': (use_params['vsini_prior'],r'$v$ sin$i$')})

    if use_params['R_p_prior'] is not None:
        free_params.update({'R_p': (use_params['R_p_prior'],r'$R_{\rm p} [R_{\rm Jup}]$')})

    if use_params['M_p_prior'] is not None:
        free_params.update({'M_p': (use_params['M_p_prior'],r'$M_{\rm p} [M_{\rm Jup}]$')})

    pt_params={}
    if PT_type=='PTknot' and use_params['fix_PT'] is None: 
        for n in range(use_params['pt_points']):
            pt_params[f'T{n}']=(use_params['T_prior'],rf'$T_{n}$') # T0 = bottom of atmosphere

    # not working correctly yet
    elif PT_type=='PTgrad' and use_params['fix_PT'] is None:
        for n in range(use_params['pt_points']):
            pt_params[f'dlnT_dlnP_{n}']=([use_params['Tgrad_lower'],0.4],rf'$\nabla T_{n}$')
        pt_params['T0']= (use_params['T_prior'], r'$T_0$') # T0 = bottom of atmosphere
       
    elif PT_type=='PTguillot' and use_params['fix_PT'] is None:
        pt_params['T_int']= (use_params['T_int_prior'], r'$T_\mathrm{int}$') # internal
        pt_params['T_equ']= (use_params['T_equ_prior'], r'$T_\mathrm{equ}$') # blackbody
        pt_params['log_k_IR']= (use_params['log_k_IR_prior'], r'log $\kappa_\mathrm{IR}$')
        pt_params['log_gamma']= (use_params['log_gamma_prior'], r'log $\gamma$')
    
    free_params.update(pt_params)

    # if equilibrium chemistry, define [Fe/H], C/O, and isotopologue ratios
    if chemistry in ['equchem','quequchem','flexequ']:
        chemistry_params={}
        if chemistry in ['equchem','quequchem']:
            if use_params['fix_solar_metall'] or chemistry=='flexequ':
                constant_params['Fe/H'] = 0.0 # solar
            elif use_params['gaussian_prior_solar_metall']:
                chemistry_params.update({'Fe/H': ({'type': 'gaussian', 'mu': 0.0, 'sigma': 0.1,
                                            'bounds': (-0.5, 0.5)}, r'[Fe/H]')})
            else:
                chemistry_params.update({'Fe/H': (use_params['metallicity_prior'], r'[Fe/H]')})
            if use_params['fix_solar_CO'] or chemistry=='flexequ':
                constant_params['C/O'] = 0.59 # solar, Asplund 2021
            else:
                chemistry_params.update({'C/O':(use_params['C_to_O_prior'], r'C/O')})
            if use_params['const_species']!=[]:
                for species_i in use_params['const_species']:
                    chemistry_params[f"log_{species_i}"]=(use_params['VMR_prior'],rf"log {species_info.loc[species_i,'mathtext_name']}")

        if chemistry=='flexequ': # vary equchem abundances by constant factor
            for species in species_names:
                if species not in ['13CO','C17O','C18O','H2(18)O']: # isotopes etrieved separately
                    chemistry_params[f'log_a_{species}'] = ([-use_params['offset_equ'],use_params['offset_equ']],
                                                        fr'log $\alpha$ {species_info.loc[species,"mathtext_name"]}')

        if '13CO' in species_names:
            chemistry_params['log_12CO_13CO_ratio'] = ([1,6], r'log $\mathrm{^{12}CO/^{13}CO}$')
        if 'C18O' in species_names:
            chemistry_params['log_C16O_C18O_ratio'] = ([1,6], r'log $\mathrm{C^{16}O/C^{18}O}$')
        if 'H2(18)O' in species_names:
            chemistry_params['log_H216O_H218O_ratio'] = ([1,6], r'log $\mathrm{H_2^{16}O/H_2^{18}O}$')
        if 'C17O' in species_names: 
            chemistry_params['log_C16O_C17O_ratio'] = ([1,6], r'log $\mathrm{C^{16}O/C^{17}O}$')
        
        if chemistry=='quequchem': # quenched equilibrium chemistry_params
            for key in use_params:
                if key.startswith("quench_"):
                    if use_params[key] == True:
                        # Extract species list
                        species_list = key.split("quench_")[1].split("_")

                        # Build parameter name
                        param_name = "log_Pqu_" + "_".join(species_list)

                        # Build mathtext label
                        math_names = [
                            species_info.loc[sp, 'mathtext_name']
                            for sp in species_list]
                        math_label = ",".join(math_names)

                        # Store prior
                        chemistry_params[param_name] = (
                            log10_pressure_range,
                            rf'log P$_{{qu}}$({math_label})')
        
    # if free chemistry_params, define VMRs
    elif chemistry=='freechem': 
        if use_params['use_partial_pressure']:
            for species_i in species_names:
                chemistry_params[f"log_p_{species_i}"]=(use_params['partial_P_prior'],rf"log $p$ {species_info.loc[species_i,'mathtext_name']}")
        else:
            for species_i in species_names:
                chemistry_params[f"log_{species_i}"]=(use_params['VMR_prior'],rf"log {species_info.loc[species_i,'mathtext_name']}")
            if 'H-' in species_names:
                chemistry_params[f"log_e-"]=(use_params['VMR_prior'],"log e$^{-}$")
                
    elif chemistry=='varchem':
        for species_i in vary_species:
            for i in range(use_params['var_n']):
                if use_params['use_partial_pressure']:
                    chemistry_params[f"log_p_{species_i}_{i}"]=(use_params['partial_P_prior'],rf"log $p$ {species_info.loc[species_i,'mathtext_name']} {i}")
                else:
                    chemistry_params[f"log_{species_i}_{i}"]=(use_params['VMR_prior'],rf"log {species_info.loc[species_i,'mathtext_name']} {i}")
        for species_i in fixed_species:
            if use_params['use_partial_pressure']:
                chemistry_params[f"log_p_{species_i}"]=(use_params['partial_P_prior'],rf"log $p$ {species_info.loc[species_i,'mathtext_name']}")
            else:
                chemistry_params[f"log_{species_i}"]=(use_params['VMR_prior'],rf"log {species_info.loc[species_i,'mathtext_name']}")

    if use_params['use_partial_pressure']: # also for H2 and He
        chemistry_params[f"log_p_H2"]=([-1,-0.05],rf"log $p$ H$_2$")
        chemistry_params[f"log_p_He"]=([-2,-0.2],rf"log $p$ He")  

    if use_params['gaussian_species']!=[]:
        for species_i in use_params['gaussian_species']:
            chemistry_params.pop(f"log_{species_i}", None) # remove if defined by freechem
            label = rf"{species_info.loc[species_i,'mathtext_name']}"
            chemistry_params[f"log_p_{species_i}"] = (log10_pressure_range,f"log $P$ {label}") 
            chemistry_params[f"std_{species_i}"] = ([0.1,1],f"$\sigma_P$ {label}")
            chemistry_params[f"log_maxVMR_{species_i}"] = (use_params['VMR_prior'],f"max log {label}") 
        
    if use_params['disk']==True: # retrieve veiling factor as linear function
        free_params.update({'phi_disk': ([0,1], r'$r_k(\lambda_\mathrm{mid}$'),#r'$\phi_\mathrm{disk}$'), # contribution 
                            'T_disk': ([300,1500], r'$T_\mathrm{disk}$')}) # temperature
        #free_params.update({'log_k_rk': ([-6,-2], r'log $k_\mathrm{rk}$'), # slope log 1/nm
                            #'d_rk': ([0,5], r'$d_\mathrm{rk}$')}) # intercept (rk at bluest wl)

    if use_params['cloud_mode']=='gray': # gray cloud deck
        if use_params['force_cloud']:
            cloud_params={'log_opa_base_gray': ([0,2], r'log $\kappa_{\mathrm{cl},0}$'), # opacity at cloud base
                        'log_P_base_gray': ([-1,0.5], r'log $P_{\mathrm{cl},0}$'), # pressure of gray cloud deck
                        'fsed_gray': ([0,10], r'$f_\mathrm{sed}$')} # sedimentation parameter for particles
        else:
            cloud_params={'log_opa_base_gray': ([-10,3], r'log $\kappa_{\mathrm{cl},0}$'), # opacity at cloud base
                        'log_P_base_gray': (log10_pressure_range, r'log $P_{\mathrm{cl},0}$'), # pressure of gray cloud deck
                        'fsed_gray': ([0,20], r'$f_\mathrm{sed}$')} # sedimentation parameter for particles
        if use_params['gray_cloud_slope']:
            cloud_params['cloud_slope'] = ([-4,0], r'$\gamma_{\mathrm{cl}}$')
        free_params.update(cloud_params)

    if use_params['cloud_species'] not in [[],None]:
        cloud_species = use_params['cloud_species'] if isinstance(use_params['cloud_species'],list) else [use_params['cloud_species']]
        constant_params['cloud_species'] = cloud_species # make list for looping logic
        cloud_params = {'log_Kzz':([5,15],r'log $K_{zz}$'),
                        'sigma_cloud': ([0.8,1.5], r'$\sigma_\mathrm{cl}$')} # width of the log-normal particle distribution
        cloud_species_pRT = []
        # amorphous or crystalline?
        mode = use_params['cloud_mode'] if use_params['cloud_mode'] in ['amorphous','crystalline'] else 'amorphous'
        constant_params['cloud_mode'] = mode
        for cloud_species_i in cloud_species:
            mathtext = species_info.loc[f'{cloud_species_i}_{mode}','mathtext_name']
            pRT_name = species_info.loc[f'{cloud_species_i}_{mode}',f'{use_params["opa_mode"]}_name']
            cloud_species_pRT.append(pRT_name)
            cloud_params[f'fsed_{cloud_species_i}'] = ([0,20], r'$f_\mathrm{sed}$'+f'{mathtext}')  # sedimentation parameter for particles
            cloud_params[f'log_P_base_{cloud_species_i}'] = (log10_pressure_range, r'log $P_{\mathrm{cl},0}$'+f'{mathtext}')
            cloud_params[f'base_MF_log_{cloud_species_i}'] = ([use_params['VMR_prior'][0],-2], r'log MF$_{\mathrm{cl},0}$'+f'{mathtext}') # mass fraction at cloud base
        free_params.update(cloud_params)
        
    if use_params['use_GP']==True: # uncertainty scaling through Gaussian processes
        GP_params={'log_a': ([-1,1], r'$\log\ a$'),
                'log_l': ([-3,1], r'$\log\ l$')}
        free_params.update(GP_params)

    free_params.update(chemistry_params)
    parameters = Parameters(free_params, constant_params)
    cube = np.random.rand(parameters.n_params)
    parameters(cube)

    retrieval_object = Retrieval(target=target,
                        parameters=parameters,
                        Nlive=Nlive,evtol=evtol,
                        chemistry=chemistry,
                        PT_type=PT_type,
                        folder_suffix=folder_suffix)

    return retrieval_object

if __name__ == '__main__':
    from datetime import datetime
    import sys, shutil, os
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    import matplotlib
    matplotlib.use('Agg') # disable interactive plotting
    from datetime import datetime
    from pprint import pformat

    class SuppressOutput:
        def __enter__(self):
            if rank != 0:
                # Redirect both stdout and stderr to devnull
                self._stdout = os.dup(1)
                self._stderr = os.dup(2)
                self.devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(self.devnull, 1)
                os.dup2(self.devnull, 2)
        def __exit__(self, exc_type, exc_val, exc_tb):
            if rank != 0:
                # Restore original stdout and stderr
                os.dup2(self._stdout, 1)
                os.dup2(self._stderr, 2)
                os.close(self.devnull)

    with SuppressOutput(): # print only once when running parallel

        start = datetime.now()
        print("Start time:", start.strftime("%Y-%m-%d %H:%M:%S"))

        # pass configuration as command line argument
        # example: config_run.py 2M0355 freechem PTgrad 200 5
        target = sys.argv[1] # 2M0355 / 2M1425 / test
        chemistry = sys.argv[2] # freechem / equchem / quequchem / flexequ / varchem
        PT_type = sys.argv[3] # PTknot / PTgrad / PTguillot
        Nlive=int(sys.argv[4]) # number of live points (integer)
        evtol=float(sys.argv[5]) # evidence tolerance (float)
        folder_suffix = sys.argv[6] if len(sys.argv)>6 else '' # for note

        retrieval_object = init_retrieval(target=target,PT_type=PT_type,
                                 chemistry=chemistry,Nlive=Nlive,
                                evtol=evtol,folder_suffix=folder_suffix)

        if rank == 0:  # only first process saves a copy of config file
            timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")
            fname = os.path.join(retrieval_object.output_dir, f"params_{timestamp}.txt")
            with open(fname, "w") as f:
                f.write("Setup dictionary:\n")
                f.write(pformat(retrieval_object.parameters.params))
                f.write("\n\nFree parameters:\n")
                f.write(pformat(retrieval_object.parameters.free_params))

        retrieval_object.run_retrieval()

        end = datetime.now()
        print("End time:  ", end.strftime("%Y-%m-%d %H:%M:%S"))
        dt_minutes = (end - start).total_seconds() / 60
        print(f"Elapsed time: {dt_minutes:.2f} minutes")