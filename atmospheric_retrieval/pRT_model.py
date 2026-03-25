import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import CubicSpline
from PyAstronomy.pyasl import fastRotBroad
# SWITCH BROADENING TO https://github.com/Adolfo1519/RotBroadInt for large wavelength range and fast rotators
from astropy import constants as const
from astropy import units as u
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
import pathlib
import gc
from utils import *
import warnings
import re
from utils import *
from scipy.linalg import LinAlgWarning
from scipy.integrate import simps
#from scipy.constants import sigma, h, c, k as sc.sigma, sc.h, sc.c, sc.k
import scipy.constants as sc
from petitRADTRANS.physics import temperature_profile_function_guillot_global

warnings.filterwarnings(action='ignore', category=LinAlgWarning) # occasional

import getpass
if getpass.getuser() == "grasser": # when runnig from LEM
    import matplotlib
    matplotlib.use('Agg') # disable interactive plotting
    path_tables = '/net/lem/data2/regt/fastchem_tables'
elif getpass.getuser() == "natalie": # when testing from my laptop
    os.environ['pRT_input_data_path'] = "/home/natalie/.local/lib/python3.8/site-packages/petitRADTRANS/input_data_std/input_data"
    path_tables = '/home/natalie/fastchem_tables'

class pRT_spectrum:

    """
    Generate spectra using petitRADTRANS for a given retrieval state.

    Handles P-T profiles, chemistry (free, equilibrium, variable), radiative
    transfer, and post-processing (RV shift, broadening, interpolation).
    """

    gc_n=0

    def __init__(self,
                 retr_obj, # retrieval object
                 contribution=False, # only for plotting radtrans.contr_em
                 interpolate=True, # interpolate onto data wavlength grid
                 PT=None, # option to set a fixed PT profile to use
                 add_species=None, # get equ abund of species not included
                 leave_out=[]):


        
        inherit_attributes = ['data_wave','instrument','species_pRT','name','pRT_key',
                              'chemistry','radtrans_objects','n_atm_layers','species_info',
                              'pressure','PT_type','cloud_mode','spectral_resolution',
                              'mask_isfinite','data_flux','use_partial_pressure','species_names']

        for attr in inherit_attributes:  # list of attributes to pass down
            setattr(self, attr, getattr(retr_obj, attr, None))

        if self.instrument=='CRIRES':
            self.n_orders=retr_obj.n_orders
            self.n_dets=retr_obj.n_dets
            self.n_pixels = retr_obj.n_pixels

        self.params=retr_obj.parameters.params
        self.p_node_skew = self.params['p_node_skew'] if self.params['p_node_skew']!=False else 1.
        self.interpolate=interpolate
        self.add_species = add_species

        if 'R_p' in self.params and 'M_p' in self.params:
            M_p = self.params['M_p'] * u.M_jup 
            R_p = self.params['R_p'] * u.R_jup
            g = (const.G * M_p / R_p**2).to(u.m / u.s**2) # gravity in SI
            self.gravity = g.to(u.cm / u.s**2).value  # Convert to cgs (cm/s^2)
        else:
            self.gravity = 10**self.params['log_g']

        if PT is None and self.params['fix_PT'] is None:
            self.temperature = self.make_pt() #P-T profile
        elif PT is not None and self.params['fix_PT'] is None:
            self.pressure, self.temperature = PT
        elif PT is None and self.params['fix_PT'] is not None:
            fixed_P, fixed_T = self.params['fix_PT']
            self.temperature = np.interp(self.pressure, fixed_P, fixed_T)
        self.vbary = retr_obj.target.vbary
            
        self.give_absorption_opacity=None # for gray cloud
        self.int_opa_cloud = np.zeros_like(self.pressure)
        self.contribution=contribution
        self.leave_out = leave_out if isinstance(leave_out, list) else [leave_out]
        self.mass_fractions = {}

        # for physical clouds
        self.cloud_species = self.params.get('cloud_species', None)
        self.cloud_f_sed = {} # dictionary of f_sed for each cloud species
        self.cloud_particle_size_std = None
        self.Kzz = None
        if self.cloud_species:
            self.Kzz = 10**self.params['log_Kzz'] * np.ones_like(self.pressure)
            self.cloud_particle_size_std = self.params['sigma_cloud']
            self.setup_physical_clouds(self.cloud_species)

        self.unphysical_params = False # flag when problems
        self.P_tot = 0.0
        self.P_surf = max(self.pressure)

        if 'log_k_rk' in self.params:
            wl_mid = np.median(self.data_wave)
            self.rk_func = lambda x: 10**self.params['log_k_rk']*np.array(x-wl_mid) + self.params['d_rk']

        if 'T_disk' in self.params:
            wl_mid = np.median(self.data_wave)
            #self.rk_func = lambda x: 10**self.params['log_k_rk']*np.array(x-wl_mid) + self.params['d_rk']
            self.BB_0 = planck_lambda_um(self.params['T_disk'],wl_mid*1e-3)
            self.rk_func = lambda x: self.params['phi_disk']*planck_lambda_um(self.params['T_disk'],x*1e-3)/self.BB_0

        if self.chemistry=='freechem': # use free chemistry with defined VMRs
            if self.use_partial_pressure:
                self.mass_fractions, self.CO, self.FeH = self.free_chemistry_use_partial_pressure(self.species_pRT,self.params)
                self.MMW = self.mass_fractions['MMW']
                self.VMR_dict = self.get_VMR_dict(self.mass_fractions)
            else:
                self.mass_fractions, self.CO, self.FeH = self.free_chemistry(self.species_pRT,self.params)
            self.MMW = self.mass_fractions['MMW']
        elif self.chemistry=='varchem':   
            self.mass_fractions, self.CO, self.FeH = self.var_chemistry(self.species_pRT,self.params)
            self.MMW = self.mass_fractions['MMW']
            self.VMR_dict = self.get_VMR_dict(self.mass_fractions)

        elif self.chemistry in ['equchem','quequchem','flexequ']: # use equilibium chemistry
            self.species_hill = retr_obj.species_hill
            self.mass_fractions = self.equ_chemistry(self.species_pRT,self.params)
            # update mass_fractions with isotopolog ratios
            if any(key in self.params for key in ['13CO','C17O','C18O','H2(18)O',
                                                  'log_12CO_13CO_ratio','log_C16O_C18O_ratio',
                                                  'log_H216O_H218O_ratio','log_C16O_C17O_ratio']):
                self.mass_fractions = self.get_isotope_mass_fractions(self.species_names,self.species_pRT,
                                                                      self.mass_fractions,self.params) 
            self.MMW = self.mass_fractions['MMW']
            # get new VMR dict, updated with isotopologs
            self.VMR_dict = self.get_VMR_dict(self.mass_fractions)

            # keeps crashing for equchem??
            pRT_spectrum.gc_n+=1
            if pRT_spectrum.gc_n>20: # make it more efficient by not running it every time
                gc.collect()
                pRT_spectrum.gc_n=0  

    def get_VMR_dict(self,mass_fractions):
        """
        Convert mass fractions to volume mixing ratios (VMRs).

        Parameters
        ----------
        mass_fractions : dict
            Mass fraction dictionary including 'MMW'.

        Returns
        -------
        dict
            VMRs keyed by species name.
        """
        VMR_dict={}
        MMW=self.MMW
        for pRT_name in mass_fractions.keys():
            if pRT_name!='MMW':
                row = self.species_info[self.species_info[self.pRT_key] == pRT_name]
                mass = row["mass"].values[0]
                name = row.index[0]
                VMR_dict[name]=mass_fractions[pRT_name]*MMW/mass
        return VMR_dict
    
    def read_species_info(self,species,info_key):
        """
        Retrieve species metadata from the species_info table.

        Parameters
        ----------
        species : str
        info_key : str

        Returns
        -------
        value
            Requested property (e.g. mass, pRT name, C/O/H content).
        """
        if info_key == self.pRT_key:
            return self.species_info.loc[species,info_key]
        if info_key == 'pyfc_name':
            return self.species_info.loc[species,'Hill_notation']
        if info_key == 'mass':
            return self.species_info.loc[species,info_key]
        if info_key == 'COH':
            return list(self.species_info.loc[species,['C','O','H']])
        if info_key in ['C','O','H']:
            return self.species_info.loc[species,info_key]
        if info_key == 'c' or info_key == 'color':
            return self.species_info.loc[species,'color']
        if info_key == 'label':
            return self.species_info.loc[species,'mathtext_name']
    
    def get_isotope_mass_fractions(self, species_names, species_pRT, mass_fractions, params):
        """
        Mass-conserving isotope framework.

        Returns
        -------
        dict
            Updated mass fractions including isotopologues.
        """

        # Convert ratios → isotope fractions
        def ratio_to_fraction(log_ratio):
            R = 10**log_ratio
            return 1.0 / (1.0 + R)

        ratios = {}

        if any('13C-16O' in s for s in species_pRT):
            R = 10**params.get('log_12CO_13CO_ratio', 15)
            ratios['13CO'] = 1.0 / (1.0 + R)

        if any('12C-18O' in s for s in species_pRT):
            R = 10**params.get('log_C16O_C18O_ratio', 15)
            ratios['C18O'] = 1.0 / (1.0 + R)

        if any('12C-17O' in s for s in species_pRT):
            R = 10**params.get('log_C16O_C17O_ratio', 15)
            ratios['C17O'] = 1.0 / (1.0 + R)

        if any('H2-18O' in s for s in species_pRT):
            R = 10**params.get('log_H216O_H218O_ratio', 15)
            ratios['H2O18'] = 1.0 / (1.0 + R)

        # Identify base species per molecule
        base_map = {}

        for sp, sp_pRT in zip(species_names, species_pRT):
            if 'CO' in sp_pRT:
                base_map.setdefault('CO', sp_pRT)
            elif 'H2O' in sp_pRT or 'H2-16O' in sp_pRT:
                base_map.setdefault('H2O', sp_pRT)

        # Apply isotope splitting
        for species_i, species_pRT_i in zip(species_names, species_pRT):

            if species_i in self.leave_out:
                continue

            if 'CO' in species_pRT_i:

                main = base_map['CO']
                f13 = ratios.get('13CO', 0.0)
                f18 = ratios.get('C18O', 0.0)
                f17 = ratios.get('C17O', 0.0)

                # isotopologues
                if '13C' in species_pRT_i:
                    mass_fractions[species_pRT_i] = f13 * mass_fractions[main]

                elif '18O' in species_pRT_i:
                    mass_fractions[species_pRT_i] = f18 * mass_fractions[main]

                elif '17O' in species_pRT_i:
                    mass_fractions[species_pRT_i] = f17 * mass_fractions[main]

                # main isotopologue (12C16O)
                elif '12C' in species_pRT_i:

                    total_iso = f13 + f18 + f17
                    mass_fractions[species_pRT_i] = (1.0 - total_iso) * mass_fractions[main]

            elif 'H2O' in species_pRT_i or 'H2-16O' in species_pRT_i:

                main = base_map['H2O']
                f18 = ratios.get('H2O18', 0.0)
                if '18' in species_pRT_i:
                    mass_fractions[species_pRT_i] = f18 * mass_fractions[main]
                else:
                    mass_fractions[species_pRT_i] = (1.0 - f18) * mass_fractions[main]

        return mass_fractions
    
    def VMR_to_MF(self, VMRs):
        """
        Convert volume mixing ratios (VMRs) to mass fractions.

        Returns
        -------
        dict
            Mass fractions including 'MMW'.
        """
        MMW = 0.
        for species_i, VMR_i in VMRs.items():
            mass_i = self.read_species_info(species_i, 'mass')
            MMW += mass_i * VMR_i

        # Convert to mass-fractions using mass-ratio
        self.mass_fractions['MMW'] = MMW * np.ones(self.n_atm_layers)
        for species_i, VMR_i in VMRs.items():
            species_pRT_i = self.read_species_info(species_i, self.pRT_key)
            mass_i = self.read_species_info(species_i, 'mass')
            mf = VMR_i * mass_i / MMW
            self.mass_fractions[species_pRT_i] = mf
        return self.mass_fractions

    def compute_gaussian_VMR(self, species_i, params):
        logP = np.log10(self.pressure)
        logP0 = params[f"log_p_{species_i}"]
        sigma = params[f"std_{species_i}"]
        log_vmr_max = params[f"log_maxVMR_{species_i}"]
        vmr_max = 10**log_vmr_max
        vmr = vmr_max * np.exp(- (logP - logP0)**2 / (2 * sigma**2))
        return vmr

    def equ_chemistry(self, species_pRT, params):
        """
        Compute equilibrium chemistry using pre-tabulated interpolation grids.

        Supports optional scaling (flexible equilibrium) and quenching.

        Returns
        -------
        dict
            Mass fractions.
        """

        species_pRT.extend(s for s in ['H2','He'] if s not in species_pRT)
        self.species_hill.extend(s for s in ['H2','He'] if s not in self.species_hill)

        if 'H-' in species_pRT:  # required for calculation
            species_pRT.extend(s for s in ['e-','H'] if s not in species_pRT)
            self.species_hill.extend(s for s in ['e-','H'] if s not in self.species_hill)
        
        if self.add_species!=None:
            species_pRT.extend((self.species_info.loc[self.add_species,self.pRT_key]))
            self.species_hill.extend((self.species_info.loc[self.add_species,'Hill_notation']))

        def load_interp_tables():
            import h5py, pathlib
            def load_hdf5(file, key):
                with h5py.File(f'{path_tables}/{file}', 'r') as f:
                    return f[key][...]

            # Load the interpolation grid (ignore N/O)
            self.P_grid  = load_hdf5('grid.hdf5', 'P')
            self.T_grid  = load_hdf5('grid.hdf5', 'T')
            self.CO_grid = load_hdf5('grid.hdf5', 'C/O')
            self.FeH_grid = load_hdf5('grid.hdf5', 'Fe/H')
            points = (self.P_grid, self.T_grid, self.CO_grid, self.FeH_grid)

            self.interp_tables = {}
            for species_i, hill_i in zip([*species_pRT, 'MMW'], [*self.species_hill, 'MMW']):
                key = 'MMW' if species_i == 'MMW' else 'log_VMR'
                if species_i in ['e-', 'H']:
                    hill_i = 'e-' if species_i == 'e-' else 'H'
                equ_table = pathlib.Path(f'{path_tables}/{hill_i}.hdf5')
                if equ_table.exists():
                    arr = load_hdf5(f'{hill_i}.hdf5', key=key)  # Load equchem abundance tables
                self.interp_tables[species_i] = RegularGridInterpolator(
                    values=arr[:,:,:,0,:], points=points, method='linear'
                )

        def get_VMRs(ParamTable):
            self.VMRs = {}

            def apply_bounds(val, grid):
                val = np.array(val)
                val[val > grid.max()] = grid.max()
                val[val < grid.min()] = grid.min()
                return val

            # Update the parameters
            self.CO  = ParamTable.get('C/O')
            self.FeH = ParamTable.get('Fe/H')

            # Apply the bounds of the grid
            P = apply_bounds(self.pressure.copy(), grid=self.P_grid)
            T = self.temperature.copy()
            CO  = apply_bounds(np.array([self.CO]).copy(), grid=self.CO_grid)[0]
            FeH = apply_bounds(np.array([self.FeH]).copy(), grid=self.FeH_grid)[0]

            T_max = self.T_grid.max()

            # Interpolate abundances
            for pRT_name_i, interp_func_i in self.interp_tables.items():

                # Clip T for interpolation (avoid out-of-bounds)
                T_clip = np.clip(T, self.T_grid.min(), self.T_grid.max())
                arr_i = interp_func_i(xi=(P, T_clip, CO, FeH))

                # hold VMR constant above 6000K, limit of equchem tables
                # Find the last valid layer below T_max
                valid_mask = T <= T_max
                if np.any(valid_mask):
                    last_valid_idx = np.where(valid_mask)[0][-1]
                    arr_i[~valid_mask] = arr_i[last_valid_idx]

                if pRT_name_i != 'MMW':
                    species_i = self.species_info[
                        self.species_info[self.pRT_key] == pRT_name_i
                    ].index[0]

                    const_species = self.params.get('const_species', [])
                    gaussian_species = self.params.get('gaussian_species', [])

                    # 🔥 Gaussian override (highest priority)
                    if species_i in gaussian_species:
                        vmr = self.compute_gaussian_VMR(species_i, params)
                        self.VMRs[species_i] = vmr

                    # 🔥 Constant VMR override
                    elif species_i in const_species:
                        log_vmr = params.get(f'log_{species_i}')
                        if log_vmr is None:
                            raise ValueError(f'Missing parameter: log_{species_i}')
                        vmr_const = 10**log_vmr
                        self.VMRs[species_i] = np.full_like(P, vmr_const)

                    else:
                        # --- existing equilibrium logic ---
                        if self.chemistry == 'flexequ' and species_i not in ['13CO','C17O','C18O','H2(18)O']:
                            vmr = (10**arr_i) * (10**params[f'log_a_{species_i}'])
                            self.VMRs[species_i] = np.clip(vmr, a_min=None, a_max=0.1)
                        else:
                            self.VMRs[species_i] = 10**arr_i

                    # Apply leave_out AFTER override
                    if species_i in self.leave_out:
                        self.VMRs[species_i].fill(0)
                else:
                    self.MMW = arr_i.copy()  # Mean molecular weight
            return self.VMRs

        load_interp_tables()
        self.VMRs = get_VMRs(params)
        self.mass_fractions = self.VMR_to_MF(self.VMRs)

        if self.chemistry == 'quequchem':
            already_quenched = set()
            quench_params = [p for p in self.params if p.startswith("log_Pqu_")]
            for qparam in quench_params:
                species_list = qparam.split("log_Pqu_")[1].split("_")
                Pqu = 10**self.params[qparam]
                idx = find_nearest(self.pressure, Pqu)
                for sp in species_list:
                    if sp not in already_quenched:
                        sp_pRT = self.species_info.loc[sp,self.pRT_key]
                        quenched_fraction = self.mass_fractions[sp_pRT][idx]
                        self.mass_fractions[sp_pRT][:idx] = quenched_fraction
                        already_quenched.add(sp)

        return self.mass_fractions
    
    def free_chemistry(self,species_pRT,params):
        """
        Free chemistry with constant VMRs per species.

        Computes mass fractions, C/O, and metallicity.

        Returns
        -------
        mass_fractions, CO, FeH
        """
        VMR_He = 0.15
        VMR_wo_H2 = 0 + VMR_He  # Total VMR without H2, starting with He
        C, O, H = 0, 0, 0
        if 'log_e-' in self.params:
            species_pRT.append('e-') if 'e-' not in species_pRT else None

        for species_i in self.species_info.index:
            species_pRT_i = self.read_species_info(species_i,self.pRT_key)
            mass_i = self.read_species_info(species_i, 'mass')
            COH_i  = self.read_species_info(species_i, 'COH')

            if species_i in ['H2', 'He']:
                continue
            if species_pRT_i in species_pRT:
                gaussian_species = self.params.get('gaussian_species', [])
                if species_i in gaussian_species:
                    VMR_i = self.compute_gaussian_VMR(species_i, params)
                else:
                    VMR_i = 10**(params[f'log_{species_i}']) * np.ones(self.n_atm_layers)

                # Convert VMR to mass fraction using molecular mass number
                self.mass_fractions[species_pRT_i] = mass_i * VMR_i
                VMR_wo_H2 += VMR_i

                # Record C, O, and H bearing species for C/O and metallicity
                C += COH_i[0] * VMR_i
                O += COH_i[1] * VMR_i
                H += COH_i[2] * VMR_i

        # Add the H2 and He abundances
        self.mass_fractions['He'] = self.read_species_info('He', 'mass')*VMR_He
        self.mass_fractions['H2'] = self.read_species_info('H2', 'mass')*(1-VMR_wo_H2)
        
        H += self.read_species_info('H2','H')*(1-VMR_wo_H2) # Add to the H-bearing species

        if VMR_wo_H2.any() > 1:
            print('\nVMR_wo_H2 > 1. Other species are too abundant!',VMR_wo_H2)

        MMW = 0 # Compute the mean molecular weight from all species
        for mass_i in self.mass_fractions.values():
            MMW += mass_i
        MMW *= np.ones(self.n_atm_layers)
        
        for species_pRT_i in self.mass_fractions.keys():
            self.mass_fractions[species_pRT_i] /= MMW # Turn the molecular masses into mass fractions
        self.mass_fractions['MMW'] = MMW # pRT requires MMW in mass fractions dictionary
        CO = C/O if np.sum(O)!=0 else np.inf
        log_CH_solar = 8.46 - 12 # Asplund et al. (2021)
        FeH = np.log10(C/H)-log_CH_solar if (C/H).any()!=0.0 else -np.inf*np.ones((len(C)))
        CO = np.nanmean(CO)
        FeH = np.nanmean(FeH)

        return self.mass_fractions, CO, FeH

    def var_chemistry(self,line_species,params): # vary with pressure
        """
        Free chemistry with pressure-dependent abundances (knots + interpolation).

        Returns
        -------
        mass_fractions, CO, FeH
        """

        CO_list=[]
        FeH_list=[]
        VMR_He = 0.15
        VMRs_list=[]

        # check how many pressure knots
        n_knots = sum(1 for key in params if re.fullmatch(r'log_H2O_\d+', key))

        for knot in range(n_knots): # points where to retrieve abundances

            VMR_wo_H2 = 0 + VMR_He  # Total VMR without H2, starting with He
            VMRs={}
            C, O, H = 0, 0, 0

            for species_i in self.species_info.index:
                line_species_i = self.read_species_info(species_i,self.pRT_key)
                mass_i = self.read_species_info(species_i, 'mass')
                COH_i  = self.read_species_info(species_i, 'COH')

                if species_i in ['H2', 'He']:
                    continue
                if line_species_i in line_species:
                    if f'log_{species_i}_{knot}' in params:
                        VMR_i = 10**(params[f'log_{species_i}_{knot}'])
                    else:
                        VMR_i = 10**(params[f'log_{species_i}']) # vertically constant for some species
                    # Convert VMR to mass fraction using molecular mass number
                    VMRs[line_species_i] = VMR_i
                    VMR_wo_H2 += VMR_i

                    # Record C, O, and H bearing species for C/O and metallicity
                    C += COH_i[0] * VMR_i
                    O += COH_i[1] * VMR_i
                    H += COH_i[2] * VMR_i

            # Add the H2 and He abundances
            VMRs['He'] = VMR_He
            H += self.read_species_info('H2','H')*(1-VMR_wo_H2) # Add to the H-bearing species
            self.VMR_wo_H2=VMR_wo_H2

            CO = C/O
            log_CH_solar = 8.46 - 12 # Asplund et al. (2021)
            FeH = np.log10(C/H)-log_CH_solar
            CO_list.append(CO)
            FeH_list.append(FeH)
            VMRs_list.append(VMRs)

        VMRs_interp = {}
        line_species.append('He')

        for species_i in self.species_info.index:
            line_species_i = self.read_species_info(species_i,self.pRT_key)
            mass_i = self.read_species_info(species_i, 'mass')
            if line_species_i in line_species and line_species_i!='H2':
                vmrs_var = []
                for kn in range(n_knots):
                    vmrs_var.append(VMRs_list[kn][line_species_i])
                log_P_knots = generate_skewed_p_nodes(self.pressure,
                                                      n_nodes=n_knots, 
                                                      skew=self.p_node_skew)
                #else:
                    #log_P_knots= np.linspace(np.log10(np.min(self.pressure)),np.log10(np.max(self.pressure)),num=n_knots)

                # use linear interpolation to avoid going into negative values cubic spline did that)
                log_vmrs=np.interp(np.log10(self.pressure), log_P_knots, np.log10(vmrs_var)) # interpolate for all layers

                VMRs_interp[line_species_i] = 10**log_vmrs #np.interp(np.log10(self.pressure), log_P_knots, mass_fracs) # interpolate for all layers
                self.mass_fractions[line_species_i]=mass_i*VMRs_interp[line_species_i]

        vmr_layers = np.empty(self.n_atm_layers) # vmr of layers
        for l in range(self.n_atm_layers):
            vmr=0
            for key in VMRs_interp.keys():
                vmr+=VMRs_interp[key][l]
            vmr_layers[l]=vmr

        self.vmr_layers=vmr_layers
        vmr_H2=np.empty(self.n_atm_layers)
        for l in range(self.n_atm_layers):
            vmr_H2[l]=1-vmr_layers[l]
            if vmr_H2[l]<0:
                print('Invalid VMR',vmr_H2[l])
                self.VMR_wo_H2=1.1
                exit_mf={}
                for key in VMRs_interp.keys():
                    exit_mf[key]=np.ones(self.n_atm_layers)*1e-12
                return exit_mf,1,1
        VMRs_interp['H2']=vmr_H2
        self.mass_fractions['H2']=self.read_species_info('H2','mass')*VMRs_interp['H2']

        mmw_layers=np.empty(self.n_atm_layers)
        for l in range(self.n_atm_layers):
            MMW = 0 # Compute the mean molecular weight from all species for each layer
            for line_species_i in self.mass_fractions.keys():
                MMW += self.mass_fractions[line_species_i][l]
            mmw_layers[l] = MMW
        self.mass_fractions['MMW'] = mmw_layers # pRT requires MMW in mass fractions dictionary

        for line_species_i in self.mass_fractions.keys():
            if line_species_i=='MMW':
                continue
            self.mass_fractions[line_species_i] /= self.mass_fractions['MMW'] # Turn the molecular masses into mass fractions
                       
        CO = np.nanmean(CO_list)
        FeH = np.nanmean(FeH_list)
        self.VMRs=VMRs_interp
        
        return self.mass_fractions, CO, FeH

    def free_chemistry_use_partial_pressure(self, species_pRT, params):
        """
        Free chemistry using retrieved partial pressures for all species.

        Reconstructs total pressure and derives mass fractions.

        Returns
        -------
        mass_fractions, CO, FeH
        """

        C, O, H = 0, 0, 0
        species_pRT.append('He')
        species_pRT.append('H2')

        # Retrieve partial pressures for all species in each layer
        P_partials = {}

        for species_i in self.species_info.index:
            species_pRT_i = self.read_species_info(species_i, self.pRT_key)
            mass_i = self.read_species_info(species_i, 'mass')
            COH_i = self.read_species_info(species_i, 'COH')

            if species_pRT_i in species_pRT:
                # Retrieve one parameter per species (constant with altitude)
                logP_i = params[f'log_p_{species_i}']
                P_partials[species_i] = 10**logP_i #* np.ones(self.n_atm_layers)  # [bar]

        # Total pressure profile reconstructed as sum of retrieved partial pressures
        P_tot = 0.0
        for P_i in P_partials.values():
            P_tot += P_i
        if P_tot > max(self.pressure):
            self.unphysical_params = True

        self.P_tot = P_tot

        # Convert partial pressures to VMR and compute mass fractions
        VMR_tot = 0.0
        for species_i, P_i in P_partials.items():
            VMR_i = P_i / P_tot
            VMR_tot += VMR_i
            species_pRT_i = self.read_species_info(species_i, self.pRT_key)
            mass_i = self.read_species_info(species_i, 'mass')
            COH_i = self.read_species_info(species_i, 'COH')

            self.mass_fractions[species_pRT_i] = mass_i * VMR_i
            C += COH_i[0] * VMR_i
            O += COH_i[1] * VMR_i
            H += COH_i[2] * VMR_i

        # Compute MMW
        MMW = sum(self.mass_fractions.values()) * np.ones(self.n_atm_layers)

        # Normalize mass fractions
        for species_pRT_i in self.mass_fractions.keys():
            self.mass_fractions[species_pRT_i] /= MMW
        self.mass_fractions['MMW'] = MMW

        # Element ratios
        CO = np.nanmean(C / O) if np.sum(O) != 0 else np.inf
        log_CH_solar = 8.46 - 12  # Asplund et al. (2021)
        FeH = np.nanmean(np.log10(C / H) - log_CH_solar) if np.any(H) else -np.inf

        return self.mass_fractions, CO, FeH
    
    def setup_physical_clouds(self, cloud_species):
        """Set up physical condensate clouds."""

        mode = self.params['cloud_mode']
        for cloud_species_i in cloud_species:
            cloud_species_pRT_i = self.species_info.loc[f'{cloud_species_i}_{mode}',self.pRT_key]

            # Pressure mask above cloud deck
            p_base = 10**self.params[f'log_P_base_{cloud_species_i}']
            mask_above_deck = self.pressure <= p_base

            # Initialize mass fractions
            self.mass_fractions[cloud_species_pRT_i] = np.zeros_like(self.pressure)
            # Set cloud abundance with sedimentation profile
            x_cloud_base = 10**self.params[f'base_MF_log_{cloud_species_i}']
            f_sed = self.params[f'fsed_{cloud_species_i}']
            
            self.mass_fractions[cloud_species_pRT_i][mask_above_deck] = (
                x_cloud_base * (self.pressure[mask_above_deck] / p_base) ** f_sed)

            self.cloud_f_sed.update({cloud_species_pRT_i: f_sed})

    def gray_cloud_opacity(self, wave_micron, pressure):
        """
        Calculate gray cloud opacity as function of wavelength and pressure.
        
        This function is called by petitRADTRANS during radiative transfer calculation.

        Args:
            wave_micron: Wavelength array in microns
            pressure: Pressure array in bar

        Returns:
            Gray cloud opacity array with shape (n_wavelength, n_pressure)
        """
        # Initialize opacity array
        opa_gray_cloud = np.zeros((len(wave_micron), len(pressure)))

        # Get cloud parameters
        p_base_gray = 10**self.params['log_P_base_gray']
        opa_base_gray = 10**self.params['log_opa_base_gray']
        f_sed_key = 'fsed_gray' if 'fsed_gray' in self.params else 'fsed'
        f_sed_gray = self.params[f_sed_key] # 2025-10-16: new parameter for gray cloud opacity, different from fsed for PT profile
        
        # No opacity below cloud base (pressure > P_base)
        # Opacity decreases with power-law above the base (pressure <= P_base)
        mask_above_deck = pressure <= p_base_gray
        
        if np.any(mask_above_deck):
            opa_gray_cloud[:, mask_above_deck] = (
                opa_base_gray * (pressure[mask_above_deck] / p_base_gray) ** f_sed_gray)

        # Apply wavelength dependence if specified
        cloud_slope = self.params.get('cloud_slope')
        if cloud_slope is not None:
            wavelength_factor = (wave_micron[:, None] / 1.0) ** cloud_slope
            opa_gray_cloud *= wavelength_factor

        return opa_gray_cloud
    
    def make_spectrum(self):
        """
        Generate model spectrum using petitRADTRANS.

        Includes radiative transfer, RV shifting, rotational and instrumental
        broadening, and optional interpolation to data grid.

        Returns
        -------
        ndarray or (list, list)
            Model spectrum (and wavelengths if not interpolated).
        """
        
        spectrum_parts=[]
        waves_parts=[]
        summed_contributions =[]
        data_shape = self.data_wave.shape
        self.model_continuum = np.ones(self.data_wave.shape)
        interp_onto_wave = self.data_wave

        if self.instrument=='CRIRES':
            orders_shape = (self.n_orders,self.n_dets*self.n_pixels)
            interp_onto_wave = self.data_wave.reshape(orders_shape)

        if isinstance(self.radtrans_objects, list)==False:
            self.radtrans_objects = [self.radtrans_objects]

        for part, radtrans in enumerate(self.radtrans_objects):

            if self.cloud_mode == 'gray': # Gray cloud opacity
                self.give_absorption_opacity= self.gray_cloud_opacity 

            if self.params['emission_or_transmission']=='emission':
                wl_cm, flux, additional_outputs = radtrans.calculate_flux(temperatures=self.temperature,
                                                                    mass_fractions=self.mass_fractions,
                                                                    mean_molar_masses = self.MMW,
                                                                    reference_gravity = self.gravity,
                                                                    return_contribution=self.contribution,
                                                                    additional_absorption_opacities_function = self.give_absorption_opacity,
                                                                    eddy_diffusion_coefficients=self.Kzz, # array of eddy diffusion coefficients for each pressure layer (constant)
                                                                    cloud_f_sed=self.cloud_f_sed, # dictionary of f_sed for each cloud species
                                                                    cloud_particle_radius_distribution_std = self.cloud_particle_radius_distribution_std,
                                                                    cloud_fraction=self.params.get('cloud_fraction', 1.0), # for cloud coverage
                                                                    frequencies_to_wavelengths=True)
                flux *= u.erg/(u.cm**2*u.s*u.cm)
            
            elif self.params['emission_or_transmission']=='transmission':
                R_p = ensure_quantity(self.params['R_p'], u.R_jup).to(u.cm).value
                wl_cm, transit_radii_cm, additional_outputs = radtrans.calculate_transit_radii(
                                                                temperatures=self.temperature,
                                                                mass_fractions=self.mass_fractions,
                                                                mean_molar_masses=self.MMW,
                                                                reference_gravity=self.gravity,
                                                                planet_radius= R_p,
                                                                reference_pressure=self.params['ref_pressure'],
                                                                additional_absorption_opacities_function = self.give_absorption_opacity,
                                                                eddy_diffusion_coefficients=self.Kzz, # array of eddy diffusion coefficients for each pressure layer (constant)
                                                                cloud_f_sed=self.cloud_f_sed, # dictionary of f_sed for each cloud species
                                                                cloud_particle_radius_distribution_std = self.cloud_particle_size_std,
                                                                cloud_fraction=self.params.get('cloud_fraction', 1.0),
                                                                return_contribution=self.contribution)
                R_s = ensure_quantity(self.params['R_s'], u.R_sun).to(u.cm).value
                flux = (transit_radii_cm / R_s)**2 * 100 # in  %

            wl_cm *= u.cm
            wl = wl_cm.to(self.params['wavelength_unit']).value
            if self.params['flux_unit'] is not None:
                if isinstance(self.params['flux_unit'], u.Quantity):
                    flux = flux.to(self.params['flux_unit']).value
                elif self.params['flux_unit']=='photons':
                    flux = self.pRT_to_photon_flux(radtrans)

            if self.contribution==True: # emission contribution

                if 'emission_contribution' in additional_outputs.keys():
                    atm_contr = additional_outputs.pop('emission_contribution')

                elif 'transmission_contribution' in additional_outputs.keys():
                    atm_contr = additional_outputs.pop('transmission_contribution')

                self.summed_contribution = np.nansum(atm_contr,axis=1) # sum over all wavelengths
                summed_contributions.append(self.summed_contribution)

            # RV + barycentric velocity shifting
            if 'rv' in self.params:
                wl_shifted = wl * (1.0 + (self.params['rv'] - self.vbary) / const.c.to('km/s').value)
                flux = np.interp(wl, wl_shifted, flux)

            # Rotational broadening (requires evenly spaced wavelength grid)
            if 'epsilon_limb' in self.params and 'vsini' in self.params:
                waves_even = np.linspace(wl.min(), wl.max(), wl.size)
                flux = np.interp(waves_even, wl, flux)
                flux = fastRotBroad(waves_even,flux,self.params['epsilon_limb'],self.params['vsini'])
                wl = waves_even  # update wavelength grid

            flux = self.instrumental_broadening(wl, flux, self.spectral_resolution)

            # Interpolate/rebin onto the data's wavelength grid
            # don't when making spectrum for cross-corr, or wavelength padding will be cut off
            if self.interpolate==True:
                ref_wave = interp_onto_wave[part]
                flux = np.interp(ref_wave, wl, flux)

            spectrum_parts.append(flux)
            waves_parts.append(wl)

            if self.contribution==True:
                self.summed_contribution = np.nanmean(summed_contributions,axis=0)  

        if self.interpolate==False:
            get_median=np.array([])
            for order in range(len(self.radtrans_objects)): # append value by value because not all the same size
                get_median=np.append(get_median,spectrum_parts[order]) 
            spectrum_parts=np.array(spectrum_parts,dtype=object)
            if self.params['normalize_spectrum']:
                spectrum_parts/=np.nanmedian(get_median) # orders not same size, np.median didn't work otherwise
            if self.params['remove_continuum']:
                for order in range(len(self.radtrans_objects)):
                    fac = len(spectrum_parts[order])/self.n_pixels
                    spectrum_parts[order] = fft_remove_continuum(spectrum_parts[order],lower_cutoff=13*fac) # 3 dets+overhead
            return spectrum_parts, waves_parts
        else:
            spectrum_parts=np.array(spectrum_parts)
            spectrum_parts = spectrum_parts.reshape(data_shape)
                
            if self.params['remove_continuum']:
                for i,part in enumerate(spectrum_parts):
                    if ('log_k_rk' in self.params) or ('T_disk' in self.params):
                        spectrum_parts[i] /=np.nanpercentile(spectrum_parts[i],97)
                        rk = self.rk_func(self.data_wave[i])
                        sp = spectrum_parts[i]
                        spectrum_parts[i] = (sp+rk*np.nanmedian(sp))/(1+rk)
                    spectrum_parts[i] /= np.nanmedian(spectrum_parts[i])
                    spectrum_parts[i][~self.mask_isfinite[i]] = np.nan
                    spec, continuum = fft_remove_continuum(spectrum_parts[i],return_continuum=True)
                    spectrum_parts[i] =spec 
                    self.model_continuum[i] = continuum
            elif self.params['normalize_spectrum']:
                spectrum_parts/=np.nanmedian(spectrum_parts) 
            
            return spectrum_parts
            
    def make_pt(self,**kwargs): 

        """
        Generate the atmospheric pressure-temperature (P-T) profile,
        according to the selected parameterization stored in ``self.PT_type``.

        Supported parameterizations
        ---------------------------
        PTknot
            Temperature nodes (``T0``, ``T1``, ...) are retrieved at fixed pressure
            knots. A cubic spline interpolation in log-pressure space is used to
            compute the temperature at each layer of the atmospheric grid.

        PTgrad
            The logarithmic temperature gradient ``d ln T / d ln P`` is retrieved
            at several pressure knots. These gradients are interpolated across
            the atmosphere and integrated from the base temperature ``T0`` to
            produce the full temperature profile.

        PTguillot
            Analytic radiative-equilibrium temperature profile based on the
            Guillot (2010) formalism. The profile depends on the intrinsic
            temperature, equilibrium temperature, infrared opacity, and the
            ratio of optical to infrared opacities.

        Returns
        -------
        ndarray
            Temperature profile for the pressure grid ``self.pressure``.
        """

        if self.PT_type=='PTknot': # retrieve temperature knots
            t_keys = [key for key in self.params.keys() if re.fullmatch(r"T\d+", key)] 
            t_keys = sorted(t_keys, key=lambda x: int(x[1:]))[::-1] # start at top of atmosphere, T0 last
            self.T_knots = []
            for key in t_keys:
                self.T_knots.append(self.params[key])
            self.T_knots = np.array(self.T_knots)
            self.log_P_knots = generate_skewed_p_nodes(self.pressure,
                                                       n_nodes=len(t_keys), 
                                                       skew=self.p_node_skew)
            sort = np.argsort(self.log_P_knots)
            x = self.log_P_knots[sort]
            y = self.T_knots[sort]

            logP = np.log10(self.pressure)

            # Adaptive interpolation
            if len(x) >= 3:
                self.temperature = CubicSpline(x, y)(logP)
            elif len(x) == 2:
                # linear interpolation
                self.temperature = interp1d(x, y, kind='linear', fill_value='extrapolate')(logP)
            elif len(x) == 1:
                # constant atmosphere
                self.temperature = np.full_like(self.pressure, y[0])
        
        if self.PT_type=='PTgrad':
            # check how many pressure knots
            n_grad = sum(1 for key in self.params if re.fullmatch(r'dlnT_dlnP_\d+', key))
            self.log_P_knots = generate_skewed_p_nodes(self.pressure,
                                                       n_nodes=n_grad, 
                                                       skew=self.p_node_skew)

            if 'dlnT_dlnP_knots' not in kwargs:
                self.dlnT_dlnP_knots=[]
                for i in range(n_grad):
                    self.dlnT_dlnP_knots.append(self.params[f'dlnT_dlnP_{i}'])
            elif 'dlnT_dlnP_knots' in kwargs: # needed for calc error on PT, upper+lower bounds passed
                self.dlnT_dlnP_knots=kwargs.get('dlnT_dlnP_knots')
            logP = np.log10(self.pressure)

            # Adaptive interpolation
            if n_grad >= 3:
                interp_kind = 'quadratic'
            elif n_grad == 2:
                interp_kind = 'linear'
            else:
                interp_kind = 'nearest'

            interp_func = interp1d(
                self.log_P_knots,
                self.dlnT_dlnP_knots,
                kind=interp_kind,
                fill_value='extrapolate')

            dlnT_dlnP = interp_func(logP)

            T_base = self.params['T0']
            ln_P = np.log(self.pressure)[::-1]
            temperature = [T_base]
            lower_T_lim = 10
            upper_T_lim = max(self.params['T_prior'])

            for i in range(1, len(ln_P)):
                dlnP = ln_P[i] - ln_P[i-1]
                lnT = np.log(temperature[-1]) + dlnP * dlnT_dlnP[::-1][i]
                next_T = np.exp(lnT)
                next_T = np.clip(next_T, lower_T_lim, upper_T_lim)
                temperature.append(next_T)

            self.temperature = np.array(temperature[::-1])

        elif self.PT_type=='PTguillot':
            T_int = self.params['T_int']
            T_equ = self.params['T_equ']
            kappa_IR = 10**self.params['log_k_IR']
            gamma = 10**self.params['log_gamma']
            self.temperature = temperature_profile_function_guillot_global(pressures=self.pressure,
                                                                    infrared_mean_opacity=kappa_IR,
                                                                    gamma=gamma,
                                                                    gravities=self.gravity,
                                                                    intrinsic_temperature=T_int,
                                                                    equilibrium_temperature=T_equ)

        return self.temperature

    def instrumental_broadening(self, wave, flux, resolution=100000, fwhm=None):

        IB = InstrumentalBroadening(wave, flux)
        if isinstance(resolution, np.ndarray):
            # Variable resolution profile
            flux_LSF = IB(fwhm=const.c.to(u.km/u.s).value/resolution, kernel='gaussian_variable')
            return flux_LSF
        else:
            # Constant resolution
            if fwhm==None: 
                flux_LSF = IB(res=resolution, kernel='gaussian')
            else: # fwhm in km/s
                flux_LSF = IB(fwhm=fwhm, kernel='gaussian')
            return flux_LSF
    
    def pRT_to_photon_flux(self,radtrans):

        nu = radtrans.freq * u.Hz # Frequency grid from pRT
        wl_um = (const.c / nu).to(u.um)
        wl_m = wl_um.to(u.m)
        # pRT output is per Hz, convert to per micron:
        flux_nu = radtrans.flux * u.erg / (u.cm**2 * u.s * u.Hz)

        # dν/dλ = -c / λ²  ⇒ |dν/dλ| = c / λ²
        dnu_dlambda = (const.c / wl_m**2).to(u.Hz / u.um)
        flux_lambda = (flux_nu * dnu_dlambda).to(u.erg / (u.cm**2 * u.s * u.um))

        E_photon = (const.h * const.c / wl_m).to(u.J)
        flux_J_m2_s_um = flux_lambda.to(u.J / u.m**2 / u.s / u.um) # Convert flux to J/m²/s/μm

        # Photon flux [photons / s / m² / μm]
        flux_photon = (flux_J_m2_s_um / E_photon).to(1 / (u.s * u.m**2 * u.um))

        radius = ensure_quantity(self.params['R_p'], u.m).value
        distance = ensure_quantity(self.params['distance'], u.m).value

        scaling = (radius / distance) ** 2
        flux_observed = flux_photon*scaling*4*np.pi
        t_obs = ensure_quantity(self.params['obstime'], u.s).value
        flux_observed*= t_obs
        return flux_observed