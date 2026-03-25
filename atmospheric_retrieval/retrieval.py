import getpass
import os
from pRT_model import pRT_spectrum
import figures as figs
from covariance import *
from log_likelihood import *
from target import Target
from parameters import Parameters
from utils import *
import re
import gc

import numpy as np
import pymultinest
import pathlib
import pickle
from petitRADTRANS.radtrans import Radtrans

import pandas as pd
import astropy.constants as const
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # pRT warning
from scipy.linalg import LinAlgWarning
warnings.filterwarnings(action='ignore', category=LinAlgWarning, module='sklearn') # occasional
warnings.simplefilter("error", RuntimeWarning)  # Convert warnings to exceptions
#warnings.filterwarnings("ignore", category=np.linalg.LinAlgError) 

class Retrieval:

    def __init__(self,target,parameters,Nlive,evtol,chemistry='freechem',
                 PT_type='PTgrad',redo=False,folder_suffix=''):
        
        self.Nlive = Nlive
        self.evtol = evtol
        self.folder_suffix = folder_suffix
        for attr in ['species_info','color','instrument','spectral_resolution','name']:
            setattr(self, attr, getattr(target, attr))
            
        self.target = target
        self.data_wave,self.data_flux,self.data_err= target.wl,target.fl,target.err
        self.mask_isfinite=target.mask_isfinite #get_mask_isfinite() # mask nans, shape (orders,detectors)    
        self.separation,self.err_eff=target.prepare_for_covariance()
        self.PT_type=PT_type
        self.parameters=parameters
        self.pRT_key = f'{self.parameters.params["opa_mode"]}_name'
        self.cloud_mode = self.parameters.params["cloud_mode"]
        self.cloud_species = self.parameters.params.get('cloud_species', None)
        if self.cloud_species:
            self.cloud_species_pRT=[]
            for cloud_species_i in self.cloud_species:
                pRT_name = self.species_info.loc[f'{cloud_species_i}_{self.cloud_mode}',self.pRT_key]
                self.cloud_species_pRT.append(pRT_name)


        self.n_atm_layers = self.parameters.params['n_atm_layers']
        log_p_upper = self.parameters.params['log_P_upper']
        log_p_lower = self.parameters.params['log_P_lower']
        self.pressure = np.logspace(log_p_lower,log_p_upper,self.n_atm_layers)
        self.use_partial_pressure = self.parameters.params['use_partial_pressure'] # retrieve partial pressure OR VMRs

        if self.parameters.params['opa_mode']=='lbl':
            if 'lbl_opacity_sampling' in self.parameters.params: # lbl opacities
                self.lbl_opacity_sampling = self.parameters.params['lbl_opacity_sampling']
        else:
            self.lbl_opacity_sampling=3
        
        if self.instrument=='CRIRES':
            self.n_orders, self.n_dets = self.target.n_orders, self.target.n_dets

        self.chemistry=chemistry # freechem/equchem/quequchem/flexequ
        self.species_names = self.parameters.params['species_names'] 
        self.species_pRT, self.species_hill =self.get_pRT_hill(self.species_names)

        self.vary_species = []
        if self.chemistry=='varchem':
            self.n_knots = sum(1 for key in self.parameters.params if re.fullmatch(r'log_H2O_\d+', key))
            self.fixed_species = []
            for species_i in self.species_names:
                if f'log_{species_i}_0' in self.parameters.params:
                    self.vary_species.append(species_i)
                else:
                    self.fixed_species.append(species_i)

        self.n_parts, self.n_pixels = self.data_flux.shape
        self.n_params = len(parameters.free_params)
        self.output_name=f'{chemistry}_{PT_type}_N{Nlive}_ev{evtol}{folder_suffix}' # output folder name
        self.cwd = os.getcwd()
        self.output_dir = pathlib.Path(f'{self.cwd}/{self.target.name}/{self.output_name}')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.parameters.params['normalize_spectrum']:
            self.ylabel = 'Normalized Flux'
        elif self.parameters.params['emission_or_transmission']=='emission':
            self.ylabel = 'Flux'
            funit = self.parameters.params['flux_unit']
            if isinstance(funit, u.Quantity):
                self.ylabel += str(funit.unit)
        elif self.parameters.params['emission_or_transmission']=='transmission':
            self.ylabel = 'Transit depth [%]'
                
        self.Cov = np.empty((self.n_parts), dtype=object) # covariance matrix
        use_GP = self.parameters.params['use_GP']
        for i in range(self.n_parts):
            mask_i = self.mask_isfinite[i] # only finite pixels
            if not mask_i.any(): # skip empty
                continue
            if use_GP==True: # use Gaussian processes covariance matrix
                maxval=10**(self.parameters.param_priors['log_l'][1])*3 # 3*max value of prior of l
                self.Cov[i] = CovGauss(err=self.data_err[i,mask_i],separation=self.separation[i], 
                                        err_eff=self.err_eff[i],max_separation=maxval)
            if use_GP==False: # use simple diagonal covariance matrix
                self.Cov[i] = Covariance(err=self.data_err[i,mask_i])
        self.LogLike = LogLikelihood(retr_obj=self,scale_flux=self.parameters.params['scale_flux'],scale_err=True)

        # redo radtrans objects when introdocuing new species
        self.radtrans_objects=self.get_radtrans_objects(redo=redo)
        self.callback_label='live_' # label for plots
        self.prefix='pmn_'

        # will be updated, but needed as None until then
        self.bestfit_params=None 
        self.posterior = None
        self.params_dict=None

    def get_pRT_hill(self,species_names): # get pRT species name and hill notations
        species_pRT=[] # pRT names
        species_hill=[] # hill notation
        for species_i in species_names:
            species_pRT.append(self.species_info.loc[species_i,self.pRT_key])
            species_hill.append(self.species_info.loc[species_i,'Hill_notation'])
        return species_pRT, species_hill

    def get_radtrans_objects(self,redo=False,for_species=None):

        lines_species=self.species_pRT.copy() if for_species==None else for_species
        continuum_opacities = ['H2-H2', 'H2-He']
        if 'H-' in lines_species:
            lines_species.remove('H-') # not a line species
            continuum_opacities.append('H-')
        if for_species!=None:
            lines_species = [self.species_info.loc[lines_species,self.pRT_key]]
        if for_species==None: # none specified
            file=pathlib.Path(f'{self.target.project_path}/radtrans_objects.pickle')
            not_exists=False
            if file.exists() and redo==False:
                radtrans_objects= load_pickle(file)
                return radtrans_objects
            else:
                not_exists=True
                
        if for_species!=None or not_exists:
            radtrans_objects=[]
            CIA = ['H2--H2-NatAbund__BoRi.R831_0.6-250mu', 'H2--He-NatAbund__BoRi.DeltaWavenumber2_0.5-500mu']
            if 'H2O' in self.species_pRT:
                CIA.append('H2-O--H2-O-NatAbund.DeltaWavenumber10_0.5-77mu')
            if 'CO2' in self.species_pRT:
                CIA.append('C-O2--C-O2-NatAbund.DeltaWavelength1e-6_3-100mu')

            if self.instrument=='CRIRES':
                self.atm_segments = self.n_orders
            else:
                self.atm_segments = self.n_parts

            for seg in range(self.atm_segments):

                # expand wavelength range based on max RV in prior
                if self.instrument == 'CRIRES':
                    idx = slice(seg * self.n_dets, (seg + 1) * self.n_dets)
                    wave_segment = self.data_wave[idx]
                else:
                    wave_segment = self.data_wave[seg]
                wlmin = np.min(wave_segment)
                wlmax = np.max(wave_segment)
                if 'rv' in self.parameters.free_params:
                    rv_prior = self.parameters.param_priors['rv']
                    frac = max(abs(v) for v in rv_prior) / const.c.to('km/s').value
                elif 'rv' in self.parameters.constant_params:
                    rv = self.parameters.constant_params['rv']
                    frac = rv / const.c.to('km/s').value
                else:
                    frac = 0
                wlmin = wlmin * (1 - frac)
                wlmax = wlmax * (1 + frac)
                wl_unit = self.parameters.params['wavelength_unit'] # astopy unit
                wlen_range_um = (np.array([wlmin,wlmax])*wl_unit).to(u.um).value

                radtrans = Radtrans(
                            pressures=self.pressure,
                            line_species=lines_species,
                            rayleigh_species= ['H2', 'He'],
                            gas_continuum_contributors=CIA,
                            wavelength_boundaries=wlen_range_um, # microns
                            cloud_species=self.cloud_species_pRT,
                            line_opacity_mode=self.parameters.params['opa_mode'],
                            line_by_line_opacity_sampling=self.lbl_opacity_sampling)# take every nth point
                radtrans_objects.append(radtrans)
                    
            if for_species==None:
                save_pickle(radtrans_objects,file)
            return radtrans_objects

    def PMN_lnL(self,cube=None,ndim=None,nparams=None):
        self.model_object=pRT_spectrum(self)      
        P_tot = self.model_object.P_tot
        P_surf = self.model_object.P_surf

        if self.use_partial_pressure and P_tot>P_surf:
            return -np.inf

        self.model_flux=self.model_object.make_spectrum()
        summed_contribution = getattr(self.model_object, "summed_contribution", [])

        for i in range(self.n_parts): # update covariance matrix
            if not self.mask_isfinite[i].any(): # skip empty
                continue
            self.Cov[i](self.parameters.params)

        ln_L = self.LogLike(self.model_flux, self.Cov, params=self.parameters.params,
                            P_tot_surf=(P_tot,P_surf), metall=self.model_object.FeH)
        
        return ln_L

    def PMN_run(self,N_live_points=400,evidence_tolerance=0.5,resume=True):
        cef_mode = self.parameters.params['const_efficiency_mode']
        smp_eff = self.parameters.params['sampling_efficiency']
        pymultinest.run(LogLikelihood=self.PMN_lnL,Prior=self.parameters,n_dims=self.parameters.n_params, 
                        outputfiles_basename=f'{self.output_dir}/{self.prefix}', 
                        verbose=True,const_efficiency_mode=cef_mode, sampling_efficiency = smp_eff,
                        n_live_points=N_live_points,resume=resume,
                        evidence_tolerance=evidence_tolerance, # high number -> stops earlier
                        dump_callback=self.PMN_callback,n_iter_before_update=5)

    def PMN_callback(self,n_samples,n_live,n_params,live_points,posterior, 
                    stats,max_ln_L,ln_Z,ln_Z_err,nullcontext):
        self.bestfit_params = posterior[np.argmax(posterior[:,-2]),:-2] # parameters of best-fitting model
        posterior = posterior[:,:-2] # remove last 2 columns
        posterior_dict={}
        for key in self.parameters.free_params.keys():
            idx=list(self.parameters.free_params).index(key)
            posterior_dict[key]=(posterior[:,idx],self.parameters.free_params[key][1])
        self.posterior=posterior_dict
        self.params_dict,self.model_flux=self.get_params_and_spectrum()
        self.model_flux[~self.mask_isfinite] = np.nan

        figs.summary_plot(self,show_params='all')
        if self.use_partial_pressure:
            plot_species = self.species_names.copy()
            plot_species.extend(s for s in ['H2','He'] if s not in plot_species)
            figs.VMR_plot(self,VMR_species=plot_species,
                          xmin=1e-8,xmax=1e0,plotlegend=True)
        elif self.chemistry in ['equchem','quequchem','flexequ','varchem']:
            figs.VMR_plot(self,VMR_species='all')
        if self.chemistry=='flexequ':
            figs.plot_scaled_abunds(self)
        if ('log_k_rk' in self.params_dict) or ('T_disk' in self.params_dict):
            figs.plot_rk(self)
            figs.plot_spectrum_split(self,plot_veiling=True)
        if self.instrument=='CRIRES':
            figs.plot_spectrum_split(self)

        plt.close('all')
        gc.collect()
     
    def PMN_analyse(self):

        # save posterior as dictionary with key: (posterior, mathtext)
        prefix = self.callback_label if self.callback_label!='final_' else ''
        post=pathlib.Path(f'{self.output_dir}/{prefix}posterior_dict.pickle')
        if post.exists():
            self.posterior=load_pickle(post)
        else:
            def run_analyzer():
                analyzer = pymultinest.Analyzer(n_params=self.parameters.n_params, 
                                            outputfiles_basename=f'{self.output_dir}/{self.prefix}')  # set up analyzer object
                stats = analyzer.get_stats()
                posterior = analyzer.get_equal_weighted_posterior() # equally-weighted posterior distribution
                posterior = posterior[:,:-1] # shape 
                return posterior, stats
            try:
                posterior, stats = run_analyzer()
            except:
                
                def fix_file(input_file):
                    with open(input_file, 'r') as f:
                        lines = f.readlines()
                    fixed_lines = []
                    sci_fix = re.compile(r'(?<![eE])(-?\d+\.\d+)([\+\-]\d{3})') # Match numbers missing 'E'
                    for line in lines:
                        fixed_line = sci_fix.sub(r'\1E\2', line)
                        fixed_lines.append(fixed_line)
                    with open(input_file, 'w') as f:
                        f.writelines(fixed_lines)

                # go through all & fix files with corrupted formatting
                pmn_files = ['pmn_.txt']
                for file in pmn_files:
                    print('\nfixing',file)
                    fix_file(f'{self.output_dir}/{file}')
                print('Fixed corrupt pmn files')
                analyzer = pymultinest.Analyzer(n_params=self.parameters.n_params, 
                                            outputfiles_basename=f'{self.output_dir}/{self.prefix}')  # set up analyzer object
                stats = analyzer.get_stats()
                posterior, stats = run_analyzer()

            posterior_dict={}
            for key in self.parameters.free_params.keys():
                idx=list(self.parameters.free_params).index(key)
                posterior_dict[key]=(posterior[:,idx],self.parameters.free_params[key][1])
            self.posterior=posterior_dict
            save_pickle(self.posterior,post)
            self.bestfit_params = np.array(stats['modes'][0]['maximum a posterior']) # read params of best-fitting model, highest likelihood
            if self.prefix=='pmn_':
                self.lnZ = stats['nested importance sampling global log-evidence']
                print(f"\nFinal lnZ = {self.lnZ}\n")
            else: # when doing exclusion retrievals
                self.lnZ_ex = stats['nested importance sampling global log-evidence']

    def get_quantiles(self, posterior): # input only one posterior at a time

        posterior = np.asarray(posterior)
        posterior = posterior[np.isfinite(posterior)]   # removes nan and inf
        quantiles = np.percentile(posterior, [16,50,84])
        median = quantiles[1]
        plus_err = quantiles[2] - median
        minus_err = quantiles[0] - median

        return median, minus_err, plus_err

    def get_params_and_spectrum(self): 

        final_dict=pathlib.Path(f'{self.output_dir}/params_dict.pickle')
        if final_dict.exists():
            self.params_dict=load_pickle(final_dict)

            for key in self.params_dict.keys():
                self.parameters.params[key]=self.params_dict[key] # set parameters to retrieved values

            # create final spectrum
            self.model_object=pRT_spectrum(self,contribution=True)
            self.model_flux0=self.model_object.make_spectrum()
            self.model_flux=np.zeros_like(self.model_flux0)
            self.summed_contribution= self.model_object.summed_contribution # average over all orders
            phi=self.params_dict['phi']
            for part in range(self.n_parts):
                self.model_flux[part]=phi[part]*self.model_flux0[part] # scale model accordingly
            self.get_ratios() 

        else:
            # create dict of constant params + evaluated params + their errors
            self.params_dict=self.parameters.constant_params.copy() # initialize dict with constant params
            for key in self.parameters.param_keys:
                median,minus_err,plus_err = self.get_quantiles(self.posterior[key][0])
                self.params_dict[key]= median # add median of evaluated params (more robust than bestfit)
                self.params_dict[f'{key}_err']=(minus_err,plus_err)
                self.parameters.params[key]=median # set parameters to retrieved values

            # create final spectrum
            self.model_object=pRT_spectrum(self,contribution=True)
            self.model_flux0=self.model_object.make_spectrum()
            self.summed_contribution= self.model_object.summed_contribution # average over all orders
            self.idx_maxcont=np.where(self.summed_contribution == np.max(self.summed_contribution))[0][0]
            self.params_dict['idx_maxcont'] = self.idx_maxcont
            self.params_dict['T_maxcont'] = self.model_object.temperature[self.idx_maxcont] # temperature at max emission contribution
            self.params_dict['log_P_maxcont'] = np.log10(self.pressure[self.idx_maxcont]) # pressure at max emission contribution
            self.get_ratios() # save isotope & element ratios in final params dict

            if 'log_g' not in self.parameters.free_params:
                self.params_dict['log_g'] = np.log10(self.model_object.gravity)

            # save abundances of species at maximum emission contribution
            if (self.chemistry in ['equchem','quequchem','flexequ']) or (self.chemistry in ['freechem','varchem'] and self.use_partial_pressure==True):
                all_species = self.species_names.copy()
                all_species.append('H2')
                all_species.append('He')
                for species_i in all_species:
                    minus,median,plus=np.percentile(np.log10(np.array(self.VMR_dict[species_i])[:,self.idx_maxcont]), [15.9,50.0,84.1], axis=0)
                    self.params_dict[f'log_{species_i}'] = median
                    self.params_dict[f'log_{species_i}_err'] = (minus-median,plus-median)
            
            # get scaling parameters phi and s^2 of bestfit model through likelihood
            lnL = self.PMN_lnL()
            self.params_dict['phi']=self.LogLike.phi
            self.params_dict['s2']=self.LogLike.s2
            if self.callback_label=='final_':
                self.params_dict['chi2']=self.LogLike.chi2_red # save reduced chi^2 of fiducial model
                self.params_dict['lnZ']=self.lnZ # save lnZ of fiducial model
                self.params_dict['lnL']=lnL

            self.model_flux=np.zeros_like(self.model_flux0)
            phi=self.params_dict['phi']
            for part in range(self.n_parts):
                self.model_flux[part]=phi[part]*self.model_flux0[part] # scale model accordingly

            spectrum=np.full(shape=(self.n_pixels*self.n_parts,2),fill_value=np.nan)
            spectrum[:,0]=self.data_wave.flatten()
            spectrum[:,1]=self.model_flux.flatten()

            if self.callback_label=='final_' and getpass.getuser() == "grasser": # when running from LEM
                save_pickle(self.params_dict,f'{self.output_dir}/params_dict.pickle')
                np.savetxt(f'{self.output_dir}/bestfit_spectrum.txt',spectrum,delimiter=' ',header='wavelength(nm) flux')
        
        return self.params_dict,self.model_flux

    def get_ratios(self): # can only be run after self.evaluate()

        bounds_array=[]
        for key in self.parameters.param_keys:
            bounds=self.parameters.param_priors[key]
            if isinstance(bounds, dict) and bounds.get('type') == 'gaussian':
                bounds_array.append(bounds['bounds'])
            else:
                bounds_array.append(bounds)
        bounds_array=np.array(bounds_array)

        # one value for each parameters = one sample
        all_samples = np.empty((len(self.posterior[list(self.posterior.keys())[0]][0]),len(list(self.parameters.free_params.keys()))))
        for k,key in enumerate(list(self.parameters.free_params.keys())):
            all_samples[:,k] = self.posterior[key][0]

        temp_dist=pathlib.Path(f'{self.output_dir}/temperature_dist.npy')
        VMR_dict=pathlib.Path(f'{self.output_dir}/VMR_dict.pickle')

        # add all ratio posteriors to posterior dict, check C/O if it has been done already
        if ('C/O' in self.posterior.keys()) and temp_dist.exists() and self.chemistry=='freechem' and self.use_partial_pressure==False:
            self.temp_dist=np.load(temp_dist)

        elif temp_dist.exists() and VMR_dict.exists() and (self.chemistry in ['equchem','quequchem','flexequ','varchem'] or (self.chemistry=='freechem' and self.use_partial_pressure==True)):
            self.temp_dist=np.load(temp_dist)
            self.VMR_dict= load_pickle(VMR_dict)
            
        elif (self.chemistry in ['equchem','quequchem','flexequ','varchem']) or (self.chemistry=='freechem' and self.use_partial_pressure==True):

            temperature_distribution=[] # for each of the n_atm_layers
            VMRs=[]

            if self.callback_label=='live_': # sparser sampling just for plotting
                max_len = 10
                if len(all_samples) > max_len:
                    step = int(np.ceil(len(all_samples) / max_len))
                    all_samples = all_samples[::step]

            for j,sample in enumerate(all_samples):
                # sample value is final/real value, need it to be between 0 and 1 depending on prior, same as cube
                cube=(sample-bounds_array[:,0])/(bounds_array[:,1]-bounds_array[:,0])
                self.parameters(cube)
                model_object=pRT_spectrum(self)
                temperature_distribution.append(np.array(model_object.temperature))
                VMRs.append(model_object.VMR_dict)
            self.temp_dist=np.array(temperature_distribution) # shape (n_samples, n_atm_layers)

            self.VMR_dict={}
            for molec in VMRs[0].keys():
                vmr_list=[]
                for i in range(len(all_samples)):
                    vmr_list.append(VMRs[i][molec])
                self.VMR_dict[molec]=vmr_list # reformat to make it easier to work with

            if self.callback_label=='final_' and getpass.getuser() == "grasser": # when running from LEM
                np.save(f'{self.output_dir}/temperature_dist.npy',self.temp_dist)
                save_pickle(self.VMR_dict,f'{self.output_dir}/VMR_dict.pickle')

            if self.chemistry in ['flexequ','varchem'] or (self.chemistry=='freechem' and self.use_partial_pressure==True):
                if 'C/H' not in self.params_dict and self.callback_label=='final_':
                    self.get_FeH_CO_from_maxcont()

        elif self.chemistry in ['freechem']:

            mathtext = []
            get_ratios = []
            if 'log_13CO' in self.parameters.param_keys:
                get_ratios.append(('12CO','13CO'))
                mathtext.append(r'log $^{12}$CO/$^{13}$CO')
            if 'log_C17O' in self.parameters.param_keys:
                get_ratios.append(('12CO','C17O'))
                mathtext.append(r'log $^{12}$CO/C$^{17}$O')
            if 'log_C18O' in self.parameters.param_keys:
                get_ratios.append(('12CO','C18O'))
                mathtext.append(r'log $^{12}$CO/C$^{18}$O')
            if 'log_H2(18)O' in self.parameters.param_keys:
                get_ratios.append(('H2O','H2(18)O'))
                mathtext.append(r'log H$_2$O/H$_2^{18}$O')

            for i,(m1,m2) in enumerate(get_ratios): # isotope ratios    
                p1=self.posterior[f'log_{m1}'][0]
                p2=self.posterior[f'log_{m2}'][0]
                log_ratio=p1-p2
                median,minus_err,plus_err=self.get_quantiles(log_ratio)
                self.params_dict[f'log_{m1}/{m2}']=median
                self.params_dict[f'log_{m1}/{m2}_err']=(minus_err,plus_err)
                self.posterior[f'log_{m1}/{m2}'] = (log_ratio,mathtext[i])

            CO_distribution=[]
            CH_distribution=[]
            temperature_distribution=[] # for each of the n_atm_layers
            
            for j,sample in enumerate(all_samples):
                # sample value is final/real value, need it to be between 0 and 1 depending on prior, same as cube
                cube=(sample-bounds_array[:,0])/(bounds_array[:,1]-bounds_array[:,0])
                self.parameters(cube)
                model_object=pRT_spectrum(self)
                CO_distribution.append(model_object.CO)
                CH_distribution.append(model_object.FeH)
                temperature_distribution.append(np.array(model_object.temperature))
            self.temp_dist=np.array(temperature_distribution) # shape (n_samples, n_atm_layers)

            median,minus_err,plus_err=self.get_quantiles(CO_distribution)
            self.params_dict['C/O']=median
            self.params_dict['C/O_err']=(minus_err,plus_err)
            self.posterior['C/O'] = (np.array(CO_distribution),'C/O')

            median,minus_err,plus_err=self.get_quantiles(CH_distribution)
            self.params_dict['C/H']=median
            self.params_dict['C/H_err']=(minus_err,plus_err)
            self.posterior['C/H'] = (np.array(CH_distribution),'[C/H]')

            if self.callback_label=='final_' and getpass.getuser() == "grasser": # when running from LEM
                np.save(f'{self.output_dir}/temperature_dist.npy',self.temp_dist)
                post=pathlib.Path(f'{self.output_dir}/posterior_dict.pickle') 
                save_pickle(self.posterior,post) # overwrite with ratio posteriors

    def get_FeH_CO_from_maxcont(self):
        
        # calc [Fe/H] & C/O from maximum emission contribution for flexequ
        idx = find_nearest(self.pressure,10**self.params_dict['log_P_maxcont'])
        free_params_orig = self.parameters.free_params.copy()
        params_orig= self.parameters.params.copy()
        
        for species_i in self.species_names:
            # add posterior of each species at max
            self.posterior[f'log_{species_i}'] = (np.log10(np.array(self.VMR_dict[species_i]).T[idx]),species_i)
            self.parameters.free_params[f'log_{species_i}'] = ([-14,-1],rf"log {self.species_info.loc[species_i,'mathtext_name']}")

        self.parameters = Parameters(self.parameters.free_params,self.parameters.constant_params)
        bounds_array=[]
        for key in self.parameters.param_keys:
            bounds=self.parameters.param_priors[key]
            bounds_array.append(bounds)
        bounds_array=np.array(bounds_array)

        CO_distribution=[]
        CH_distribution=[]
        all_samples = np.empty((len(self.posterior[list(self.posterior.keys())[0]][0]),len(list(self.parameters.free_params.keys()))))
        for k,key in enumerate(list(self.parameters.free_params.keys())):
            all_samples[:,k] = self.posterior[key][0]

        if self.callback_label=='live_': # sparser sampling just for plotting
            max_len = 10
            if len(all_samples) > max_len:
                step = int(np.ceil(len(all_samples) / max_len))
                all_samples = all_samples[::step]

        orig_chem = str(self.chemistry)
        orig_P = self.use_partial_pressure

        self.chemistry='freechem' # to get Fe/H & C/O
        self.use_partial_pressure = False
        for j,sample in enumerate(all_samples):
            cube=(sample-bounds_array[:,0])/(bounds_array[:,1]-bounds_array[:,0])
            self.parameters(cube)
            model_object=pRT_spectrum(self)
            CO_distribution.append(model_object.CO)
            CH_distribution.append(model_object.FeH)

        self.chemistry=orig_chem
        self.use_partial_pressure = orig_P
        self.parameters = Parameters(free_params_orig,self.parameters.constant_params)
        self.parameters.params =params_orig

        median,minus_err,plus_err=self.get_quantiles(CO_distribution)
        self.params_dict['C/O']=median
        self.params_dict['C/O_err']=(minus_err,plus_err)
        self.posterior['C/O'] = (np.array(CO_distribution),'C/O')

        median,minus_err,plus_err=self.get_quantiles(CH_distribution)
        self.params_dict['C/H']=median
        self.params_dict['C/H_err']=(minus_err,plus_err)
        self.posterior['C/H'] = (np.array(CH_distribution),'[C/H]')

        if self.callback_label=='final_' and getpass.getuser() == "grasser": # when running from LEM
            post=pathlib.Path(f'{self.output_dir}/posterior_dict.pickle') 
            save_pickle(self.posterior,post) # overwrite with ratio posteriors

    def evaluate(self,only_abundances=False,only_params=None,split_corner=True,
                 callback_label='final_',makefigs=True):
        self.callback_label=callback_label
        self.PMN_analyse() # get/save bestfit params and final posterior
        self.params_dict,self.model_flux=self.get_params_and_spectrum() # all params + scaling phi + s^2
        self.model_flux[~self.mask_isfinite] = np.nan
        if makefigs:
            if callback_label=='final_':
                figs.make_all_plots(self,only_abundances=only_abundances,only_params=only_params,split_corner=split_corner)
            else:
                figs.summary_plot(self,show_params='all')
        CCF_results=pathlib.Path(f'{self.output_dir}/CCF_ACF_dict.pickle')
        if CCF_results.exists():
            self.ccf_acf_dict=load_pickle(CCF_results)
        
    def cross_correlation(self,ccf_species,noiserange=100,save_template=False): # can only be run after evaluate()

        ccf_dict={}
        ccf_acf_dict={} # save cross-correlations and auto-correlations
        orig_params_dict=self.params_dict
        CCF_results=pathlib.Path(f'{self.output_dir}/CCF_ACF_dict.pickle')
        if CCF_results.exists():
            ccf_acf_dict=load_pickle(CCF_results)

        if isinstance(ccf_species, list)==False:
            ccf_species=[ccf_species] # if only one, make list so that it works in the for loop

        RVs=np.arange(-500,500,1) # km/s

        for j,species_i in enumerate(ccf_species):

            if CCF_results.exists()==False:
                # function only for CRIRES spectra anyway
                crires_shape = (self.n_orders,self.n_dets,self.n_pixels)
                to_reshape= [self.data_flux, self.data_err, self.data_wave, self.mask_isfinite]
                data_flux, data_err, data_wave, mask_isfinite = [var.reshape(crires_shape) for var in to_reshape]
                Cov = self.Cov.reshape(7,3)
                phi = self.params_dict['phi'].reshape(7,3)

                # create final model without opacity from a certain specie
                exclusion_dict=self.params_dict.copy()
                if self.chemistry=='freechem':
                    exclusion_dict[f'log_{species_i}']=-14 # exclude from model
                   
                # interpolate=False for CCF: not interpolated onto data_wave so that wl padding not cut off
                self.parameters.params=exclusion_dict
                if self.chemistry in ['equchem','quequchem','flexequ']:
                    exclusion_model_object=pRT_spectrum(self,interpolate=False,leave_out=species_i)
                else:
                    exclusion_model_object=pRT_spectrum(self,interpolate=False)
                
                # shape (n_orders,length of uninterpolated wavelengths), must still be interpolated 
                exclusion_model,exclusion_model_wl=exclusion_model_object.make_spectrum()

                self.parameters.params=orig_params_dict        
                model_flux_broad,_=pRT_spectrum(self,interpolate=False).make_spectrum()

                beta=1.0-RVs/const.c.to('km/s').value
                CCF = np.zeros((self.n_orders,self.n_dets,len(RVs)))
                ACF = np.zeros((self.n_orders,self.n_dets,len(RVs))) # auto-correlation
                if save_template:
                    template_wl_list,template_fl_list = [],[]

                residuals_list,species_template_list = [],[]

                for order in range(self.n_orders):
                    for det in range(self.n_dets):

                        if np.isnan(data_flux[order,det]).all():
                            residuals_list.append([])
                            species_template_list.append([])
                            if save_template:
                                template_wl_list.append([])
                                template_fl_list.append([])
                            continue # skip empty order/det, CCF and ACF remains 0 

                        wl_data=data_wave[order,det,mask_isfinite[order,det]] 
                        fl_data=data_flux[order,det,mask_isfinite[order,det]] 

                        wl_excl=exclusion_model_wl[order]
                        fl_excl=exclusion_model[order]*phi[order,det]
                        fl_final=model_flux_broad[order]*phi[order,det]

                        # data minus model without certain species
                        fl_excl_rebinned=interp1d(wl_excl,fl_excl)(wl_data) # rebin to allow subtraction
                        residuals=fl_data-fl_excl_rebinned
                        residuals-=np.nanmean(residuals) # mean should be at zero
                        Cov[order,det].get_cholesky() # in case it hasn't been called yet
                        cov_0_res=Cov[order,det].solve(residuals)
                        residuals_list.append(residuals)
                        
                        # excluded species template: complete final model minus final model w/o species
                        species_template=fl_final-fl_excl
                        if save_template:
                            template_wl_list.append(wl_excl)
                            template_fl_list.append(species_template)
                        species_template_rebinned=interp1d(wl_excl,species_template)(wl_data) # rebin for Cov
                        species_template_rebinned-=np.nanmean(species_template_rebinned) # mean should be at zero
                        species_template_list.append(species_template_rebinned)
                        cov_0_temp=Cov[order,det].solve(species_template_rebinned)
                        wl_shift=wl_data[:, np.newaxis]*beta[np.newaxis, :]
                        template_shift=interp1d(wl_excl,species_template)(wl_shift) # interpolate template onto shifted wl
                        template_shift-=np.nanmean(template_shift) # mean should be at zero

                        CCF[order,det]=(template_shift.T).dot(cov_0_res)
                        ACF[order,det]=(template_shift.T).dot(cov_0_temp)

                figs.plot_ccf_residuals(self,residuals_list,species_template_list,species_i)
                CCF_sum=np.sum(np.sum(CCF,axis=0),axis=0) # sum CCF over all orders detectors
                ACF_sum=np.sum(np.sum(ACF,axis=0),axis=0)
                noise=np.std(CCF_sum[np.abs(RVs)>noiserange]) # mask out regions close to expected RV
                try:
                    CCF_norm = CCF_sum/noise # get ccf map in S/N units
                    ACF_norm = ACF_sum/noise
                except:
                    print('Invalid nosie = ',noise)
                    CCF_norm = CCF_sum
                    ACF_norm = ACF_sum

                if save_template:
                    spectrum=np.full(shape=(len(flatten_list(template_fl_list)),2),fill_value=np.nan)
                    spectrum[:,0]=flatten_list(template_wl_list)
                    spectrum[:,1]=flatten_list(template_fl_list)
                    np.savetxt(f'{self.output_dir}/{species_i}_ccf_template.txt',spectrum,delimiter=' ',header='wavelength(nm) flux')
                SNR=CCF_norm[np.where(RVs==0)[0][0]]
                self.parameters.params=orig_params_dict

            else:
                CCF_norm,ACF_norm,SNR = ccf_acf_dict[species_i]

            ccf_dict[f'SNR_{species_i}']=SNR
            ccf_acf_dict[species_i]=(CCF_norm,ACF_norm,SNR)
            print(f'{species_i} S/N =',np.round(SNR,decimals=2))

        self.ccf_acf_dict = ccf_acf_dict
        figs.CCF_plot_all(self,ccf_species,noiserange=100)

        if CCF_results.exists()==False and ccf_species==self.species_names:
            save_pickle(ccf_acf_dict,CCF_results)

        self.params_dict.update(ccf_dict)
        save_pickle(self.params_dict,f'{self.output_dir}/params_dict.pickle') # overwrite with CCF SNR

        return ccf_dict
    
    def get_continious_spectrum(self): # for cross-corr
        besfit =pathlib.Path(f'{self.output_dir}/bestfit_spectrum_continuous.txt')
        if besfit.exists():
            file=np.genfromtxt(besfit,skip_header=1,delimiter=' ')
            model_flux, model_wl =file[:,0],file[:,1]
        else:
            model_flux, model_wl =pRT_spectrum(self,interpolate=False).make_spectrum()
            spectrum=np.full(shape=(len(flatten_list(model_flux)),2),fill_value=np.nan)
            spectrum[:,0]=flatten_list(model_wl)
            spectrum[:,1]=flatten_list(model_flux)
            np.savetxt(besfit,spectrum,delimiter=' ',header='wavelength(nm) flux')
        return model_wl, model_flux

    def bayes_evidence(self,bayes_species,evidence_dict,retrieval_output_dir):

        bayes_dict=evidence_dict
        self.output_dir=pathlib.Path(f'{self.output_dir}/evidence_retrievals') # store output in separate folder
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print('\n ----------------- Current bayes_dict= ----------------- \n',bayes_dict)

        if isinstance(bayes_species, list)==False:
            bayes_species=[bayes_species] # if only one, make list so that it works in for loop

        for species_i in bayes_species: # exclude from retrieval

            self.prefix=f'pmn_wo{species_i}_' 
            finish=pathlib.Path(f'{self.output_dir}/final_wo{species_i}_posterior_dict.pickle')
            if finish.exists():
                print(f'\n ----------------- Evidence retrieval for {species_i} already done ----------------- \n')
                setback_prior=False
            else:
                print(f'\n ----------------- Starting evidence retrieval for {species_i} ----------------- \n')
                setback_prior=True
                if self.chemistry=='freechem':
                    original_prior=self.parameters.param_priors[f'log_{species_i}']
                    self.parameters.param_priors[f'log_{species_i}']=[-15,-14] # exclude from retrieval
                elif self.chemistry in ['equchem','quequchem','flexequ']:
                    if species_i=='13CO':
                        key='log_C12_13_ratio'
                    elif species_i=='H2(18)O':
                        key='log_H2O16_18_ratio'
                    original_prior=self.parameters.param_priors[key]
                    self.parameters.param_priors[key]=[14,15] # exclude from retrieval

                self.callback_label=f'live_wo{species_i}_'
                self.PMN_run(N_live_points=self.N_live_points,evidence_tolerance=self.evidence_tolerance,resume=True)
            
            self.callback_label=f'final_wo{species_i}_'
            self.evaluate(callback_label=self.callback_label) # gets self.lnZ_ex
            ex_model=pRT_spectrum(self).make_spectrum()      
            lnL = self.LogLike(ex_model, self.Cov) # call function to generate chi2
            chi2_ex = self.LogLike.chi2_red # reduced chi^2
            lnB,sigma=self.compare_evidence(self.lnZ, self.lnZ_ex)
            print(f'sigma_{species_i}=',sigma)
            bayes_dict[f'lnBm_{species_i}']=lnB
            bayes_dict[f'sigma_{species_i}']=sigma
            bayes_dict[f'chi2_wo_{species_i}']=chi2_ex  
            save_pickle(bayes_dict,f'{retrieval_output_dir}/evidence_dict.pickle') # save results at each step

            # set back param priors for next retrieval
            if setback_prior==True:
                if self.chemistry=='freechem':
                    self.parameters.param_priors[f'log_{species_i}']=original_prior 
                elif self.chemistry in ['equchem','quequchem','flexequ']:
                    if species_i=='13CO':
                        key='log_C12_13_ratio'
                    elif species_i=='H2(18)O':
                        key='log_H2O16_18_ratio'
                    self.parameters.param_priors[key]=original_prior
            
        return bayes_dict

    def compare_evidence(self,ln_Z_A,ln_Z_B):
        '''
        Convert log-evidences of two models to a sigma confidence level
        Originally from Benneke & Seager (2013), adapted from samderegt/retrieval_base
        '''

        from scipy.special import lambertw as W
        from scipy.special import erfcinv

        ln_B = ln_Z_A-ln_Z_B
        sign=1
        if ln_B<0: # ln_Z_B larger -> second model favored
            sign=-1
            ln_B*=sign # can't handle negative values (-> nan), multiply back later
        try:
            p = np.real(np.exp(W((-1.0/(np.exp(ln_B)*np.exp(1))),-1)))
            sigma = np.sqrt(2)*erfcinv(p)
        except RuntimeWarning:
            sigma=np.inf 
        return ln_B*sign,sigma*sign

    def run_retrieval(self): 

        retrieval_output_dir=self.output_dir # save end results here

        print(f'\n ------ {self.target.name} - {self.chemistry} - {self.PT_type} - Nlive: {self.Nlive} - ev: {self.evtol} - {self.folder_suffix} ------ \n')

        # run main retrieval if hasn't been run yet, else skip to cross-corr and bayes
        final_dict=pathlib.Path(f'{self.output_dir}/params_dict.pickle')
        if final_dict.exists()==False:
            print('\n ----------------- Starting main retrieval. ----------------- \n')
            self.PMN_run(N_live_points=self.Nlive,evidence_tolerance=self.evtol)
        else:
            print('\n ----------------- Main retrieval exists. ----------------- \n')
        self.evaluate() # created and saves self.params_dict

        if self.instrument=='CRIRES': # ccf only for high-res
            ccf_dict=self.cross_correlation(self.species_names) # cross-corr all species
            self.params_dict.update(ccf_dict)
            save_pickle(self.params_dict,f'{retrieval_output_dir}/params_dict.pickle') # overwrite with added CCF SNR
    
        print('Parameters:\n',self.params_dict)
        bayes_species = self.parameters.params['evidence_retrievals_species']
        if bayes_species!= []:
            evidence_dict=pathlib.Path(f'{retrieval_output_dir}/evidence_dict.pickle')
            if evidence_dict.exists()==False: # to avoid overwriting sigmas from other evidence retrievals
                print('\n ----------------- Creating evidence dict ----------------- \n')
                self.evidence_dict={}
            else:
                print('\n ----------------- Continuing existing evidence dict ----------------- \n')
                self.evidence_dict= load_pickle(evidence_dict)

            bayes_dict=self.bayes_evidence(bayes_species,evidence_dict=self.evidence_dict,retrieval_output_dir=retrieval_output_dir)
            print('\n ----------------- Final evidence dict ----------------- \n',bayes_dict)
            save_pickle(bayes_dict,f'{retrieval_output_dir}/evidence_dict.pickle') # save new results in separate dict

        output_file=pathlib.Path('retrieval.out')
        if output_file.exists():
            os.system(f"mv {output_file} {retrieval_output_dir}")

        print('\n ----------------- Done ---------------- \n')

        
        

