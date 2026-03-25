import numpy as np
import pandas as pd
import os
import pickle
import pathlib
import astropy.constants as const
from astropy import units as u
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import convolve1d
from petitRADTRANS import physical_constants as cst

from scipy import constants as sc
import pathlib
import warnings

warnings.simplefilter("error", RuntimeWarning)  # Convert warnings to exceptions
import getpass
if getpass.getuser() == "grasser": # when runnig from LEM
    path_tables = '/net/lem/data2/regt/fastchem_tables'
elif getpass.getuser() == "natalie": # when testing from my laptop
    path_tables = '/home/natalie/fastchem_tables'

def scale_between(ymin,ymax,arr):
    try:
        scale=(ymax-ymin)/(np.nanmax(arr)-np.nanmin(arr))
    except RuntimeWarning:
        scale = (ymax-ymin)
    scaled_arr=scale*(arr-np.nanmin(arr))+ymin
    return scaled_arr

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def save_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
def flatten_list(xss):
    return [x for xs in xss for x in xs]

def ensure_quantity(x, unit):
    return x.to(unit) if isinstance(x, u.Quantity) else x * unit

def darken_color(color, amount=0.7):
    import matplotlib.colors as mcolors
    import colorsys
    try:
        c = mcolors.to_rgb(color)
    except ValueError:
        raise ValueError(f"Invalid color: {color}")
    h, l, s = colorsys.rgb_to_hls(*c)
    l *= amount  # reduce lightness
    darker_rgb = colorsys.hls_to_rgb(h, l, s)
    return darker_rgb

def get_ratios(retr_obj,equ_too=False): # in case not all ratios are in retrieval
    if retr_obj.chemistry in ['equchem','quequchem','flexequ']:
        if retr_obj.chemistry in ['equchem','quequchem']:
            ratios_default = ['C/O','Fe/H']
        elif retr_obj.chemistry=='flexequ':
            ratios_default = ['C/O','C/H']
        check_ratios = ['log_C12_13_ratio','log_O16_17_ratio','log_O16_18_ratio','log_H2O16_18_ratio']
        for r in check_ratios:
            if r in retr_obj.parameters.param_keys:
                ratios_default.append(r)
    elif retr_obj.chemistry in ['freechem','varchem']:
        suffix='_0' if retr_obj.chemistry=='varchem' else ''
        ratios_default = ['C/O','C/H']
        ratios_default_equ = ['C/O','Fe/H']
        check_ratios = ['log_12CO/13CO','log_12CO/C17O','log_12CO/C18O','log_H2O/H2(18)O']
        ratios_equ = ['log_C12_13_ratio','log_O16_17_ratio','log_O16_18_ratio','log_H2O16_18_ratio']
        check_molec = [f'13CO{suffix}',f'C17O{suffix}',f'C18O{suffix}',f'H2(18)O{suffix}']
        for m,r,e in zip(check_molec,check_ratios,ratios_equ):
            if f'log_{m}' in retr_obj.parameters.param_keys:
                ratios_default.append(r)
                ratios_default_equ.append(e)
    if equ_too==False:
        return ratios_default
    else:
        return ratios_default, ratios_default_equ

class PSG_input: # for LIFE retrievals

    def __init__(self,name):
        self.name = name
        self.table = self.create_table()
        self.pressure = self.table['Pressure'].to_numpy()
        self.temperature = self.table['Temperature'].to_numpy()

    def create_table(self): # convert PSG input file into useable table

        with open(f'./LIFE/{self.name}/{self.name}_psg_input.txt', "r") as file:
            lines = file.readlines()

        rows = []
        for line in lines:
            if "<ATMOSPHERE-LAYER-" in line:
                index = line.index(">")
                rows.append(line[index+1:-1]) # remove \n from end of row

        columns = [ "Pressure", "Temperature", "Altitude", "H2", "He", "H2O", "CH4", "C2H6", "CO2", "C2H2", "C2H4", "CO",
                    "H2CO", "NH3", "SO2", "H2S", "SO", "CS2", "OCS", "DMS", "C2H6S2"]

        df = pd.DataFrame([row.split(",") for row in rows])
        df.columns = columns
        df = df.astype(float) # Convert all columns to float
        df = df.iloc[::-1] # reverse order, bc pRT reads temps from top to bottom of atmosphere

        return df

# from DGonzalezPicos/broadpy
class InstrumentalBroadening:
    
    c = const.c.to(u.km/u.s).value
    sqrt8ln2 = np.sqrt(8 * np.log(2))
    
    available_kernels = ['gaussian','gaussian_variable']
    
    def __init__(self, x, y):
        
        self.x = x # units of wavelength
        self.y = y # units of flux (does not matter)
        self.spacing = np.mean(2*np.diff(self.x) / (self.x[1:] + self.x[:-1]))
    
    def __call__(self, res=None, fwhm=None, gamma=None, truncate=4.0, kernel='auto'):
        '''Instrumental broadening
        provide either instrumental resolution lambda/delta_lambda or FWHM in km/s'''
        kernel = self.__read_kernel(res=res, fwhm=fwhm, gamma=gamma) if kernel == 'auto' else kernel
        
        if kernel == 'gaussian':
            fwhm = fwhm if fwhm is not None else (self.c / res)
            _kernel = self.gaussian_kernel(fwhm, truncate)
            
        if kernel == 'gaussian_variable':
            _kernels, lw = self.gaussian_variable_kernel(fwhm, truncate)
            y_pad = np.pad(self.y, (lw, lw), mode='reflect')
            y_matrix = np.lib.stride_tricks.sliding_window_view(y_pad, window_shape=(2 * lw + 1))
            y_lsf = np.einsum('ij, ij->i', _kernels, y_matrix)
            return y_lsf
            
        y_lsf = convolve1d(self.y, _kernel, mode='nearest')
        return y_lsf
    
    @classmethod
    def gaussian_profile(self, x, x0, sigma):
        '''Gaussian function'''
        return np.exp(-0.5 * ((x - x0) / sigma)**2)# / (sigma * np.sqrt(2*np.pi))
    
    def gaussian_kernel(self,fwhm,truncate=4.0,):
        ''' Gaussian kernel
        
        Parameters
        ----------
        fwhm : float
            Full width at half maximum of the Gaussian kernel in km/s
        truncate : float
            Truncate the kernel at this many standard deviations from the mean (default: 4.0)
        
        Returns
        -------
        kernel : array
            Convolution kernel
        '''
        # Adapted from scipy.ndimage.gaussian_filter1d        
        sd = (fwhm/self.c) / self.sqrt8ln2 / self.spacing
        lw = int(truncate * sd + 0.5)
    
        kernel_x = np.arange(-lw, lw+1)
        kernel = self.gaussian_profile(kernel_x, 0, sd)
        kernel /= np.sum(kernel)  # normalize the kernel
        return kernel
    
    def gaussian_variable_kernel(self, fwhm, truncate=4.0):
        ''' Gaussian kernel with variable FWHM
        
        Parameters
        ----------
        fwhm : array
            Full width at half maximum of the Gaussian kernel in km/s
        truncate : float
            Truncate the kernel at this many standard deviations from the mean (default: 4.0)
        
        Returns
        -------
        kernel : array
            Convolution kernel
        '''
        sd = (fwhm/self.c) / self.sqrt8ln2 / self.spacing
        lw = int(truncate * sd.max() + 0.5)
        x = np.arange(-lw, lw + 1)
        
        # Use broadcasting to create a 2D array of Gaussian kernels
        kernels = np.exp(-0.5 * (x[None, :] / sd[:, None]) ** 2)
        kernels /= kernels.sum(axis=1)[:, None]
        return kernels, lw

def fill_nan_nearest(arr): # fill nans with the nearest non-nan
    orig_shape = arr.shape
    arr = arr.flatten()
    nans = np.isnan(arr)
    not_nans = np.where(~nans)[0]
    if not len(not_nans):
        raise ValueError("Array is all NaNs.")
    nearest_index = np.abs(np.subtract.outer(np.arange(len(arr)), not_nans)).argmin(1)
    arr = arr[not_nans[nearest_index]]
    arr = arr.reshape(orig_shape)
    return arr

def fft_remove_continuum(flux, lower_cutoff=None, divide=True, return_continuum=False):

    if np.isnan(flux).all():
        if return_continuum:
            return flux, np.ones(flux.shape)
        else:
            return flux

    # Remove frequencies below ~1/100 to 1/200 of array length
    if lower_cutoff is None:
        lower_cutoff = len(flux) // 150  # ≈ 13 if n_pixels=2048

    flux = np.asarray(flux)
    n = len(flux)
    
    # can't handle NaNs
    isnan = np.isnan(flux)
    if np.any(isnan):
        flux = np.interp(np.arange(n), np.arange(n)[~isnan], flux[~isnan])
        
    # FFT and filtering
    fft_flux = np.fft.rfft(flux)
    fft_continuum = fft_flux.copy()
    fft_highfreq = fft_flux.copy()
    fft_continuum[lower_cutoff:] = 0 # Zero out high frequencies = continuum
    fft_highfreq[:lower_cutoff] = 0 # Zero out low frequencies = continuum-removed signal

    # Inverse FFT to get time-domain signals
    continuum = np.fft.irfft(fft_continuum, n=n)

    # divide by extracted continuum
    if divide:
        filtered = flux/continuum
    # use only high-frequencies as continuum-removed flux
    else:
        filtered = np.fft.irfft(fft_highfreq, n=n) + np.ones(n)

    filtered[isnan] = np.nan
    if return_continuum:
        return filtered, continuum
    else:
        return filtered

def generate_skewed_p_nodes(pressure, n_nodes=7, skew=0.8):
    """
    Generate pressure nodes in log space, skewed toward the bottom (high pressure).
    
    Parameters:
        p_min: float, minimum pressure (top of atmosphere)
        p_max: float, maximum pressure (bottom)
        n_nodes: int, total number of pressure nodes
        skew: float > 0, higher means more concentrated at top (skew > 1),
                        skew < 1 concentrates at bottom.
    
    Returns:
        Array of pressures in bars.
    """
    p_min = np.min(pressure)
    p_max = np.max(pressure)
    x = np.linspace(0, 1, n_nodes)
    x_skewed = x**skew  # Skewed toward 1 (bottom) for skew > 1
    log_p = np.log10(p_min) + x_skewed * (np.log10(p_max) - np.log10(p_min))
    return log_p

def planck_lambda_um(T, lam_um):
    """
    Planck function B_lambda in units of W / m² / μm / sr.

    Handles scalar or array T and scalar or array lam_um.
    Returns:
      T scalar, λ scalar → scalar
      T scalar, λ array  → (Nwave,)
      T array,  λ scalar → (Nsamples,)
      T array,  λ array  → (Nsamples, Nwave)
    """
    T = np.asarray(T)
    lam_um = np.asarray(lam_um)

    # Convert λ to meters
    lam_m = lam_um * 1e-6

    c1 = 2 * sc.h * sc.c**2
    c2 = sc.h * sc.c / sc.k

    # Prepare shapes with broadcasting:
    # If λ is scalar → lam_m_b is shape (1,)
    # If T is scalar → T_b is shape (1,)
    if lam_m.ndim == 0:
        lam_m_b = lam_m  # scalar
    else:
        lam_m_b = lam_m[None, :]  # (1, Nwave)

    if T.ndim == 0:
        T_b = T  # scalar
    else:
        T_b = T[:, None]  # (Nsamples, 1)

    exponent = c2 / (lam_m_b * T_b)
    B_lambda = (c1 / lam_m_b**5) / (np.exp(exponent) - 1)

    # Convert per meter → per micron
    B_lambda = B_lambda * 1e-6

    # Return shapes matching user input
    if T.ndim == 0 and lam_m.ndim == 0:
        return B_lambda  # scalar
    if T.ndim == 0:
        return B_lambda[0]  # (Nwave,)
    if lam_m.ndim == 0:
        return B_lambda[:, 0]  # (Nsamples,)

    return B_lambda  # (Nsamples, Nwave)