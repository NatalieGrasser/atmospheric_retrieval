import numpy as np
import pathlib
import os
import re
from astropy.coordinates import SkyCoord
from PyAstronomy.pyasl import helcorr
from utils import * 

class Target:

    def __init__(self,name,properties):
        """
        Class representing an observed target spectrum and its associated metadata.

        This class handles loading spectral data, computing observational properties
        (e.g., barycentric velocity correction, spectral resolution), and preparing
        data structures for further analysis such as covariance estimation.

        Parameters
        ----------
        name : str
            Name of the target. Also used to locate the project directory.
        properties : dict
            Dictionary containing target-specific settings and observational metadata.
        """

        self.name = name
        self.instrument = properties['instrument']
        self.color= properties['color']
        self.remove_continuum = properties['remove_continuum']
        self.emission_or_transmission = properties['emission_or_transmission']

        # use project-specific custom species_info if exists, else use default
        self.project_path = f'{os.getcwd()}/{self.name}'
        custom_species_info = pathlib.Path(f'{self.project_path}/species_info.csv')
        if custom_species_info.exists(): 
            self.species_info = pd.read_csv(custom_species_info, index_col=0)
        else:
            self.species_info = pd.read_csv(os.path.join('species_info.csv'), index_col=0)
        
        self.n_parts=1
        if self.instrument == 'CRIRES':
            self.n_orders=7
            self.n_dets=3
            self.n_parts=self.n_orders*self.n_dets
            self.n_pixels=2048
            self.K2166=np.array([[[1921.318,1934.583], [1935.543,1948.213], [1949.097,1961.128]],
                            [[1989.978,2003.709], [2004.701,2017.816], [2018.708,2031.165]],
                            [[2063.711,2077.942], [2078.967,2092.559], [2093.479,2106.392]],
                            [[2143.087,2157.855], [2158.914,2173.020], [2173.983,2187.386]],
                            [[2228.786,2244.133], [2245.229,2259.888], [2260.904,2274.835]],
                            [[2321.596,2337.568], [2338.704,2353.961], [2355.035,2369.534]],
                            [[2422.415,2439.061], [2440.243,2456.145], [2457.275,2472.388]]])          

        # observational setup defined
        if all(k in properties for k in ('ra', 'dec', 'JD','obs_long','obs_lat', 'obs_alt')): 
            coords = SkyCoord(ra=properties['ra'], dec=properties['dec'], frame='icrs')
            self.vbary, _ = helcorr(obs_long=properties['obs_long'], obs_lat=properties['obs_lat'], 
                                    obs_alt=properties['obs_alt'], ra2000=coords.ra.value,
                                    dec2000=coords.dec.value,jd=properties['JD'])
        else:
            self.vbary = 0

        data_spec = properties['input_spectrum']
        if isinstance(data_spec, list): # possibly inhomogeneous
            self.n_parts = len(data_spec)
            self.wl,self.fl,self.err,self.n_pixels = [],[],[],[]
            for data_file in data_spec:
                wl,fl,err = self.load_spectrum(data_file)
                self.wl.append(wl)
                self.fl.append(fl)
                self.err.append(err)
                self.n_pixels.append(len(wl))
        else:
            wl,fl,err = self.load_spectrum(data_spec)
            if wl.dtype == object: # inhomogeneous
                self.n_parts = len(wl)
                self.wl,self.fl,self.err,self.n_pixels = [],[],[],[]
                for w,f,e in zip(wl,fl,err):
                    self.wl.append(w)
                    self.fl.append(f)
                    self.err.append(e)
                    self.n_pixels.append(len(w))
            else:
                self.n_pixels = len(wl)
                self.wl  = np.reshape(wl,  (self.n_parts, self.n_pixels))
                self.fl  = np.reshape(fl,  (self.n_parts, self.n_pixels))
                self.err = np.reshape(err, (self.n_parts, self.n_pixels))

        self.mask_isfinite = self.get_mask_isfinite()

        if self.remove_continuum:
            for i in range(len(fl)):
                if np.isnan(self.fl[i]).all()==False:
                    mask = self.mask_isfinite[i]
                    self.fl[i] = fft_remove_continuum(self.fl[i])
                    self.fl[i][~mask] =np.nan

        self.fwhm = properties['fwhm'] if 'fwhm' in properties else None
        self.spectral_resolution = self.calc_resolution()
        #self.spectral_resolution = np.nanmedian(self.spectral_resolution)

    def load_spectrum(self,input_file):
        """
        Load the observed spectrum from file and reshape into internal format.

        Parameters
        ----------
        input_file : str or None
            Name of the input file. If None, defaults to '<target_name>_spectrum.txt'.

        Returns
        -------
        wl, fl, err : ndarray
            Wavelength, flux, uncertainty arrays of shape (n_parts, n_pixels) each.
        """

        def _clean_col(name):
            """Normalize column names for matching"""
            name = name.lower()
            name = re.sub(r'\(.*?\)', '', name)  # remove units in brackets
            return name.strip()

        def _find_column(columns, keywords, exclude_keywords=None):
            exclude_keywords = exclude_keywords or []

            # First pass: exact/strong matches
            for col in columns:
                clean = _clean_col(col)

                if any(k == clean for k in keywords):
                    if not any(ex in clean for ex in exclude_keywords):
                        return col

            # Second pass: substring matches
            for col in columns:
                clean = _clean_col(col)

                if any(k in clean for k in keywords):
                    if not any(ex in clean for ex in exclude_keywords):
                        return col

            return None
        
        # check if transit depth, or if it should still be squared
        def _needs_square(colname):
            clean = _clean_col(colname)

            # indicators that it's already squared
            squared_indicators = ["2", "^2", "**2", "squared", "sq"]

            # indicators that it's a ratio
            ratio_indicators = ["rp/rs", "rprs", "rp_rs", "rplanet/rstar"]

            has_ratio = any(r in clean for r in ratio_indicators)
            has_square = any(s in clean for s in squared_indicators)

            return has_ratio and not has_square

        # 1. Resolve file path
        if input_file is not None:
            file = pathlib.Path(f'{self.project_path}/{input_file}')
        else:
            file = pathlib.Path(f'{self.project_path}/{self.name}_spectrum.txt')

        if not file.exists():
            raise FileNotFoundError(file)

        # 2. Detect format
        suffix = file.suffix.lower()
        header = ''
        if suffix == ".npz":
            wl = data['wavelengths']
            fl = data['flux']
            err = data['err']
        else:
            if suffix == ".csv":
                df = pd.read_csv(file)
            else:
                # try reading header
                with open(file, 'r') as f:
                    first_line = f.readline()

                if first_line.startswith("#"):
                    header = first_line[1:].strip().split()
                    data = np.genfromtxt(file, skip_header=1)
                    df = pd.DataFrame(data, columns=header)
                else:
                    # no header → fallback
                    data = np.genfromtxt(file)
                    df = pd.DataFrame(data)

            # 3. Identify columns
            cols = df.columns
            wl_col  = _find_column(cols, ["wave", "wl", "lambda"])
            fl_col = _find_column(cols, ["flux", "depth", "rp", "rplanet"],
                                exclude_keywords=["err", "unc", "sigma"])
            err_col = _find_column(cols, ["err","unc","sigma"])

            # 4. Fallback if no names
            if wl_col is None or fl_col is None:
                # assume standard ordering
                wl_col  = cols[0]
                fl_col  = cols[1]
                err_col = cols[2] if len(cols) > 2 else None

            # 5. Assign arrays
            wl  = df[wl_col].to_numpy()
            fl  = df[fl_col].to_numpy()
            err = df[err_col].to_numpy() if err_col else np.full_like(fl, np.nan)

            if any('ppm' in s for s in header):
                fl*=1e-6
                err*=1e-6

            # convert Rp/Rs → depth if needed
            if _needs_square(fl_col):
                #print(f"[INFO] Column '{fl_col}' detected as Rp/Rs → squaring to get transit depth.")
                fl = fl**2

            if self.emission_or_transmission=='transmission':
                fl*=100 # so that it is in % (nicer)
                err*=100

        return wl,fl,err
    
    def calc_resolution(self):
        """
        Compute the spectral resolution of the observed spectrum.

        If FWHM (in pixels) is provided, the resolution is calculated as:
            R = lambda / (FWHM * pixel_size)

        Otherwise, the resolution is estimated from the wavelength sampling.

        Returns
        -------
        spectral_resolution : float or ndarray
            Median spectral resolution (if FWHM provided) or array of resolution
            values per wavelength point.
        """
        
        res = []
        if self.fwhm is not None:
            for i in range(self.n_parts):
                wave = self.wl[i]
                pix_size = np.median(np.diff(wave))
                fwhm = pix_size*self.fwhm
                R = np.median(wave)/fwhm
                res.append(np.nanmedian(R))
            return res

        else:
            res = []
            for i in range(self.n_parts):
                wl = self.wl[i]
                wl = np.sort(wl) # ensure sorted!
                delta_wl = np.gradient(wl)
                R = wl / delta_wl
                res.append(np.nanmedian(R))
            return res

    def get_mask_isfinite(self):
        """
        Generate a mask identifying valid (finite) flux and error values.

        Returns
        -------
        mask_isfinite : ndarray of bool
            Boolean array of shape (n_parts, n_pixels), where True indicates
            valid (finite) flux and uncertainty values.
        """

        self.mask_isfinite= []
        for i in range(self.n_parts):
            mask_i = np.isfinite(self.fl[i]) & np.isfinite(self.err[i])
            self.mask_isfinite.append(mask_i)
        return self.mask_isfinite
    
    def prepare_for_covariance(self):
        """
        Prepare wavelength separations and effective uncertainties for covariance modeling.

        Returns
        -------
        separation : ndarray of object
            Array of length n_parts, each entry containing a 2D array of pairwise
            wavelength separations.
        err_eff : ndarray of object
            Array of length n_parts containing the median uncertainty per part.
        """

        self.separation = np.empty((self.n_parts), dtype=object)
        self.err_eff = np.empty((self.n_parts), dtype=object)
        for i in range(self.n_parts):
            mask_i = self.mask_isfinite[i] # Mask the arrays, on-the-spot is slower
            wave_i = self.wl[i][mask_i]
            separation_i = np.abs(wave_i[None,:]-wave_i[:,None]) # wavelength separation
            self.separation[i] = separation_i
            err_i = self.err[i][mask_i]  
            self.err_eff[i] = np.nanmedian(err_i) if err_i.size != 0 else np.nan
        return self.separation,self.err_eff