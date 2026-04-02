import os
import cloud_cond as cloud_cond
from pRT_model import pRT_spectrum
from utils import *
import numpy as np
import corner
import copy
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import pyplot as plt, ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from astropy import constants as const
from matplotlib.ticker import MultipleLocator
from labellines import labelLines
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgb
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
import colorsys
import matplotlib.ticker as ticker
import matplotlib as mpl
mpl.rcParams['legend.handlelength'] = 1.7
import warnings
import pathlib
import math
import re
import getpass
import pandas as pd
import glob
from petitRADTRANS.radtrans import Radtrans
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
from petitRADTRANS.plotlib import plot_radtrans_opacities
from petitRADTRANS.plotlib import plot_opacity_contributions

warnings.filterwarnings("ignore", category=UserWarning) 
if getpass.getuser() == "grasser": # when runnig from LEM
    path_tables = '/net/lem/data2/regt/fastchem_tables'
    bobcat_dir = "/net/lem/data1/grasser/models/SonoraBobcat/structures_m+0.0"  # folder with the files
    sphinx_pt_dir = "/net/lem/data1/grasser/models/SPHINX/ATMS/*"
    sonora_dir = "/net/lem/data1/grasser/models/SonoraBobcat/"
elif getpass.getuser() == "natalie": # when testing from my laptop
    path_tables = '/home/natalie/fastchem_tables'
            
def plot_spectrum_inset(retr_obj,inset=True,fs=10,leg_fs=None,
                        plot_veiling=False, rescale=False,**kwargs):

    wave=retr_obj.data_wave
    flux=retr_obj.data_flux
    err=retr_obj.data_err
    flux_m=np.copy(retr_obj.model_flux)
    mask = np.isfinite(flux)
    flux_m[~mask] = np.nan

    suffix=''
    low=[]
    up=[]
    leg_fs=fs if leg_fs==None else leg_fs

    # normalized differently, add continuum back for plotting
    # wont work like this, need to save continuum from original spectrum
    if 'ROXs12' in retr_obj.target.name and rescale: 
        if rescale:
            model = pRT_spectrum(retr_obj)
            m_flux = model.make_spectrum()
            continuum = model.model_continuum
            scale = [flux, err, flux_m]
            flux, err, flux_m = [var*continuum for var in scale]

    wl_unit = retr_obj.parameters.params['wavelength_unit'].to_string('latex')
    pm_xlim = 0.0 # in um
    s2 = retr_obj.params_dict['s2']
    if retr_obj.instrument=='CRIRES':
        pm_xlim = 10 # in nm
        if retr_obj.target.name=='test_ROXs12B':
            s2*=1.5       

    if plot_veiling: # show spectrum without veiling for comparison
        retr_obj2 = copy.deepcopy(retr_obj) 
        #retr_obj2.parameters.params.pop('log_k_rk')
        if 'log_k_rk' in retr_obj2.parameters.params:
            retr_obj2.parameters.params['log_k_rk']=-6
            retr_obj2.parameters.params['d_rk']=0
        elif 'T_disk' in retr_obj2.parameters.params:
            retr_obj2.parameters.params['phi_disk']=0
        model_wo_veiling = pRT_spectrum(retr_obj2).make_spectrum()
        noveil_c = 'm'
        suffix='_noveil'

    if 'ax' in kwargs:
        ax=kwargs.get('ax')
    else:
        fig,ax=plt.subplots(2,1,figsize=(10,2.5),dpi=200,gridspec_kw={'height_ratios':[2,0.7]})

    for part in range(retr_obj.n_parts):
        if np.isnan(flux[part]).all():
            #flux_m[part]/=np.nanmedian(flux_m[part])
            flux_m[part]*=np.nan # fixed ylim issue for A
            continue
        # add error for scale
        order = slice(part,part+3)
        if part%3==0 and np.nansum(flux[order])!=0 and retr_obj.instrument=='CRIRES': # skip empty orders 
            errmean=np.nanmean(err[order]*retr_obj.params_dict['s2'][order].reshape(3,1))
            errmean=np.nanmean(err[order]*retr_obj.params_dict['s2'][order].reshape(3,1))
            ax[1].errorbar(np.min(wave[order])-3, 0, yerr=errmean, ecolor='k',elinewidth=1, capsize=2)
            std_mean = np.nanstd(flux[order]-flux_m[order])
            ax[1].errorbar(np.min(wave[order])-6, 0, yerr=std_mean, ecolor=retr_obj.color,elinewidth=1, capsize=2)

        if retr_obj.parameters.params['data_as_points']==False: 
            lw=0.8
            ax[0].plot(wave[part],flux[part],lw=lw,alpha=1,c='k',label='Data')
            ax[1].plot(wave[part],flux[part]-flux_m[part],lw=lw,c=retr_obj.color,label='residuals')
            lower=flux[part]-err[part]*retr_obj.params_dict['s2'][part]
            upper=flux[part]+err[part]*retr_obj.params_dict['s2'][part]
            low.append(np.nanmin(lower))
            up.append(np.nanmax(upper))
            ax[0].fill_between(wave[part],lower,upper,color='k',alpha=0.15,label=f'1 $\sigma$')
                    
        elif retr_obj.parameters.params['data_as_points']: 
            size=2
            lw=1.5
            elw=0.6          
            ax[0].errorbar(wave[part],flux[part],yerr=err,fmt='o',markersize=size,elinewidth=elw,c='k',label='Data')
            ax[1].scatter(wave[part],flux[part]-flux_m[part],s=size,c=retr_obj.color)

        #if np.isnan(flux[part]).all():# scale will be weird, set manually
            #flux_m[part] = scale_between(ymin,ymax,flux_m[part])
        if plot_veiling:
            ax[0].plot(wave[part],model_wo_veiling[part],lw=lw,alpha=0.8,c=noveil_c,label='r$_k$=0')
            if 'd_rk' in retr_obj.params_dict:
                rk = np.round(retr_obj.params_dict('d_rk'),decimals=2)
            elif 'T_disk' in retr_obj.params_dict:
                rk = np.round(retr_obj.params_dict('phi_disk'),decimals=2)            
            model_label = f'r$_k$={rk}'
        else:
            model_label = 'Bestfit'
        ax[0].plot(wave[part],flux_m[part],lw=lw,alpha=0.8,c=retr_obj.color,label=model_label)
        if part==0:
            if retr_obj.instrument=='CRIRES':
                lines = [Line2D([0], [0], color='k',linewidth=2,label='Data'),
                    #mpatches.Patch(color='k',alpha=0.15,label='1$\sigma$'),
                    Line2D([0], [0], color=retr_obj.color, linewidth=2,label='Bestfit')]
                ax[0].legend(handles=lines,fontsize=leg_fs) # to only have it once
            elif retr_obj.instrument=='LIFE':
                ax[0].legend(fontsize=leg_fs)
        ax[1].plot([np.min(wave[part]),np.max(wave[part])],[0,0],lw=0.8,alpha=1,c='k')

    ax[0].set_ylabel(retr_obj.ylabel,fontsize=fs)
    ax[1].set_ylabel('Res.',fontsize=fs)
    ax[0].set_xlim(np.min(wave)-pm_xlim,np.max(wave)+pm_xlim)
    ax[0].set_ylim(np.nanmin(np.array([flux,flux_m])),np.nanmax(np.array([flux,flux_m])))
    ax[1].set_xlim(np.min(wave)-pm_xlim,np.max(wave)+pm_xlim)
    tick_spacing=10
    ax[1].xaxis.set_minor_locator(ticker.MultipleLocator(tick_spacing))
    ax[0].tick_params(labelsize=fs)
    ax[1].tick_params(labelsize=fs)
    
    if np.isnan(flux[:3]).all():
        ax[0].set_xlim(np.min(wave[3:])-pm_xlim,np.max(wave)+pm_xlim)
        ax[1].set_xlim(np.min(wave[3:])-pm_xlim,np.max(wave)+pm_xlim)

    if inset==True and retr_obj.instrument=='CRIRES':
        suffix='_inset'
        parts = slice(15,18) # 5th order
        axins = ax[0].inset_axes([0,-1.3,1,0.75]) # left, bottom, width, height
        for p in [15,16,17]:
            lower=flux[p]-err[p]*s2[p]#[:, np.newaxis]
            upper=flux[p]+err[p]*s2[p]#[:, np.newaxis]
            axins.fill_between(wave[p],lower,upper,color='k',alpha=0.15,label=f'1 $\sigma$')
            axins.plot(wave[p],flux[p],lw=0.8,c='k')
            if plot_veiling:
                axins.plot(wave[p],model_wo_veiling[p],lw=lw,alpha=0.8,c=noveil_c,label='w/o veiling')
            axins.plot(wave[p],flux_m[p],lw=0.8,c=retr_obj.color,alpha=0.8)
        x1, x2 = np.min(wave[parts]),np.max(wave[parts])
        axins.set_xlim(x1, x2)
        box,lines=ax[0].indicate_inset_zoom(axins,edgecolor="black",alpha=0.2,lw=0.8,zorder=1e3)
        axins.set_ylabel(retr_obj.ylabel,fontsize=fs)
        axins.tick_params(labelsize=fs)
        ax[1].set_facecolor('none') # to avoid hiding lines
        ax[0].set_xticks([])
        
        axins2 = axins.inset_axes([0,-0.3,1,0.3])
        axins2.plot(wave[parts].flatten(),flux[parts].flatten()-flux_m[parts].flatten(),lw=0.8,c=retr_obj.color)
        axins2.plot([np.min(wave[parts]),np.max(wave[parts])],[0,0],lw=0.8,alpha=1,c='k')
        axins2.set_xlim(x1, x2)
        axins2.set_xlabel(rf'Wavelength [{wl_unit}]',fontsize=fs)
        axins2.set_ylabel('Res.',fontsize=fs)
        tick_spacing=1
        axins2.xaxis.set_minor_locator(ticker.MultipleLocator(tick_spacing))
        axins2.tick_params(labelsize=fs)
    else:
        ax[1].set_xlabel(rf'Wavelength [{wl_unit}]',fontsize=fs) # if no inset

    plt.subplots_adjust(wspace=0, hspace=0)
    if 'ax' not in kwargs:
        name = f'bestfit{suffix}' if retr_obj.callback_label=='final_' else f'{retr_obj.callback_label}bestfit{suffix}'
        fig.savefig(f'{retr_obj.output_dir}/{name}.pdf', bbox_inches='tight')
        plt.close()

def plot_spectrum_split(retr_obj,show_opacities=None,plot_components=False,
                        plot_cloud=False, plot_veiling=False,lw=0.9,
                        alpha_model=1,rescale=False):

    # function only for CRIRES spectra anyway
    crires_shape = (retr_obj.n_orders,retr_obj.n_dets,retr_obj.n_pixels)

    if show_opacities!=None: # overplot species to check features
        opacities={}
        species_string = ''

        if isinstance(show_opacities, list)==False:
            show_opacities=list(show_opacities)

        for spec in show_opacities: 
            species_string+=str(spec)
            opa_orders=[]
            for order in range(7):
                wlen_range=np.array([np.min(retr_obj.target.K2166[order]),np.max(retr_obj.target.K2166[order])])*1e-3 # nm to microns
                atm = Radtrans(line_species=[retr_obj.species_info.loc[spec,retr_obj.pRT_key]],
                                    rayleigh_species = [],
                                    continuum_opacities = [],
                                    wlen_bords_micron=wlen_range, 
                                    mode='lbl',
                                    lbl_opacity_sampling=3) # take every nth point (=3 in deRegt+2024)
                
                wave_cm, opas = atm.get_opa(np.array([retr_obj.params_dict['T_maxcont']]).reshape(1))
                opa = opas[retr_obj.species_info.loc[spec,retr_obj.pRT_key]].flatten()

                wl_shifted= wave_cm*1e7*(1.0+(retr_obj.params_dict['rv']-retr_obj.target.vbary)/const.c.to('km/s').value)
                waves_even = np.linspace(np.min( wave_cm*1e7), np.max( wave_cm*1e7), np.array(wave_cm).size) # wavelength array has to be regularly spaced
                opa = np.interp(waves_even, wl_shifted, opa)

                parts = slice(order*3,order*3+3)
                opa_interp = np.interp(retr_obj.data_wave[parts].flatten(), waves_even, opa)
                opa_orders.append(opa_interp)
            opacities[spec] = np.array(opa_orders).reshape(crires_shape)
        retr_obj.opacities=opacities

    if rescale:
        model = pRT_spectrum(retr_obj)
        m_flux = model.make_spectrum()
        continuum = model.model_continuum.reshape(crires_shape)

    retr=retr_obj
    residuals=(retr.data_flux-retr.model_flux).reshape(crires_shape)
    to_reshape= [np.copy(retr_obj.data_flux), retr_obj.data_err, retr_obj.data_wave, np.copy(retr_obj.model_flux)]
    data_flux, data_err, data_wave, model_flux = [var.reshape(crires_shape) for var in to_reshape]
    s2 = retr.params_dict['s2'].reshape((crires_shape[:-1]))

    if rescale:
        scale_arrs = [data_flux, data_err, model_flux]
        data_flux, data_err, model_flux = [var*continuum for var in scale_arrs]

    n_sub = 20
    n_gs = 6
    plot_orders = np.linspace(0,6,7,dtype=int)
    if np.isnan(data_flux[0].flatten()).all(): # often full of tellurics, remove subplot if empty
         n_sub-=3
         n_gs = 5
         plot_orders = np.linspace(1,6,6,dtype=int)

    figsize=(10,13)
    gridspec_kw={'height_ratios':[2,0.9,0.57]*n_gs+[2,0.9]}
    #if 'log_k_rk' in retr.params_dict:
        #rk_func = lambda x: 10**retr.params_dict['log_k_rk']*np.array(x) + retr.params_dict['d_rk']
    if plot_components==True:
        primary_flx= retr.model_object.primary_broadened
        secondary_flx = retr.model_object.secondary_flux
        # was not normalized correctly for some reason? norm all in same way
        #for i in range(retr_obj.n_orders):
            #for j in range(retr_obj.n_dets):
                #if np.isnan(data_flux[i,j]).all()==False:
                    #data_flux[i,j]/=np.nanmedian(data_flux[i,j])
                    
        phi_sec = retr_obj.params_dict['phi_secondary']
        print('Phi star=',1-phi_sec)
        print('Phi BD  =',phi_sec)
        #figsize=(9,15)
        gridspec_kw={'height_ratios':[2,0.4,0.57]*n_gs+[2,0.4]}

    if plot_veiling: # show spectrum without veiling for comparison
        retr_obj2 = copy.deepcopy(retr_obj) 
        mask = np.isfinite(data_flux)
        if 'log_k_rk' in retr_obj2.parameters.params:
            retr_obj2.parameters.params['log_k_rk']=-6
            retr_obj2.parameters.params['d_rk']=0
        elif 'T_disk' in retr_obj2.parameters.params:
            retr_obj2.parameters.params['phi_disk']=0
        model_wo_veiling = pRT_spectrum(retr_obj2).make_spectrum().reshape(crires_shape)
        model_wo_veiling[~mask] = np.nan
        noveil_c = 'm'

    if plot_cloud: # show spectrum without cloud for comparison
        retr_obj2 = copy.deepcopy(retr_obj) 
        mask = np.isfinite(data_flux)
        retr_obj2.cloud_mode=None
        model_wo_cloud = pRT_spectrum(retr_obj2).make_spectrum().reshape(crires_shape)
        model_wo_cloud[~mask] = np.nan
        nocloud_c = 'm'

    fig,ax=plt.subplots(n_sub,1,figsize=figsize,dpi=200,gridspec_kw=gridspec_kw)
    x=0
    
    for idx,order in enumerate(plot_orders):
        ax1=ax[x]
        ax2=ax[x+1]
        min_array, max_array = [],[]
        opa_orders = []
        
        if x!=(n_sub-2): # last ax cannot be spacer, or xlabel also invisible
            ax3=ax[x+2] #for spacing
        for det in range(3):
            mask = np.isnan(data_flux[order,det])
            if retr_obj.name=='ROXs12A' and (order,det)==(3,0):
                wl = data_wave[order,det]
                wl_mask = (wl>2152.57)&(wl<2152.77)
                model_flux[order,det][wl_mask] = np.nan
                data_flux[order,det][wl_mask] = np.nan
                residuals[order,det][wl_mask] = np.nan

            ax1.plot(data_wave[order,det],data_flux[order,det],lw=lw,alpha=1,c='k',label='data')
            if plot_veiling:
                if np.isnan(data_flux[order,det]).all()==False:
                    ax1.plot(data_wave[order,det],model_wo_veiling[order,det],lw=lw,c=noveil_c,label='r$_k$=0')
                    noveil_label='r$_k$=0'
                    if 'd_rk' in retr_obj.params_dict:
                        rk = np.round(retr_obj.params_dict['d_rk'],decimals=2)
                    elif 'phi_disk' in retr_obj.params_dict:
                        wl = data_wave.flatten()
                        wl_mid = np.median(data_wave)
                        T_disk = retr_obj.params_dict['T_disk']
                        phi_disk = retr_obj.params_dict['phi_disk']
                        BB_0 = planck_lambda_um(T_disk,wl_mid*1e-3)
                        BB_veil = planck_lambda_um(T_disk,wl*1e-3)
                        rk = np.round(phi_disk,decimals=2)
                    model_label = f'r$_k$={rk}'
            elif plot_cloud:
                if np.isnan(data_flux[order,det]).all()==False:
                    ax1.plot(data_wave[order,det],model_wo_cloud[order,det],lw=lw,c=nocloud_c,label='no cloud')
                    nocloud_label='no cloud'
                    model_label = 'cloud'
            else:
                model_label = 'Bestfit'
            ax1.plot(data_wave[order,det],model_flux[order,det],lw=lw,alpha=alpha_model,c=retr_obj.color,label='model')
            ax1.set_xlim(np.nanmin(data_wave[order])-1,np.nanmax(data_wave[order])+1)
            if plot_components==True:
                prim_c='orange'
                sec_c='dodgerblue'
                prim_flx= primary_flx[order,det]
                sec_flx = secondary_flx[order,det]
                
                sec_flx[mask] = np.nan
                ax1.plot(data_wave[order,det], prim_flx, label='A',lw=0.8, c=prim_c)
                ax1.plot(data_wave[order,det], sec_flx,lw=0.8, label='B', c=sec_c)
                
                if np.isfinite(prim_flx).any() and np.isfinite(sec_flx).any():
                    min_array.append(np.nanmin([prim_flx,sec_flx]))
                    max_array.append(np.nanmax([prim_flx,sec_flx]))
                #if np.isnan(data_flux[order,det]).all(): # then model is only secondary flux
                    #ax1.plot(data_wave[order,det], model_sec_flx*avg_B,lw=0.8, c=sec_c)
                    #min_array.append(np.nanmin(model_sec_flx*avg_B))
                    #max_array.append(np.nanmax(model_sec_flx*avg_B))
            #if 'log_k_rk' in retr.params_dict:
                #ax1.plot(data_wave[order,det], rk_func(data_wave[order,det]),lw=0.8, label='D', c='yellowgreen')

            if show_opacities==None: # else show opacities
                ax2.plot(data_wave[order,det],residuals[order,det],lw=0.8,alpha=1,c=retr_obj.color,label='residuals')
                ax2.set_xlim(np.nanmin(data_wave[order])-1,np.nanmax(data_wave[order])+1)

            # add error for scale
            #if retr.callback_label=='final_':
            if retr_obj.target.name=='ROXs12A':
                s2[order,det]*=1.7
            lower=data_flux[order,det]-data_err[order,det]*s2[order,det]
            upper=data_flux[order,det]+data_err[order,det]*s2[order,det]
            ax1.fill_between(data_wave[order,det],lower,upper,color='k',alpha=0.2,label=f'1 $\sigma$')
            err = data_err[order,det] if np.all(np.isnan(data_err[order,det]))==False else 0
            errmean=np.nanmean(err*s2[order,det])
            
            if show_opacities==None:
                #print(order, det)
                if np.nansum(data_flux[order,det])!=0: # skip empty orders
                    ax2.errorbar(np.min(data_wave[order,det])-0.25, 0, yerr=errmean, 
                                ecolor='k', elinewidth=1, capsize=2)
                    std_mean = np.nanstd(residuals[order,det])
                    ax2.errorbar(np.min(data_wave[order,det])-0.4, 0, yerr=std_mean, 
                                ecolor=retr_obj.color, elinewidth=1, capsize=2)
                    #for i in range(3):
                    #    i+=1
                    #    ax2.fill_between(data_wave[order,det], -i*errmean, y2=i*errmean, color='k',alpha=0.05)        
            
            if x==0 and det==0:
                ncol=2
                lines = [Line2D([0], [0], color='k',linewidth=2,label='Data'),
                        #mpatches.Patch(color='k',alpha=0.15,label='1$\sigma$'),
                        Line2D([0], [0], color=retr.color, linewidth=2,label=model_label)]
                if plot_components==True:
                    lines.append(Line2D([0], [0], color=prim_c,linewidth=2,label='A'))
                    lines.append(Line2D([0], [0], color=sec_c,linewidth=2,label='B'))
                    ncol+=2
                if plot_veiling:
                    lines.append(Line2D([0], [0], color=noveil_c,linewidth=2,label=noveil_label))
                    ncol+=1
                elif plot_cloud:
                    lines.append(Line2D([0], [0], color=nocloud_c,linewidth=2,label=nocloud_label))
                    ncol+=1
                leg=ax1.legend(handles=lines,fontsize=12,ncol=ncol,bbox_to_anchor=(0.47,1.4),loc='upper center')
                leg.get_frame().set_linewidth(0.0)

            if show_opacities==None:
                ax2.plot([np.min(data_wave[order,det]),np.max(data_wave[order,det])],[0,0],lw=0.8,c='k')
            else:
                opacities_legend=[]
                for i,species in enumerate(show_opacities):
                    opas=opacities[species]
                    ymax=np.nanmax([np.nanmax(model_flux[order]),np.max(data_flux[order])])#*0.9
                    ymin=np.nanmin([np.nanmin(model_flux[order]),np.min(data_flux[order])])#*1.1
                    if f'log_{species}' in retr_obj.params_dict:
                        opa=opas[order,det]*10**retr_obj.params_dict[f'log_{species}']
                        y_cutoff = 1e-8
                        tick_spacing=1
                        ax2.xaxis.set_minor_locator(ticker.MultipleLocator(tick_spacing))
                    else: 
                        opa=opas[order,det]
                        y_cutoff = 1e-1 if np.max(opa)>1e-1 else np.min(opa)
                    for a in [ax1,ax2]:
                        a.xaxis.set_minor_locator(ticker.MultipleLocator(1))
                        a.grid(which='minor', axis='x', linestyle=':', linewidth=0.6,alpha=0.9)
                        a.grid(which='major', axis='x', linestyle='-', linewidth=0.6,alpha=1)
                        a.tick_params(axis='x', which='minor', bottom=False)

                    opa_scaled=scale_between(ymax,ymin,opa)
                    opa_orders.append(opa)
                    cl = retr_obj.species_info.loc[species,'color']
                    ax2.plot(data_wave[order,det],opa,lw=0.8,c=cl)
                    #ax1.plot(data_wave[order,det],opa_scaled,lw=0.8,c=opacities_colors[i])
                    opacities_legend.append(Line2D([0],[0],color=cl,
                                                   linewidth=2,linestyle='-',label=species))
                if idx==0 and det==0:
                    ax2.legend(handles=opacities_legend,fontsize=10,ncol=len(show_opacities))

        if np.isnan(data_flux[order]).all():
            #if 'log_k_rk' in retr.params_dict:
                #min_array.append(np.nanmin([data_flux,model_flux,disk_flux]))
                #max_array.append(np.nanmax([data_flux,model_flux,disk_flux]))
            #else:
            min_array.append(np.nanmin([data_flux,model_flux]))
            max_array.append(np.nanmax([data_flux,model_flux]))
        else:
            min_array.append(np.nanmin([data_flux[order],model_flux[order]]))
            max_array.append(np.nanmax([data_flux[order],model_flux[order]]))

        #min1=np.nanmin(np.array([retr.data_flux[order]-retr.data_err[order],retr.model_flux[order]]))
        #max1=np.nanmax(np.array([retr.data_flux[order]+retr.data_err[order],retr.model_flux[order]]))
        min1=np.nanmin(np.array(min_array))
        max1=np.nanmax(np.array(max_array))
        ax1.set_ylim(min1,max1)
        if np.nansum(residuals[order])!=0 and show_opacities==None:
            ax2.set_ylim(np.nanmin(residuals[order]),np.nanmax(residuals[order]))
        elif show_opacities==None:# if empty order full of nans
            ax2.set_ylim(-0.1,0.1)
        else:
            ax2.set_xlim(np.min(data_wave[order]),np.max(data_wave[order]))
            ax1.set_xlim(np.min(data_wave[order]),np.max(data_wave[order]))
            ax2.set_yscale('log')
            if np.min(opa_orders)<y_cutoff:
                ax2.set_ylim(y_cutoff,np.max(opa_orders))
        ax1.tick_params(labelbottom=False)  # don't put tick labels at bottom
        ax1.tick_params(axis="both")
        ax2.tick_params(axis="both")
        ax1.set_ylabel('Normalized Flux')
        if show_opacities==None:
            ax2.set_ylabel('Res.')
        else:
            ax2.set_ylabel('[g/cm$^2$]')
        ax1.tick_params(labelsize=9)
        ax2.tick_params(labelsize=9)
        if x!=(n_sub-1):
            ax3.set_visible(False) # invisible for spacing
        x+=3
    ax[(n_sub-2)].set_xlabel('Wavelength [nm]')
    fig.tight_layout()
    plt.subplots_adjust(wspace=0,hspace=0)
    if show_opacities!=None:
        name = f'bestfit_{species_string}'
    elif plot_veiling:
        name = 'bestfit_veiling' if retr_obj.callback_label=='final_' else f'{retr_obj.callback_label}bestfit_veiling'
    elif plot_cloud:
        name = 'bestfit_cloud' if retr_obj.callback_label=='final_' else f'{retr_obj.callback_label}bestfit_cloud'
    elif plot_components==False:
        name = 'bestfit_split' if retr_obj.callback_label=='final_' else f'{retr_obj.callback_label}bestfit_split'
    else:
        name = 'bestfit_components' if retr_obj.callback_label=='final_' else f'{retr_obj.callback_label}components'
    fig.savefig(f'{retr_obj.output_dir}/{name}.pdf')
    plt.close()

def photosphere_gradient(cont,pressure,ax,col='gray',xmin=0,xmax=1,
                            show_curve=False,fine_grid=False):
    cont_norm = abs(cont) / cont.max() *0.25
    pres = pressure
    if fine_grid:
        p_fine = np.logspace(np.min(pres),np.max(pres), 500)
        cont_fine = np.interp(np.log10(p_fine), np.log10(pres), cont_norm)
        pres = p_fine
        cont_norm = cont_fine           
    for i in range(len(pres)-1):
        ax.fill_betweenx([pres[i], pres[i+1]], xmin, xmax,
                        color=col,alpha=cont_norm[i],
                        linewidth=0, edgecolor='none')
    if show_curve:
        xmax0 = 1600
        xmin0 = 0
        cont_norm_small = cont / cont.max() * (xmax0-xmin0)+xmin0
        ax.plot(cont_norm_small,pressure,color=col,linestyle='dotted',alpha=0.8)
    return

def get_SPHINX_PT_envelope(Teff_range=None, logg_range = None, 
                           logZ_range = None, CtoO_range = None,
                           ax=None, plot_individual=False, 
                           fill_env=False, comparison_pt=[]):
    
    def in_range(value, r):
        """Return True if r is None or value is within r."""
        if r is None:
            return True
        return (r[0] <= value <= r[1])
    
    def parse_params(fname):
        # Example filename: Teff_3900.0_logg_5.5_logZ_+0.75_CtoO_0.9.txt
        pattern = (r"Teff_(?P<Teff>\d+(?:\.\d+)?)"
                    r"_logg_(?P<logg>\d+(?:\.\d+)?)"
                    r"_logZ_(?P<logZ>[+-]?\d+(?:\.\d+)?)"
                    r"_CtoO_(?P<CtoO>\d+(?:\.\d+)?)\.txt$")
        match = re.search(pattern, fname)
        if not match:
            return None

        return {k: float(v) for k, v in match.groupdict().items()}
    
    all_Teff, all_logg, all_logZ, all_CtoO, selected_profiles = [],[],[],[],[]

    for path in glob.glob(sphinx_pt_dir):
        params = parse_params(path)
        if params is None:
            continue

        # Selection:
        if (in_range(params["Teff"], Teff_range) and
            in_range(params["logg"], logg_range) and
            in_range(params["logZ"], logZ_range) and
            in_range(params["CtoO"], CtoO_range)):

            # Store parameters
            all_Teff.append(params["Teff"])
            all_logg.append(params["logg"])
            all_logZ.append(params["logZ"])
            all_CtoO.append(params["CtoO"])

            # Load data
            data = np.loadtxt(path, comments="#")
            T = data[:,0]
            P = data[:,1]
            selected_profiles.append((T, P))

    # Build a common pressure grid (log-spaced)
    all_P = np.hstack([P[P>0] for (_, P) in selected_profiles])
    P_grid = np.logspace(np.log10(all_P.min()), np.log10(all_P.max()), 200)

    T_interp = []
    for T, P in selected_profiles:
        sort_idx = np.argsort(P)
        P_sorted = P[sort_idx]
        T_sorted = T[sort_idx]
        # Interpolate, using np.interp which assumes ascending P_sorted
        T_interp.append(np.interp(P_grid, P_sorted, T_sorted))

    T_interp = np.array(T_interp)
    T_min = np.nanmin(T_interp, axis=0)  # safer: ignore NaNs
    T_max = np.nanmax(T_interp, axis=0)

    if ax is not None: # plot

        def effective_range(user_range, values):
            """Return user_range if given, otherwise (min, max) of values."""
            if user_range is not None:
                return user_range
            return (min(values), max(values))
        
        Teff_eff = effective_range(Teff_range, all_Teff)
        logg_eff = effective_range(logg_range, all_logg)
        logZ_eff = effective_range(logZ_range, all_logZ)
        CtoO_eff = effective_range(CtoO_range, all_CtoO)

        t_label   = f"T: {Teff_eff[0]:g}-{Teff_eff[1]:g} K"
        g_label   = f"log$g$: {logg_eff[0]:g}-{logg_eff[1]:g}"
        feh_label = f"[M/H]: {logZ_eff[0]:g}-{logZ_eff[1]:g}"
        co_label  = f"C/O: {CtoO_eff[0]:g}-{CtoO_eff[1]:g}"
        env_label = f"SPHINX models\n{t_label}\n{g_label}\n{feh_label}\n{co_label}"

        if plot_individual:
            for T, P in selected_profiles:
                ax.plot(T, P, alpha=0.1, color='k')
        if fill_env:
            ax.fill_betweenx(P_grid, T_min, T_max, alpha=0.15)
            comparison_pt.append(Line2D([0], [0], color='k', alpha=0.15, linewidth=10, linestyle='solid',label=env_label))
        else:
            ax.plot(P_grid, T_min, linestyle='dashdot',c='k',alpha=0.3)
            ax.plot(P_grid, T_max, linestyle='dashdot',c='k',alpha=0.3)
            comparison_pt.append(Line2D([0], [0], color='k', alpha=0.3, linewidth=2, linestyle='dashdot',label=env_label))
        return comparison_pt
    else:
        return P_grid, T_min, T_max

def get_sonora_PT_envelope(temp_range=None,
                            logg_range=None,
                            feh_range=None,
                            co_range=None,
                            ax=None,
                            plot_individual=False,
                            fill_env=False,
                            comparison_pt=[]):
    """
    Plot PT envelope of Sonora Bobcat models.
    temp_range, logg_range, feh_range, co_range: tuples (min, max)
    If None, use full range.
    """

    CO_SOLAR = 0.55  # adjust if you use a different solar C/O

    def in_range(val, r):
        return True if r is None else (r[0] <= val <= r[1])

    # Parse filename
    def parse_params(fname):
        # e.g. t200g1000nc_m-0.5_co0.5.dat  or t200g1000nc_m-0.5.dat
        pattern = r"t(?P<T>\d+)" \
                  r"g(?P<g>\d+)" \
                  r"nc_m(?P<feh>[+-]?\d+(?:\.\d+)?)" \
                  r"(?:_co(?P<co>[+-]?\d+(?:\.\d+)?))?\.dat$"
        match = re.search(pattern, fname)
        if not match:
            return None
        d = match.groupdict()
        Teff = float(d["T"])
        g_mks = float(d["g"])
        logg = np.log10(g_mks * 100)  # convert m/s² to cm/s²
        feh = float(d["feh"])
        co_rel = float(d["co"]) if d["co"] is not None else 1.0
        co_abs = co_rel * CO_SOLAR
        return {"Teff": Teff, "logg": logg, "feh": feh, "CtoO": co_abs}

    selected_profiles = []
    all_Teff, all_logg, all_feh, all_CtoO = [], [], [], []

    all_possible_CtoO = []
    for folder in glob.glob(f"{sonora_dir}/structures*"):
        for fpath in glob.glob(f"{folder}/*.dat"):
            params = parse_params(fpath.split("/")[-1])
            if params is None:
                continue
            all_possible_CtoO.append(params["CtoO"])

            if in_range(params["Teff"], temp_range) and \
                in_range(params["logg"], logg_range) and \
                in_range(params["feh"], feh_range) and \
                (co_range is None or in_range(params["CtoO"], co_range)):

                all_Teff.append(params["Teff"])
                all_logg.append(params["logg"])
                all_feh.append(params["feh"])
                all_CtoO.append(params["CtoO"])
                data = np.loadtxt(fpath, comments="#", skiprows=1)
                P = data[:,1]  # bar
                T = data[:,2]  # K
                selected_profiles.append((T, P))

    if not selected_profiles:
        print("No profiles found for the selected ranges!")
        return None

    # Build common pressure grid
    all_P = np.hstack([P[P>0] for (_, P) in selected_profiles])
    P_grid = np.logspace(np.log10(all_P.min()), np.log10(all_P.max()), 200)

    T_interp = []
    for T, P in selected_profiles:
        sort_idx = np.argsort(P)
        P_sorted = P[sort_idx]
        T_sorted = T[sort_idx]
        T_interp.append(np.interp(P_grid, P_sorted, T_sorted))

    T_interp = np.array(T_interp)
    T_min = np.nanmin(T_interp, axis=0)
    T_max = np.nanmax(T_interp, axis=0)

    if ax is not None:
        def effective_range(user_range, values, full_values=None):
            if user_range is not None:
                return user_range
            if full_values is not None:
                return (min(full_values), max(full_values))
            return (min(values), max(values))
        
        Teff_eff = effective_range(temp_range, all_Teff)
        logg_eff = effective_range(logg_range, all_logg)
        logZ_eff = effective_range(feh_range, all_feh)
        CtoO_eff = effective_range(co_range, all_CtoO, full_values=all_possible_CtoO)

        t_label   = f"T: {Teff_eff[0]:g}-{Teff_eff[1]:g} K"
        g_label   = f"log$g$: {logg_eff[0]:g}-{logg_eff[1]:g}"
        feh_label = f"[M/H]: {logZ_eff[0]:g}-{logZ_eff[1]:g}"
        co_label  = f"C/O: {CtoO_eff[0]:g}-{CtoO_eff[1]:g}"
        env_label = f"Sonora Bobcat\n{t_label}\n{g_label}\n{feh_label}\n{co_label}"
        
        if plot_individual:
            for T, P in selected_profiles:
                ax.plot(T, P, alpha=0.1, color='k')
        if fill_env:
            ax.fill_betweenx(P_grid, T_min, T_max, alpha=0.15)
            comparison_pt.append(Line2D([0], [0], color='k', alpha=0.15, linewidth=10, linestyle='solid',label=env_label))
        else:
            ax.plot(T_min, P_grid, linestyle='dashdot',c='k',alpha=0.3)
            ax.plot(T_max, P_grid, linestyle='dashdot',c='k',alpha=0.3)
            comparison_pt.append(Line2D([0], [0], color='k', alpha=0.3, linewidth=2, linestyle='dashdot',label=env_label))

        return comparison_pt
    else:
        return P_grid, T_min, T_max

def plot_pt(retr_obj,fs=12,figsize=5,comp_pt=True,show_cond=False,
            show_contr=True,contr_as_curve=False,leg_fs=12,
            save_jpg=False,temp_lim=None,get_xmin_xmax=False,
            sb_temps=None, sb_logg=None, sb_feh=None, sb_co=None,# sonora bobcat envelope
            sx_temps=None, sx_logg=None, sx_feh=None, sx_co=None,# sphinx
            temp_padding=10,colors=[],show_legend=True,**kwargs):

    if colors==[]:
        colors=[retr_obj.color]

    leg_fs = fs*0.9 if retr_obj.target.name=='test_ROXs12B' else leg_fs
    legend_labels=kwargs.get('legend_labels',None)
    if 'retr_obj2' in kwargs: # compare two retrs
        retr_obj2=kwargs.get('retr_obj2')
        colors.append(retr_obj2.color)
    else:
        retr_obj2=retr_obj # to make later if statements work

    if show_cond: # show condensation curve
        if retr_obj.chemistry in ['equchem','quequchem']:
            C_O = retr_obj.model_object.params['C/O']
            Fe_H = retr_obj.model_object.params['Fe/H']
        elif retr_obj.chemistry =='flexequ':
            C_O = retr_obj.params_dict['C/O']
            Fe_H = retr_obj.params_dict['C/H']
        elif retr_obj.chemistry=='freechem':
            C_O = retr_obj.model_object.CO
            Fe_H = retr_obj.model_object.FeH   

    if 'ax' in kwargs:
        ax=kwargs.get('ax')
    else:
        fig,ax=plt.subplots(1,1,figsize=(figsize,figsize),dpi=200)
    #cloud_species = ['MgSiO3(c)', 'Fe(c)', 'KCl(c)', 'Na2S(c)']
    #cloud_labels=['MgSiO$_3$(c)', 'Fe(c)', 'KCl(c)', 'Na$_2$S(c)']
    #cs_colors=['gold','goldenrod','peru','sandybrown']
    cloud_species = ['MgSiO3(c)', 'Fe(c)']
    cloud_labels=['MgSiO$_3$(c)', 'Fe(c)']
    cs_colors=['goldenrod','sandybrown']

    # if pt profile and condensation curve don't intersect, clouds have no effect
    if retr_obj.target.name in ['2M0355','2M1425','test','test_corr','testsys'] and show_cond:
        for i,cs in enumerate(cloud_species):
            cs_key = cs[:-3]
            if cs_key == 'KCl':
                cs_key = cs_key.upper()
            P_cloud, T_cloud = getattr(cloud_cond, f'return_T_cond_{cs_key}')(Fe_H, C_O)
            pi=np.where((P_cloud>min(retr_obj.model_object.pressure))&(P_cloud<max(retr_obj.model_object.pressure)))[0]
            ax.plot(T_cloud[pi], P_cloud[pi], lw=1.3, label=cloud_labels[i], ls=':',c=cs_colors[i])
        # https://github.com/cphyc/matplotlib-label-lines
        labelLines(ax.get_lines(),align=False,fontsize=fs*0.8,drop_label=True)
    
    # compare with sonora bobcat T=1400K, logg=4.65 -> 10**(4.65)/100 =  446 m/s²
    #file=np.loadtxt('t1400g562nc_m0.0.dat')
    comparison_pt=[]

    # plot envelope of SPHINX PT profiles
    if any(v is not None for v in [sx_temps, sx_logg, sx_feh,sx_co]):
        comparison_pt = get_SPHINX_PT_envelope(Teff_range=sx_temps, logg_range = sx_logg, 
                            logZ_range = sx_feh, CtoO_range = sx_co,
                            ax=ax, plot_individual=False, comparison_pt=comparison_pt)

    if any(v is not None for v in [sb_temps, sb_logg, sb_feh,sb_co]):
        comparison_pt = get_sonora_PT_envelope(temp_range=sb_temps, logg_range = sb_logg, 
                            feh_range = sb_feh,co_range = sb_co,
                            ax=ax, plot_individual=False, comparison_pt=comparison_pt)
        
    teff = r'$T_{\mathrm{eff}}$'
    if retr_obj.target.name in ['test','test_corr','testsys']:
        from retrieval import Retrieval
        from parameters import Parameters
        from testspec import test_parameters
        test_par = Parameters({}, test_parameters)
        test_par.param_priors['log_l']=[-3,0]
        test_ret=Retrieval(target=retr_obj.target,parameters=test_par,
                            species_names=retr_obj.species_names, 
                           Nlive=retr_obj.Nlive,evtol=retr_obj.evtol,
                            chemistry='freechem',PT_type='PTgrad')
        test_ret.model_object=pRT_spectrum(test_ret)
        ax.plot(test_ret.model_object.temperature,test_ret.pressure,linestyle='dashdot',c='blueviolet',lw=2) 
        comparison_pt.append(Line2D([0], [0], color='blueviolet', linewidth=2, linestyle='dashdot',label='Input'))

    if 'comparison_PT_path' in retr_obj.parameters.params and comp_pt==True:
        if retr_obj.parameters.params['comparison_PT_path'] is not None:
            file=np.loadtxt(retr_obj.parameters.params['comparison_PT_path'])
            pres=file[:,0] # bar
            temp=file[:,1] # K
            ax.plot(temp,pres,linestyle=retr_obj.parameters.params['comparison_PT_linestyle'],
                        c=retr_obj.parameters.params['comparison_PT_color'],linewidth=2)
            comparison_pt.append(Line2D([0], [0], color='blueviolet', linewidth=2, 
                            linestyle='dashdot',label=retr_obj.parameters.params['comparison_PT_label']))

    if 'retr_obj2' in kwargs: # if compare two retrs, specify object name in legend
        object_label=f'{retr_obj.target.name} $P$-$T$'
        contr_label=f'{retr_obj.target.name} contr.'
    else:
        object_label='$P$-$T$ profile'
        contr_label='Photosphere'

    lines=[]
    # plot PT-profile + errors on retrieved temperatures
    def plot_temperature(retr_obj,ax,olabel,color): 

        # Compute quantiles for 1-3 sigma
        # temp_dist shape: (samples, n_layers)
        quantiles = np.array([
            np.percentile(retr_obj.temp_dist[:, i], [0.2, 2.3, 15.9, 50.0, 84.1, 97.7, 99.8])
            for i in range(retr_obj.temp_dist.shape[1])])
        
        # Median temperature profile
        median = quantiles[:, 3]
        
        # Plot sigma envelopes and median
        ax.fill_betweenx(retr_obj.pressure, quantiles[:,0], quantiles[:,-1], color=color, alpha=0.15)
        ax.fill_betweenx(retr_obj.pressure, quantiles[:,1], quantiles[:,-2], color=color, alpha=0.15)
        ax.fill_betweenx(retr_obj.pressure, quantiles[:,2], quantiles[:,-3], color=color, alpha=0.15)
        ax.plot(median, retr_obj.pressure, color=color, lw=2)
        
        # Scatter knots for PT_knot
        if retr_obj.PT_type == 'PTknot':
            t_keys = sorted([k for k in retr_obj.params_dict if re.fullmatch(r"T\d+", k)],
                            key=lambda x: int(x[1:]))[::-1]
            medians_knots = [retr_obj.params_dict[k] for k in t_keys]
            ax.scatter(medians_knots, 10**retr_obj.model_object.log_P_knots, color=color)
        
        # Set x-limits
        xmin = max(0, np.min(quantiles[:,0]))-temp_padding
        xmax = np.max(quantiles[:,-1])+temp_padding

        if xmin<=0:
            xmin=0
            ax.set_xlim(left=0)
        return xmin,xmax

    xmin,xmax=plot_temperature(retr_obj,ax,object_label,color=colors[0])
       
    if retr_obj.target.name=='2M0355':
        lines.append(Line2D([0], [0], color='cornflowerblue', linewidth=2, linestyle='dashdot',label='Zhang+2022'))
    
    if 'retr_obj2' in kwargs: # compare two retrs
        retr_obj2=kwargs.get('retr_obj2')
        object_label2=f'{retr_obj2.target.name} $P$-$T$'
        xmin2,xmax2=plot_temperature(retr_obj2,ax,object_label2,color=colors[1])
        xmin=np.nanmin([xmin,xmin2])
        xmax=np.nanmax([xmax,xmax2])
        if show_contr:
            if temp_lim is not None:
                xmin, xmax = temp_lim[0],temp_lim[1]
            summed_contr2=retr_obj2.summed_contribution
            prs = retr_obj2.model_object.pressure
            if contr_as_curve:
                contribution_plot2=summed_contr2/np.max(summed_contr2)*(xmax-xmin)+xmin
                ax.plot(contribution_plot2,prs,linestyle='dotted',
                        lw=1.5,alpha=0.8,color=colors[1])
                lines.append(Line2D([0], [0], color=colors[1], alpha=0.8,linewidth=1.5, 
                                    linestyle='--',label=f'{retr_obj2.target.name} contr.'))
            else:
                photosphere_gradient(summed_contr2,prs,ax,colors[1],xmin,xmax)
    
    if 'retr_obj3' in kwargs:
        retr_obj3=kwargs.get('retr_obj3')
        colors.append(retr_obj3.color)
        object_label3=f'{retr_obj3.target.name} retr'
        xmin3,xmax3=plot_temperature(retr_obj3,ax,object_label3,color=colors[2])
        xmin=np.nanmin([xmin,xmin2,xmin3])
        xmax=np.nanmax([xmax,xmax2,xmax3])
        if show_contr:
            summed_contr3=retr_obj3.summed_contribution
            contribution_plot3=summed_contr3/np.max(summed_contr3)*(xmax-xmin)+xmin
            ax.plot(contribution_plot3,retr_obj3.model_object.pressure,linestyle='dashed',
                    lw=1.5,alpha=0.8,color=colors[2])
            lines.append(Line2D([0], [0], color=colors[2], alpha=0.8,linewidth=1.5, 
                                linestyle='--',label=f'{retr_obj3.target.name} contr.'))

    if show_contr:
        summed_contr=retr_obj.summed_contribution 
        prs = retr_obj.model_object.pressure
        if temp_lim is not None:
            xmin, xmax = temp_lim[0],temp_lim[1]
        if contr_as_curve:
            contribution_plot=summed_contr/np.max(summed_contr)*(xmax-xmin)+xmin
            ax.plot(contribution_plot,prs,linestyle='dotted',
                    lw=1.5,alpha=0.8,color=colors[0])
            lines.append(Line2D([0], [0], color=colors[0], alpha=0.8,
                                linewidth=1.5, linestyle='dotted',label=contr_label))
            lines = lines[:1]+[lines[-1]]+lines[1:-1] # move to second position in legend instead of last
        else:
            photosphere_gradient(summed_contr,prs,ax,colors[0],xmin,xmax)
            #lines.append(Line2D([0], [0], color=retr_obj.color, alpha=0.3,
                                #linewidth=5, linestyle='solid',label=contr_label))
            tmax = retr_obj.params_dict['T_maxcont']
            pmax = 10**retr_obj.params_dict['log_P_maxcont']
            #ax.text(tmax*1.5,pmax*0.8, 'Emission', c=retr_obj.color)

        if retr_obj.target.name in ['Sorg1X','Sorg20X']:
            #psg_contr=np.genfromtxt(f'LIFE/{retr_obj.target.name}/{retr_obj.target.name}_contr.txt',skip_header=1,delimiter=' ')
            #psg_contr=psg_contr[:,1:] # exclude first column (wavelength)
            #psg_contr=np.sum(psg_contr,axis=0)[::-1] # sum over all wavelengths, change order
            #psg_contr =psg_contr/np.max(psg_contr)*(xmax-xmin)+xmin
            input_contr = np.load(f'LIFE/{retr_obj.target.name}/input_pRT_summed_contr.npy')
            prs = PSG_input(retr_obj.target.name).pressure
            if xmin<=0:
                xmin=0
                ax.set_xlim(left=0)
            if contr_as_curve:
                input_contr =input_contr/np.max(input_contr)*(xmax-xmin)+xmin
                ax.plot(input_contr,prs,linestyle='dotted',
                        lw=1.5,alpha=0.8,color=retr_obj.target.in_col)
                lines.append(Line2D([0], [0], color=retr_obj.target.in_col, linewidth=1.5, linestyle='dotted',label='Input phot.'))
            else:
                photosphere_gradient(input_contr,prs,ax,retr_obj.target.in_col,xmin,xmax)
    
    ax.set(xlabel='Temperature [K]', ylabel='Pressure [bar]',yscale='log',
        ylim=(np.nanmax(retr_obj.model_object.pressure),
        np.nanmin(retr_obj.model_object.pressure)),xlim=(xmin,xmax))

    if legend_labels!=None:
        lines=[]
        retr_objects=[retr_obj,retr_obj2]
        if 'retr_obj3' in kwargs:
            retr_objects.append(retr_obj3)
        for r,l in zip(retr_objects,legend_labels):
            lines.append(Line2D([0], [0], color=r.color,linewidth=2,linestyle='-',label=l))
            #if show_contr:
                #lines.append(Line2D([0], [0], color=r.color, alpha=0.8,
                                    #linewidth=1.5, linestyle='--',label=f'{l} emission'))
                
    if 'comparison_pt' in locals():
        lines.extend(comparison_pt)

    handlelength=2.5
    if retr_obj.target.name in ['Sorg1X','Sorg20X']:
        lines = lines[:1] + lines[-1:] + lines[1:-1]
        handlelength = 1.7

    if show_legend:
        ax.legend(handles=lines,fontsize=leg_fs,handlelength=handlelength,loc='upper right')
    ax.tick_params(labelsize=fs)
    ax.set_xlabel('Temperature [K]', fontsize=fs)
    ax.set_ylabel('Pressure [bar]', fontsize=fs)
    if temp_lim is not None:
        ax.set_xlim(temp_lim)

    if 'ax' not in kwargs: # save as separate plot
        fig.tight_layout()
        name = 'PT_profile' if retr_obj.callback_label=='final_' else f'{retr_obj.callback_label}PT_profile'
        if 'retr_obj2' in kwargs:
            name='comparison/PT_profiles'
        fig.savefig(f'{retr_obj.output_dir}/{name}.pdf')
        if save_jpg:
            fig.savefig(f'{retr_obj.output_dir}/{name}.jpg')
        plt.close()
    
    if get_xmin_xmax:
        return xmin,xmax
    else:
        return

def get_plotposterior_labels(retr_obj,param_names,get_medians=False):
    param_labels = [] # mathtext  
    medians = []
    plot_posterior=np.empty((len(param_names),len(retr_obj.posterior[list(retr_obj.posterior.keys())[0]][0]))).T
    for i,key in enumerate(param_names):
        species_i = key[4:] #without log
        if species_i in retr_obj.vary_species: # posterior at maximum emission contribution
            log10_vmr_at_maxemcont = np.log10(np.array(retr_obj.VMR_dict[species_i])[:,retr_obj.params_dict['idx_maxcont']])
            plot_posterior[:,i]=log10_vmr_at_maxemcont
            param_labels.append(rf"log {retr_obj.species_info.loc[species_i,'mathtext_name']}")
            medians.append(np.nanmedian(log10_vmr_at_maxemcont))
        else:
            param_labels.append(retr_obj.posterior[key][1]) 
            plot_posterior[:,i]=np.array(retr_obj.posterior[key][0]).T
            medians.append((retr_obj.get_quantiles(retr_obj.posterior[key][0]))[0])
    if get_medians==True:
        return plot_posterior, param_labels, medians
    else:
        return plot_posterior, param_labels

def get_true_values(retr_obj,param_names,param_labels=None,
                    titles=None,fig=None,fs=None,plot_lines=True):
    # add true values of test spectrum, plotting didn't work bc x-axis range so small, some didn't show up  
    if retr_obj.target.name in ['test','test_corr','testsys']:
        from testspec import test_parameters,test_mathtext
    elif retr_obj.target.name=='test_ROXs12B':
        from testspec_ROXs12B import test_parameters,test_mathtext
    
    compare=[]
    for key_i in param_names:
        if key_i in test_parameters.keys():
            label_i = test_mathtext[key_i]
            value_i=test_parameters[key_i]
            compare.append(value_i) # add only those values that are used in cornerplot
    
    if fig is not None:
        x=0
        for i in range(len(compare)):
            titles[x] = titles[x]+'\n'+f'in: {compare[i]}'
            fig.axes[x].title.set_text(titles[x])
            if plot_lines:
                fig.axes[x].axvline(compare[i],c='r', alpha=0.7)
            x+=len(param_labels)+1
        
        if plot_lines:
            axes = np.array(fig.axes).reshape(len(param_labels), len(param_labels))
            for i in range(len(param_labels)):
                for j in range(len(param_labels)):
                    if i > j:  # only 2D panels
                        ax = axes[i, j]
                        ax.axvline(compare[j], color='r', alpha=0.7)
                        ax.axhline(compare[i], color='r', alpha=0.7)

        # Adjust tick label font size
        if fs is not None: # change fontsize
            axes = fig.get_axes()
            for ax in axes:
                ax.tick_params(axis="both", labelsize=fs*0.5)  # Tick label font size
        return
    else:
        return compare

def cornerplot(retr_obj,getfig=False,figsize=20,fs=12,plot_label='',alphas=False,
            only_abundances=False,only_params=None,not_abundances=False,ratios=False,
            cloud_params=False,xtickfs=None,labelpad=-0.35):
    
    param_names = []
    if only_abundances==True: # plot only abundances
        plot_label='_abunds'
        if retr_obj.chemistry in ['freechem','varchem']:
            abunds=[]
            param_names=[]
            species=retr_obj.species_names
            for spec in species:
                if spec in retr_obj.vary_species:
                    median_at_maxemcont = np.nanmedian(np.array(retr_obj.VMR_dict[spec])[:,retr_obj.params_dict['idx_maxcont']])
                    abunds.append(median_at_maxemcont)
                else:
                    #suffix='_0' if spec in retr_obj.vary_species else ''
                    abunds.append(retr_obj.params_dict[f'log_{spec}'])
                param_names.append(f'log_{spec}')

            abunds, param_names = zip(*sorted(zip(abunds, param_names)))
            param_names = param_names[::-1] # sort from most to least abundant

    elif only_params is not None: # keys of specified parameters to plot
        param_names = only_params
        plot_label='_some'

    elif not_abundances==True: # plot all except abundances
        plot_label='_rest'
        set_diff = np.setdiff1d(list(retr_obj.parameters.free_params.keys()),[f'log_{s}' for s in retr_obj.species_names])
        for key in set_diff:
            if not re.search(r'_\d+', key): # not include rest of abundances for varchem
                param_names.append(key)

    elif ratios==True:
        figsize=9
        plot_label='_ratios'
        param_names = get_ratios(retr_obj)

    elif alphas==True: # only for flexequ, scaling factors of equchem
        plot_label='_alphas'
        param_names=[]
        species=retr_obj.species_names
        for spec in species:
            if spec not in ['13CO','C17O','C18O','H2(18)O']:
                param_names.append(f"log_a_{spec}")

    elif cloud_params:
        plot_label='_cloud'
        figsize=6
        param_names=['log_opa_base_gray','log_P_base_gray','fsed_gray']
        if 'cloud_slope' in retr_obj.parameters.params:
            param_names.append('cloud_slope')

    else: # all params: avoid, could be too big
        plot_label='_all'
        param_names= list(retr_obj.parameters.free_params.keys())

    compare=None
    if retr_obj.target.name in ['test','test_corr','testsys','test_ROXs12B']:
        compare = get_true_values(retr_obj,param_names,param_labels=None,
                    titles=None,fig=None,fs=None)

    plot_posterior, param_labels, medians = get_plotposterior_labels(retr_obj,param_names,get_medians=True)
    ranges = []
    for k in range(plot_posterior.shape[1]):
        lo, hi = np.percentile(plot_posterior[:, k], [0.5, 99.5])  # broad limits
        # extend to include comparison value
        if compare is not None:
            lo = min(lo, compare[k])
            hi = max(hi, compare[k])   
        # add a margin
        margin = 0.05 * (hi - lo)
        ranges.append((lo - margin, hi + margin))
    #rk_idx = only_params.index("phi_disk")
    #ranges[rk_idx] = (0,0.015) # Fe/H 
    #feh_idx = only_params.index("Fe/H")
    #ranges[feh_idx] = (-0.4,-0.16) # Fe/H 

    fig = plt.figure(figsize=(figsize,figsize)) # fix size to avoid memory issues
    fig = corner.corner(plot_posterior, 
                        labels=param_labels, 
                        title_kwargs={'fontsize':fs},
                        label_kwargs={'fontsize':fs*0.8},
                        color=retr_obj.color,
                        linewidths=0.5,
                        fill_contours=True,
                        quantiles=[0.16,0.5,0.84],
                        title_quantiles=[0.16,0.5,0.84],
                        show_titles=True,
                        hist_kwargs={'density': False,
                                'fill': True,
                                'alpha': 0.5,
                                'edgecolor': 'k',
                                'linewidth': 1.0},
                        fig=fig,
                        quiet=True,
                        range=ranges)
    
    # split title to avoid overlap with plots
    titles = [axi.title.get_text() for axi in fig.axes]
    for i, title in enumerate(titles):
        if len(title) > 1:
            title_split = title.split('=')
            titles[i] = title_split[0] + '\n ' + title_split[1]
        fig.axes[i].title.set_text(titles[i])

    #corner.overplot_lines(fig,medians,color=retr_obj.color,lw=1.3,linestyle='solid') # plot median values of posterior

    if retr_obj.target.name in ['test','test_corr','testsys','test_ROXs12B']:
        get_true_values(retr_obj,param_names,param_labels,
                        titles,fig,fs=None,plot_lines=True)

    if "Sorg" in retr_obj.target.name and only_abundances==True:
        # input VMRs at input's maximum emission contr
        VMR_maxcont = load_pickle(f'LIFE/{retr_obj.target.name}/VMR_maxcont.pickle')

        # maximum VMR of each species in inpu
        # emission from some species come from higher altitudes
        VMR_maxtot = load_pickle(f'LIFE/{retr_obj.target.name}/VMR_maxtotal.pickle') 
        sorted_keys = [key for key in param_names if key in VMR_maxcont] # so that it's at correct position
        sorted_keys = [key for key in param_names if key in VMR_maxtot] # so that it's at correct position
        
        comp_maxcont=[]
        comp_maxtot=[]
        for key_i in sorted_keys:
            maxcont=VMR_maxcont[key_i]
            maxtot = VMR_maxtot[key_i]
            if key_i in param_names:
                comp_maxcont.append(maxcont) # add only those values that are used in cornerplot
                comp_maxtot.append(maxtot)
        x=0
        for i in range(len(comp_maxcont)):
            titles[x] = titles[x]+'\n'+f'em: {np.round(comp_maxcont[i],decimals=2)}'+'\n'+f'max: {np.round(comp_maxtot[i],decimals=2)}'
            fig.axes[x].title.set_text(titles[x])
            x+=len(param_labels)+1

        # Adjust tick label font size
        axes = fig.get_axes()
        for ax in axes:
            ax.tick_params(axis="both", labelsize=fs*0.5)  # Tick label font size
    
    elif "Sorg" in retr_obj.target.name:
        from LIFE_pRT import get_input_dict
        def count_dlnT_dlnP_keys(d):
            count = 0
            n = 0
            while True:
                key = f"dlnT_dlnP_{n}"
                if key in d:
                    count += 1
                    n += 1
                else:
                    break
            return count
        n_pt = count_dlnT_dlnP_keys(retr_obj.params_dict)
        LIFE_parameters = get_input_dict(retr_obj.target.name,retr_obj.pressure,n_pt)
        compare=[]
        for key_i in param_names:
            if key_i in LIFE_parameters.keys():
                value_i=LIFE_parameters[key_i]
                compare.append(value_i) # add only those values that are used in cornerplot
        x=0
        for i in range(len(compare)):
            titles[x] = titles[x]+'\n'+f'in: {np.round(compare[i],decimals=2)}'
            fig.axes[x].title.set_text(titles[x])
            fig.axes[x].axvline(compare[i],c='r',lw=0.8,ls='dashed')
            x+=len(param_labels)+1

        # Adjust tick label font size
        axes = fig.get_axes()
        xtickfs = fs*0.5 if xtickfs is None else xtickfs
        for ax in axes:
            ax.tick_params(axis="both", labelsize=xtickfs)  # Tick label font size
            if ax.get_xlabel():
                ax.xaxis.set_label_coords(0.5, labelpad)  # move downward manually
            if ax.get_ylabel():
                ax.yaxis.set_label_coords(labelpad, 0.5)  # move left manually

    plt.subplots_adjust(wspace=0,hspace=0)

    if getfig==False:
        name= f'cornerplot{plot_label}' if retr_obj.callback_label=='final_' else f'{retr_obj.callback_label}cornerplot{plot_label}'
        fig.savefig(f'{retr_obj.output_dir}/{name}.pdf',
                    bbox_inches="tight",dpi=200)
        plt.close()
    else:
        ax = np.array(fig.axes)
        return fig, ax

def make_all_plots(retr_obj,only_params=None,
                    split_corner=False,comp_equ=True):

    summary_plot(retr_obj,show_params='all')
    plot_contribution_per_species(retr_obj)

    if retr_obj.instrument=='CRIRES':
        plot_spectrum_split(retr_obj)
        plot_spectrum_inset(retr_obj)
        plot_pt(retr_obj)
    if ('log_k_rk' in retr_obj.params_dict) or ('T_disk' in retr_obj.params_dict):
        plot_rk(retr_obj)
    if 'cloud_slope' in retr_obj.params_dict:
        cornerplot(retr_obj,cloud_params=True)
    if retr_obj.chemistry in ['freechem','varchem','quequchem']:
        comp_equ=True # compare with what equchem abundances would be like
    else:
        comp_equ=False
    VMR_plot(retr_obj,VMR_species='all',comp_equ=comp_equ) # show all
   # VMR_plot(retr_obj,comp_equ=comp_equ) # show most abundant (with errors)

    if retr_obj.chemistry in ['freechem','varchem','flexequ']:
        cornerplot(retr_obj,ratios=True)# plot ratios, already in equchem cornerplot by default
    
    if split_corner and retr_obj.chemistry in ['freechem','varchem']: # split corner plot to avoid massive files
        cornerplot(retr_obj,only_abundances=True)
        cornerplot(retr_obj,not_abundances=True)
    elif only_params is not None: # make cornerplot with all parameters, could be huge
        cornerplot(retr_obj,only_params=only_params)
    else:
        cornerplot(retr_obj)
    
    if retr_obj.chemistry=='flexequ':
        plot_scaled_abunds(retr_obj)
    
def summary_plot(retr_obj,**kwargs):

    fs=12
    figsize=14

    if retr_obj.instrument=='LIFE':

        fig = plt.figure(figsize=(10,10))
        l, b, w, h = [0.1, 0.47, 0.6, 0.17] # left, bottom, width, height

        ax_logg = fig.add_axes([l+w+0.08, b, 0.15, 0.15])  # Top-right, optional
        ax_spec = fig.add_axes([l,b,w,h])   # Top-right
        h2 = 0.3
        ax_pt = fig.add_axes([l+w, 0.1, 0.25, h2])    # Bottom-left
        ax_vmr = fig.add_axes([l, 0.1, w, h2]) # Bottom-right
        ax_opa = fig.add_axes([l, 0.64, w, 0.15]) 

        ax_pt.set_zorder(10)   # big number = drawn later = on top
        ax_vmr.set_zorder(1)   # big number = drawn later = on top
        ax_pt.patch.set_facecolor("white")   # or any color you want
        ax_pt.patch.set_alpha(1.0)           # fully opaque

        if 'log_g' in retr_obj.parameters.free_params.keys():
            param_names= ['log_g']
            plot_posterior, param_labels, medians = get_plotposterior_labels(retr_obj,param_names,get_medians=True)
            ax_logg.hist(plot_posterior, bins=20, color=retr_obj.color)
            ax_logg.set_yticks([])
            minus_err, median, plus_err = np.percentile(plot_posterior, [16.0,50.0,84.0])
            title = f'log$g$ = {np.round(medians[0],decimals=2)}$^{{+{np.round(plus_err-medians[0],decimals=2)}}}_{{{np.round(minus_err-medians[0],decimals=2)}}}$'
            darker = darken_color(retr_obj.target.color,amount=0.4)
            ax_logg.axvline(x=minus_err,color=darker,alpha=0.5,linestyle='dashed')
            ax_logg.axvline(x=median,color=darker,alpha=0.7)
            ax_logg.axvline(x=plus_err,color=darker,alpha=0.5,linestyle='dashed')
            ax_logg.axvline(x=3.09,color=retr_obj.target.in_col,alpha=0.8,linestyle='dashdot')
            ax_logg.set_title(f'{title} \nin: 3.09',fontsize=fs)
            ax_logg.tick_params(labelsize=fs)
            ax_logg.xaxis.set_minor_locator(MultipleLocator(0.1))
            ax_logg.tick_params(which='minor', length=2)
            species_bbox= (0.6, 2.1)
        else:
            ax_logg.axis('off')
            species_bbox= (0.5, 0.5)

        ax_res = fig.add_axes([l,b-0.03,w,h-0.12])
        plot_spectrum_inset(retr_obj,ax=(ax_spec,ax_res),inset=False,fs=fs,leg_fs=fs*0.8)
        ax_res.cla()
        ax_res.axis('off')
        ax_spec.set_xlabel(r'Wavelength [$\mathrm{\mu}$m]',fontsize=fs)
        #ax_spec.set_ylabel('photons s$^{-1}$ m$^{-3}$',fontsize=fs)
        ax_spec.set_ylabel('photons/s/m$^3$',fontsize=fs)
        ax_spec.xaxis.set_minor_locator(MultipleLocator(1))
        ax_spec.tick_params(which='minor', length=2)

        if retr_obj.parameters.constant_params['fix_PT'] is None:
            plot_pt(retr_obj,ax=ax_pt,fs=fs,leg_fs=fs*0.8,contr_as_curve=True)
            ax_pt.yaxis.tick_right()
            ax_pt.yaxis.set_label_position("right")
            ax_pt.xaxis.set_minor_locator(MultipleLocator(50))   # small ticks every 10
            ax_pt.minorticks_on()
            ticks = ax_pt.get_xticks()
            ax_pt.set_xticks(ticks[1:-1])
            # add to PT plot
            pres = PSG_input(retr_obj.target.name).pressure
            input_contr = np.load(f'LIFE/{retr_obj.target.name}/input_pRT_summed_contr.npy')
            input_contr =input_contr/np.max(input_contr)
            idx_maxcont=np.where(input_contr == np.max(input_contr))[0][0]
            input_P_maxcont = pres[idx_maxcont] # pressure at max emission contribution
            input_P_maxcont = pres[idx_maxcont] # pressure at max emission contribution
            if retr_obj.parameters.constant_params['fix_PT'] is None:
                ax_pt.axhline(y=input_P_maxcont, xmin=0, xmax=1,alpha=0.3, color=retr_obj.target.in_col,linestyle='dashdot')
                ax_pt.axhline(y=10**retr_obj.params_dict['log_P_maxcont'], xmin=0, xmax=1,alpha=0.3, color=retr_obj.color,linestyle='dashdot')
        else:
            ax_pt.axis('off')

        plot_species = retr_obj.species_names.copy()
        plot_species.append('H2')
        plot_species.append('He')
        line_props = VMR_plot(retr_obj,fs=fs,ax=ax_vmr,VMR_species=plot_species,
                                xmin=1e-10,xmax=1e-0,plotlegend=False,
                                sigma=2,show_contr=False)
        colors, labels=[],[]
        for col,lab in line_props:
            colors.append(col)
            labels.append(lab)
        lines=[]
        for col,lab in zip(colors,labels):
            lines.append(Line2D([0],[0],color=col,linewidth=2.5,label=lab))
        ax_logg.legend(handles=lines,bbox_to_anchor=species_bbox, handlelength=1.3,
                    ncol=3,loc='upper center',fontsize=fs*0.9,frameon=False)

        plot_contribution_per_species(retr_obj,ax=ax_opa,fs=fs,plot_legend=False)
        ax_opa.set_xticks([])
        ax_opa.set_xlabel('')
        ax_opa.tick_params(axis='y', labelsize=fs)

        name = 'summary' if retr_obj.callback_label=='final_' else f'{retr_obj.callback_label}summary'
        fig.savefig(f'{retr_obj.output_dir}/{name}.pdf',
                    bbox_inches="tight",dpi=200)
        plt.close()
        return

    elif retr_obj.chemistry in ['equchem','quequchem','flexequ']:
        if retr_obj.chemistry in ['equchem','quequchem']:
            only_params=['rv','vsini']
            if {'Fe/H', 'C/O'}.issubset(retr_obj.parameters.free_params):
                only_params.extend(['C/O', 'Fe/H'])
        elif retr_obj.chemistry =='flexequ':
            only_params=['rv','vsini']
        if 'log_g' in retr_obj.parameters.free_params:
            only_params.append('log_g')
        if '13CO' in retr_obj.species_names:
            only_params.append('log_C12_13_ratio')
        if 'C18O' in retr_obj.species_names:
            only_params.append('log_O16_18_ratio')
        if 'H2(18)O' in retr_obj.species_names:
            only_params.append('log_H2O16_18_ratio')
        if 'C17O' in retr_obj.species_names:
            only_params.append('log_O16_17_ratio')
    elif retr_obj.chemistry in ['freechem','varchem']:
        only_params=['rv','vsini']
        if 'log_g' in retr_obj.parameters.free_params:
            only_params.append('log_g')
        if retr_obj.instrument=='LIFE':
            if 'log_g' in retr_obj.parameters.free_params.keys():
                only_params=['log_g']
        abunds=[]
        param_names=[]
        species=retr_obj.species_names
        for spec in species:
            suffix='_0' if spec in retr_obj.vary_species else ''
            abunds.append(retr_obj.params_dict[f'log_{spec}{suffix}'])
            param_names.append(f"log_{spec}{suffix}")
        abunds, param_names = zip(*sorted(zip(abunds, param_names)))
        only_params.extend(param_names[-8:][::-1]) # get most abundant species
    if 'show_params' in kwargs:
        only_params=kwargs.get('show_params')
        if only_params=='all':
            only_params = retr_obj.parameters.free_params
        #figsize=15
        #fs=9

    fig, ax = cornerplot(retr_obj,getfig=True,only_params=only_params,figsize=figsize,fs=fs)
    l, b, w, h = [0.42,0.84,0.55,0.15]#[0.37,0.84,0.6,0.15] # left, bottom, width, height
    ax_spec = fig.add_axes([l,b,w,h])
    ax_res = fig.add_axes([l,b-0.03,w,h-0.12])
    plot_spectrum_inset(retr_obj,ax=(ax_spec,ax_res),inset=False,fs=fs*1.2)

    l, b, w, h = [0.68,0.47,0.29,0.29] # left, bottom, width, height
    ax_PT = fig.add_axes([l,b,w,h])
    plot_pt(retr_obj,ax=ax_PT,fs=fs*1.2)
    name = 'summary' if retr_obj.callback_label=='final_' else f'{retr_obj.callback_label}summary'
    fig.savefig(f'{retr_obj.output_dir}/{name}.pdf',
                bbox_inches="tight",dpi=200)
    plt.close()

def compare_retrievals(retr_obj1,retr_obj2,fs=12,with_pt=True,
                       show_contr=True,contr_as_curve=False,
                       ratios_logg=False,figsize=12,save_jpg=False,
                       show_pt_legend=True,pt_coord=None,
                       colors = [], **kwargs):

    legend_labels=kwargs.get('legend_labels',None)
    show_legend=show_pt_legend
    num=2 # number of retrs
    suffix=''
    l, b, w, h = [0.58,0.65,0.39,0.39] # left, bottom, width, height for PT plot

    if 'show_params' in kwargs:
        only_params=kwargs.get('show_params')
        posterior1, labels = get_plotposterior_labels(retr_obj1,only_params)
        posterior2, _ = get_plotposterior_labels(retr_obj2,only_params)

    elif retr_obj1.chemistry=='freechem' and retr_obj2.chemistry=='freechem':

        if 'show_params' in kwargs:
            only_params=kwargs.get('show_params')
        elif ratios_logg==True:
            suffix = 'ratios_'
            with_pt=False
            figsize=9
            only_params= ['log_g','C/O','C/H','log_12CO/13CO','log_H2O/H2(18)O']
        else:
            suffix = 'abunds_'
            only_params=['log_H2O','log_CO','log_13CO','log_CH4','log_H2S','log_HF','log_H2(18)O','log_NH3']

        posterior1, labels = get_plotposterior_labels(retr_obj1,only_params)
        posterior2, _ = get_plotposterior_labels(retr_obj2,only_params)

    elif retr_obj1.chemistry=='freechem' and retr_obj2.chemistry in ['equchem','quequchem','flexequ']:

        suffix='chems'
        only_params=['log_g','C/O','C/H','log_12CO/13CO','log_H2O/H2(18)O']

        posterior1, labels = get_plotposterior_labels(retr_obj1,only_params)

        # add log_O16_18_ratio to equchem again, bc freechem has C18O and H218O ratios
        #only_params=['log_g','C/O','Fe/H','log_C12_13_ratio','log_O16_17_ratio','log_O16_18_ratio','log_O16_18_ratio']
        only_params=['log_g','C/O','Fe/H','log_C12_13_ratio','log_H2O16_18_ratio']
        posterior2, labels = get_plotposterior_labels(retr_obj2,only_params)
        figsize=8.5
        legend_labels = ['Free','Equ']
        l, b, w, h = [0.62,0.72,0.33,0.33] # left, bottom, width, height for PT plot

        if 'retr_obj3' in kwargs: # quequchem
            num=3
            retr_obj3=kwargs.get('retr_obj3')
            posterior3, labels = get_plotposterior_labels(retr_obj3,only_params)
            legend_labels.append('Quequ')

    else: # two equchem/equchem retrievals

        if 'show_params' in kwargs:
            only_params=kwargs.get('show_params')
        posterior1, labels = get_plotposterior_labels(retr_obj1,only_params)
        posterior2, _ = get_plotposterior_labels(retr_obj2,only_params)

    both_posteriors = np.vstack([posterior1, posterior2])
    fig = plt.figure(figsize=(figsize,figsize)) # fix size to avoid memory issues
    
    if colors==[]:
        colors_list = [retr_obj1.color,retr_obj2.color]
    else:
        colors_list = colors

    def plot_corner(posterior,retr_obj,labels,fig,col,
                    setfig=False,getfig=False,nbins=30):
        
        if setfig: # pass both posteriors to set ax sizes, make invisible tho
            alpha=0.
        else:
            alpha=0.5
        
        ranges = [(np.min([np.min(posterior1[:, i]),np.min(posterior2[:, i])]), 
                   np.max([np.max(posterior1[:, i]),np.max(posterior2[:, i])])) for i in range(posterior1.shape[1])]

        if retr_obj.target.name=='2M1425' and suffix =='abunds_':
            ranges[3] = (-11,-4.7) # increase range for CH4 to make better visible
            ranges[4] = (-11,-3.5) # increase range for H2S to make better visible
            ranges[6] = (-12,-4.7) # increase range for H218O to make better visible
        if 'ROXs' in retr_obj.target.name:
            if "rv" in only_params:
                rv_idx = only_params.index("rv")
                ranges[rv_idx] = (-5.6,-2.4) # vsini
            if 'vsini' in only_params:
                vsini_idx = only_params.index("vsini")
                ranges[vsini_idx] = (2.5,9) # vsini
            if 'C/O' in only_params:
                CO_idx = only_params.index("C/O")
                ranges[CO_idx] = (0.45,0.93) # C/O  
            FeH_idx = only_params.index("Fe/H")
            ranges[FeH_idx] = (-0.65,0.05) # Fe/H 
            COiso_idx = only_params.index("log_C12_13_ratio")
            ranges[COiso_idx] = (1.47,2.1) # 12/13C 

        fig = corner.corner(posterior, 
                        labels=labels, 
                        title_kwargs={'fontsize': fs},
                        label_kwargs={'fontsize': fs*0.8},
                        color=col,
                        linewidths=0.5,
                        fill_contours=True,
                        quantiles=[0.16,0.5,0.84],
                        title_quantiles=[0.16,0.5,0.84],
                        show_titles=True,
                        plot_contours=True,
                        bins=nbins,
                        hist_kwargs={'density': False,
                                    'fill': True,
                                    'alpha': alpha,
                                    'edgecolor': 'k',
                                    'linewidth': 1.0},
                        fig=fig,
                        quiet=True,
                        range=ranges)
        
        titles = [axi.title.get_text() for axi in fig.axes]
        if getfig:
            return fig,titles,ranges
        else:
            return titles,ranges
        
    fig = corner.corner(both_posteriors,
                    bins=50,
                    color='gray',
                    hist_kwargs={'density': False, 'alpha': 0.0},  # invisible hist for scaling
                    plot_datapoints=False,
                    fill_contours=False,
                    plot_contours=False,
                    show_titles=False)
    titles1,_=plot_corner(posterior1,retr_obj1,labels,fig,col=colors_list[0])
    titles2,ranges=plot_corner(posterior2,retr_obj2,labels,fig,col=colors_list[1])
    enum=[0,1]
    titles_list=[titles1,titles2]

    if 'retr_obj3' in kwargs:
        titles3=plot_corner(posterior3,retr_obj3,labels,fig)
        enum=[0,1,2]
        titles_list.append(titles3)
        colors_list.append(retr_obj3.color)

    for i, axi in enumerate(fig.axes):
        fig.axes[i].title.set_visible(False) # remove original titles
        fig.axes[i].xaxis.label.set_fontsize(fs)
        fig.axes[i].yaxis.label.set_fontsize(fs)
        fig.axes[i].tick_params(axis='both', which='major', labelsize=fs*0.8)
        fig.axes[i].tick_params(axis='both', which='minor', labelsize=fs*0.8)

    # Diagonal plots (1D histograms) are at positions 0, ndim+1, 2*(ndim+1), ...
    ndim = posterior1.shape[1]
    quantile_styles = [
    {'alpha':0.5, 'linestyle':'--'},  # lower
    {'alpha':1.0, 'linestyle':'-'},   # median
    {'alpha':0.5, 'linestyle':'--'}   # upper
    ]
    n_quantiles = len([0.16,0.5,0.84])  # or whatever you set in corner]

    for i in range(ndim):
        ax = fig.axes[i*(ndim+1)]
        lines = ax.get_lines()  # all quantile lines for all posteriors

        for p_idx in range(2):
            for q_idx, style in enumerate(quantile_styles):
                line_idx = p_idx*n_quantiles + q_idx
                line = lines[line_idx]
                line.set_alpha(style['alpha'])
                line.set_linestyle(style['linestyle'])
                line.set_color(colors_list[p_idx])  # optional: color per posterior
        
    for run,titles_list,color in zip(enum,titles_list,colors_list):
        # add new titles
        for j, title in enumerate(titles_list):
            if title == '':
                continue
            
            # first only the name of the parameter
            s = title.split('=')
            if len(enum)==2:
                y_title=1.45
                y=y_title-(0.2*(run+1))
            elif len(enum)==3:
                y_title=1.47
                y=y_title-0.12-(0.13*(run+0.2))
            if run == 0: # first retr, add parameter name
                fig.axes[j].text(0.5, y_title, s[0], fontsize=fs,
                                ha='center', va='bottom',
                                transform=fig.axes[j].transAxes,
                                color='k',
                                weight='normal')
            # add parameter value with custom color and spacing
            fig.axes[j].text(0.5, y, s[1], fontsize=fs,
                            ha='center', va='bottom',
                            transform=fig.axes[j].transAxes,
                            color=darken_color(color,amount=0.9),
                            weight='normal')
            
    # --- Fix the 1D histogram y-axis limits ---
    def compute_hist_peaks(posteriors, bins=30, ranges=None, density=False, pad=1.1):
        """
        Compute per-dimension maximum histogram heights across multiple posteriors.
        Uses the SAME bins and SAME range per dimension as corner will use.
        - posteriors: list of (N, ndim) arrays
        - bins: int number of bins (must equal the bins passed to corner for plotting)
        - ranges: list of (xmin, xmax) tuples, one per dimension (must equal 'range' passed to corner)
        - density: same as hist_kwargs['density']
        - pad: multiplier
        Returns list of y_max per dimension (counts if density=False).
        """
        ndim = posteriors[0].shape[1]
        if ranges is None:
            # default to combined min/max if you didn't precompute ranges
            combined = np.vstack(posteriors)
            ranges = [(np.min(combined[:, i]), np.max(combined[:, i])) for i in range(ndim)]

        y_max = []
        for dim in range(ndim):
            xmin, xmax = ranges[dim]
            peak = 0.0
            # compute histogram with the exact same bins and the exact same range
            for p in posteriors:
                hist, edges = np.histogram(p[:, dim], bins=bins, range=(xmin, xmax), density=density)
                peak = max(peak, hist.max())
            y_max.append(peak * pad)
        return y_max

    y_max = compute_hist_peaks([posterior1, posterior2], bins=50, ranges=ranges, density=False, pad=1.7)
    ndim = posterior1.shape[1] # number of parameters
    for dim in range(ndim):
        ax = fig.axes[dim * (ndim + 1)]  # diagonal axes
        ax.set_ylim(0, y_max[dim])

    if "Sorg" in retr_obj1.target.name:
        from LIFE_pRT import get_input_dict
        def count_dlnT_dlnP_keys(d):
            count = 0
            n = 0
            while True:
                key = f"dlnT_dlnP_{n}"
                if key in d:
                    count += 1
                    n += 1
                else:
                    break
            return count
        n_pt = count_dlnT_dlnP_keys(retr_obj1.params_dict)
        LIFE_parameters = get_input_dict(retr_obj1.target.name,retr_obj1.pressure,n_pt)
        compare=[]
        for key_i in only_params:
            if key_i in LIFE_parameters.keys():
                value_i=LIFE_parameters[key_i]
                compare.append(value_i) # add only those values that are used in cornerplot
        x=0
        for i in range(len(compare)):
            #titles[x] = titles[x]+'\n'+f'in: {np.round(compare[i],decimals=2)}'
            #fig.axes[x].title.set_text(titles[x])
            fig.axes[x].axvline(compare[i],c=retr_obj1.target.in_col,lw=1,ls='dashed')
            x+=len(labels)+1

    plt.subplots_adjust(wspace=0, hspace=0)

    if with_pt==True:
        if pt_coord is None:
            ax_PT = fig.add_axes([l,b,w,h])
        else:
            ax_PT = fig.add_axes(pt_coord)
        if 'retr_obj3' not in kwargs:
            plot_pt(retr_obj1,retr_obj2=retr_obj2,ax=ax_PT,
                    legend_labels=legend_labels,comp_pt=False,
                    colors=colors,show_legend=show_legend,
                    show_contr=show_contr,contr_as_curve=show_contr)
        else:
            plot_pt(retr_obj1,retr_obj2=retr_obj2,
                    ax=ax_PT,retr_obj3=retr_obj3,
                    legend_labels=legend_labels,comp_pt=False,
                    show_legend=show_legend,
                    show_contr=show_contr,contr_as_curve=show_contr)

    else: # show legend for corner plot if not in PT plot
        for r,l in zip([retr_obj1,retr_obj2],legend_labels):
            lines.append(Line2D([0], [0], color=r.color,linewidth=5,linestyle='-',label=l,alpha=0.8))
        plt.legend(handles=lines,
                    fontsize=fs*1.4,
                    loc='upper right',
                    bbox_to_anchor=(1., 1.1),  # top right corner of the figure
                    bbox_transform=plt.gcf().transFigure,  # interpret coords in figure space
                    frameon=False
                )

    if show_pt_legend==False and legend_labels is not None:
        leg_lines=[]
        if len(legend_labels)!=len([retr_obj1,retr_obj2]): # input part of legend
            colors = [colors[0],colors[1],retr_obj1.target.in_col]
        for c,l in zip(colors,legend_labels):
            leg_lines.append(Line2D([0], [0], color=c,linewidth=3,linestyle='-',label=l))
        plt.legend(handles=leg_lines,handlelength=1.5,fontsize=fs,
                    loc='center',                      # which point of legend box is anchored
                    bbox_to_anchor=(0.4, 1),         # (x, y) in figure coords
                    bbox_transform=fig.transFigure)     # ← KEY LINE)

    comparison_dir=pathlib.Path(f'{retr_obj1.output_dir}/comparison') # store output in separate folder
    comparison_dir.mkdir(parents=True, exist_ok=True)

    fig.set_size_inches(figsize,figsize)
    fig.savefig(f'{comparison_dir}/cornerplot_{suffix}{num}.pdf',bbox_inches="tight",dpi=200)
    if save_jpg:
        fig.savefig(f'{comparison_dir}/cornerplot_{suffix}{num}.jpg',bbox_inches="tight",dpi=200)
    plt.close()

def VMR_plot(retr_obj,fs=10,n=8,VMR_species='all',comp_equ=False,sigma=2,
                addname=False,plotlegend=False,wH2He=False,show_contr=True,
                xmin=1e-12,xmax=1e-2,save_jpg=False,add_species=None,
                ymin=None,ymax=None,figsize=(5.5,3.5),linestyles=[],**kwargs):

    prefix = retr_obj.callback_label if retr_obj.callback_label=='live_' else ''
    suffix=''
    output_dir=retr_obj.output_dir
    ymin = np.min(retr_obj.pressure) if ymin==None else ymin
    ymax = np.max(retr_obj.pressure) if ymax==None else ymax
    xmax=1e0 if wH2He else 1e-2

    if 'ax' in kwargs:
        ax=kwargs.get('ax')
    else:
        fig,ax=plt.subplots(1,1,figsize=figsize,dpi=200)

    legend_labels=0
    chemleg=[] # legend for chemistry
    pressure=retr_obj.model_object.pressure

    # plot n most abundant species
    if VMR_species==None:
        suffix='_few'
        abunds=[]
        species=retr_obj.species_names
        if wH2He and retr_obj.chemistry in ['equchem','quequchem','flexequ']:
            species.extend(s for s in ['H2','He'] if s not in species)
        
        if retr_obj.chemistry in ['freechem','varchem']:
            for spec in species:
                s='_0' if spec in retr_obj.vary_species else ''
                abunds.append(retr_obj.params_dict[f"log_{spec}{s}"])
        
        elif retr_obj.chemistry in ['equchem','quequchem','flexequ']: # use VMRs where emission contribution is maximal
            for spec in species:
                abunds.append(np.median(retr_obj.VMR_dict[spec],axis=0)[find_nearest(retr_obj.pressure,10**retr_obj.params_dict['log_P_maxcont'])])

        abunds, species = zip(*sorted(zip(abunds, species)))
        VMR_species=species[-n:][::-1] # get n largest

    elif VMR_species=='all':
        suffix='_all'
        VMR_species = retr_obj.species_names
        if wH2He and retr_obj.chemistry in ['equchem','quequchem','flexequ']:
            VMR_species.extend(s for s in ['H2','He'] if s not in VMR_species)
    else:
        suffix = '_few'
        
    additional = 2 if wH2He else 0
    legend_ncol = (len(retr_obj.species_names) + 1 + additional) // 2 # max 2 in column

    def plot_VMRs(retr_obj,ax,ax2,ls=None,main=True):
        
        if main: # main retrieval
            alpha=1 if ls is None else ls
            linestyle='solid'
        else: # comparing to smth else
            alpha= 0.4
            linestyle='dashed' if ls is None else ls

        label_map = {'freechem': 'Free','equchem': 'Equ','flexequ': 'Flex',
                            'varchem': 'Var','quequchem': 'Quench'}
        chemleg.append(Line2D([0], [0], color='k',linestyle=linestyle,linewidth=2,
                        alpha=alpha,label=label_map.get(retr_obj.chemistry)))

        if main:
            contribution_plot=retr_obj.summed_contribution/np.max(retr_obj.summed_contribution)*(xmax-xmin)+xmin
            global vmr_emcont
            vmr_emcont, = ax2.plot(contribution_plot,pressure,lw=1,alpha=0.5,color=retr_obj.color,linestyle=linestyle)
            ax2.set_xlim(np.min(contribution_plot),np.max(contribution_plot))
            ax2.set_ylim(ymin,ymax)
            contr_max=pressure[np.where(retr_obj.summed_contribution==np.max(retr_obj.summed_contribution))[0]]
            ax2.set_yscale('log')

        for species in VMR_species:
            color=retr_obj.species_info.loc[species,'color']
            label=retr_obj.species_info.loc[species,'mathtext_name']
            if retr_obj.chemistry=='freechem' and retr_obj.use_partial_pressure==False:
                label=label if legend_labels==0 else '_nolegend_' 
                if species=='He':
                    VMR=0.15
                elif species=='H2':
                    VMR_wo_H2 = 0.15 # He
                    for spc in VMR_species:
                        if spc not in ['H2','He']:
                            VMR_wo_H2 += 10**retr_obj.params_dict[f'log_{spc}']
                    VMR = 1-VMR_wo_H2
                else:
                    VMR=10**retr_obj.params_dict[f'log_{species}']
                    p = 10**np.array(np.percentile(retr_obj.posterior[f'log_{species}'][0],[0.2,2.3,15.9,50.0,84.1,97.7,99.8], axis=-1))
                    median = p[3]
                    sm = p[3 - sigma]
                    sp = p[3 + sigma]
                if 'retr_obj2' not in kwargs or retr_obj.target.name in ['Sorg1X','Sorg20X']:
                    ax.plot(np.ones_like(pressure)*VMR,pressure,label=label,linestyle=linestyle,c=color)
                    if suffix!='_all' and species not in ['H2','He']: # only show errors when not showing all species, or will be cluttered
                        ax.fill_betweenx(pressure,sm,sp,color=color,alpha=0.1) # 95% confidence interval
                else: # plot only as point to avoid cluttering
                    ax.scatter(VMR,contr_max, color=color,s=13)
                    ax.plot([sm,sp],[contr_max,contr_max], color=color,lw=1.5)
            elif (retr_obj.chemistry in ['equchem','quequchem','flexequ','varchem']) or (retr_obj.chemistry=='freechem' and retr_obj.use_partial_pressure==True):
                label=label if legend_labels==0 else '_nolegend_'
                if main:
                    p=np.percentile(retr_obj.VMR_dict[species], [0.2,2.3,15.9,50.0,84.1,97.7,99.8], axis=0)
                    median = p[3]
                    sm = p[3 - sigma]
                    sp = p[3 + sigma]
                    ax.plot(median,pressure,label=label,alpha=alpha,linestyle=linestyle,c=color)
                    ax.fill_betweenx(pressure,sm,sp,color=color,alpha=0.1) 
                else: # compare with computed equchem based on same params
                    ax.plot(retr_obj.model_object.VMR_dict[species],pressure,label=label,alpha=alpha,linestyle=linestyle,c=color)

        if add_species!=None:
            model_object=pRT_spectrum(retr_obj,add_species=add_species)
            color=retr_obj.species_info.loc[add_species,'color']
            label=retr_obj.species_info.loc[add_species,'mathtext_name']
            ax.plot(model_object.VMR_dict[add_species],pressure,label=label,alpha=alpha,linestyle=linestyle,c=color)
    
    #ax2 = ax.inset_axes([0,0,1,1]) # [x0, y0, width, height] , for emission contribution
    ax2 = ax.inset_axes([1,0,0.1,1])
    ls=linestyles[0] if linestyles!=[] else None
    plot_VMRs(retr_obj,ax=ax,ax2=ax2,ls=ls)
    legend_labels=1 if 'retr_obj2' not in kwargs else 0 # only make legend labels once 

    # compare VMRs to equilibrium chemistry with other retrieved params remaining equal
    if comp_equ==True:
        #suffix+='_compequ'
        from retrieval import Retrieval
        from parameters import Parameters
        parameters_equ = retr_obj.params_dict
        if retr_obj.chemistry!='quequchem':
            parameters_equ.update({'C/O': retr_obj.params_dict['C/O'], 
                                    'Fe/H': retr_obj.params_dict['C/H'],
                                    'const_species': [],
                                    'gaussian_species': []})
            ratios_free,ratios_equ = get_ratios(retr_obj,equ_too=True)
            for r,e in zip(ratios_free,ratios_equ):
                parameters_equ.update({e: retr_obj.params_dict[r]})
        parameters_equ = Parameters({}, parameters_equ)
        if retr_obj.parameters.params['use_GP']:
            parameters_equ.param_priors['log_l']=[-3,0]

        retr_equ = Retrieval(target=retr_obj.target,parameters=parameters_equ,
                                  Nlive=retr_obj.Nlive,
                                  evtol=retr_obj.evtol,chemistry='equchem',
                                  PT_type=retr_obj.PT_type)

        retr_equ.model_object=pRT_spectrum(retr_equ,contribution=True)
        retr_equ.model_flux0=retr_equ.model_object.make_spectrum() # to get contr_em
        retr_equ.summed_contribution= retr_equ.model_object.summed_contribution # average over all orders
        plot_VMRs(retr_equ,ax=ax,ax2=ax2,main=False)

        # folder created when initializing retrieval object, delete afterwards
        if os.path.exists(retr_equ.output_dir) and not os.listdir(retr_equ.output_dir):  # Check if folder exists and is empty
            os.rmdir(retr_equ.output_dir)  # Remove empty folder

    if show_contr:
        summed_contr=retr_obj.summed_contribution 
        prs = retr_obj.model_object.pressure
        photosphere_gradient(summed_contr,prs,ax,retr_obj.color,xmin,xmax)

    if 'retr_obj2' in kwargs: # compare two retrs
        suffix='_2'
        retr_obj2=kwargs.get('retr_obj2')
        plt.gca().set_prop_cycle(None) # reset color cycle
        ls=linestyles[1] if linestyles!=[] else None
        plot_VMRs(retr_obj2,ax=ax,ax2=ax2,ls=ls)
        legend_labels=1
        comparison_dir=pathlib.Path(f'{retr_obj.output_dir}/comparison') # store output in separate folder
        comparison_dir.mkdir(parents=True, exist_ok=True)
        output_dir=comparison_dir

    if 'retr_obj3' in kwargs: # compare three retrs
        suffix='_3'
        retr_obj3=kwargs.get('retr_obj3')
        plt.gca().set_prop_cycle(None) # reset color cycle
        ls=linestyles[2] if linestyles!=[] else None
        plot_VMRs(retr_obj3,ax=ax,ax2=ax2,ls=ls)

    if retr_obj.target.name in ['Sorg1X','Sorg20X']:
        tab = PSG_input(retr_obj.target.name).table
        pres = PSG_input(retr_obj.target.name).pressure
        input_contr = np.load(f'LIFE/{retr_obj.target.name}/input_pRT_summed_contr.npy')
        input_contr =input_contr/np.max(input_contr)*(xmax-xmin)+xmin
        idx_maxcont=np.where(input_contr == np.max(input_contr))[0][0]
        input_P_maxcont = pres[idx_maxcont] # pressure at max emission contribution
        ax2.plot(input_contr,pres,linestyle='dotted',lw=1.5,alpha=0.5,color=retr_obj.target.in_col)
        ax2.axhline(y=input_P_maxcont, xmin=0, xmax=1,alpha=0.3, color=retr_obj.target.in_col,linestyle='dashdot')
        ax2.axhline(y=10**retr_obj.params_dict['log_P_maxcont'], xmin=0, xmax=1,alpha=0.3, color=retr_obj.color,linestyle='dashdot')
        ax.axhline(y=input_P_maxcont, xmin=0, xmax=1,alpha=0.3, color=retr_obj.target.in_col,linestyle='dashdot')
        ax.axhline(y=10**retr_obj.params_dict['log_P_maxcont'], xmin=0, xmax=1,alpha=0.3, color=retr_obj.color,linestyle='dashdot')

        for species in tab.columns.tolist():
            if species in VMR_species:
                color=retr_obj.species_info.loc[species,'color']
                if species=='H2O':
                    if retr_obj.target.const_H2O=='':
                        ax.plot(tab[species],pres,alpha=1,linestyle='solid',c=color)
                    else:
                        h2o_vmr = np.ones_like(tab[species])*0.023412
                        ax.plot(h2o_vmr,pres,alpha=1,linestyle='solid',c=color)
                else:
                    ax.plot(tab[species],pres,alpha=1,linestyle='solid',c=color)
        chemleg.append(Line2D([0], [0], color='k',linestyle='solid',linewidth=2,alpha=0.7,label='Input'))
        chemleg.append(Line2D([0], [0], color=retr_obj.color, linewidth=2,alpha=0.3, linestyle='dashdot',label='Retr $P_\mathrm{max}$'))
        chemleg.append(Line2D([0], [0], color=retr_obj.target.in_col, linewidth=2,alpha=0.3, linestyle='dashdot',label='Input $P_\mathrm{max}$'))

    if comp_equ==True or 'retr_obj2' in kwargs or retr_obj.target.name in ['Sorg1X','Sorg20X']:
        leg2=ax.legend(handles=chemleg,fontsize=fs*0.8,loc='upper left')

    if addname==True:
        from matplotlib import colors
        fc=colors.to_rgba(retr_obj.color)
        fc = fc[:-1] + (0.5,) # <--- Change the alpha value of facecolor to be 0.7
        ax.annotate(retr_obj.target.name, xy=(0.11,0.08), xycoords='axes fraction',
                    ha='center', va='center', c='k', fontsize=fs*1.1, 
                    bbox={'boxstyle':'round', 'fc':fc, 'ec':'k'})
    
    ax2.axis('off')
    ax2.invert_yaxis()
    ax2.set_facecolor('none')
    ax.set(xlabel='Volume mixing ratio', ylabel='Pressure [bar]',yscale='log',xscale='log',
        ylim=(ymax,ymin),xlim=(xmin,xmax))   
    ax.tick_params(labelsize=fs)
    ax.xaxis.set_major_locator(mticker.LogLocator(numticks=999))
    ax.xaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
    ax.set_xlabel('Volume mixing ratio', fontsize=fs)
    ax.set_ylabel('Pressure [bar]', fontsize=fs)
    if 'ax' in kwargs and plotlegend==False:
        handles, labels = ax.get_legend_handles_labels()
        line_props=[]
        for handle,label in zip(handles, labels):
            line_props.append((handle.get_color(),label))
        return line_props
    else:
        leg_fs = fs*0.85
        leg=ax.legend(fontsize=leg_fs,ncol=legend_ncol,loc='upper center',
                        bbox_to_anchor=(0.5, 1.22),frameon=False)
        for lh in leg.legend_handles:
            lh.set_alpha(1)
        for line in leg.get_lines():
            line.set_linestyle('-')
        if comp_equ==True or 'retr_obj2' in kwargs or retr_obj.target.name in ['Sorg1X','Sorg20X']:
            ax.add_artist(leg2)
        if 'ax' not in kwargs:
            if add_species!=None:
                suffix += f'_{add_species}'
            fig.tight_layout()
            fig.savefig(f'{output_dir}/{prefix}VMRs{suffix}.pdf', bbox_inches='tight')
            if save_jpg:
                fig.savefig(f'{output_dir}/{prefix}VMRs{suffix}.jpg', bbox_inches='tight')
            plt.close()

def CCF_plot_all(retr_obj,ccf_species,noiserange=100,show_ACF=False,axes=None,
                 suffix='_all',save_jpg=False,ls='solid',lw=1,alpha=1,
                 legend_labels=None,fs=12,figsize=None,xlim=300,**kwargs): # plot all CCFs

    savefig = True if axes is None else False
    if ccf_species != retr_obj.species_names:
        suffix = '_few'
        #suffix ='_'
        #for species in ccf_species:
            #suffix+=species
    RVs=np.arange(-500,500,1) # km/s
    number=len(ccf_species)
    nrows=number//2+number%2
    ncols=2 if len(ccf_species) >1 else 1
    if figsize is not None:
        figsize=figsize
    else:
        figsize = (5,nrows*1.3)
    if len(ccf_species)==1:
        figname = f'CCF_{ccf_species}' if isinstance(ccf_species, list)==False else f'CCF_{ccf_species[0]}'
        figsize= (4,2.5)
    else:
        comp_dir = pathlib.Path(f'{retr_obj.output_dir}/comparison/')
        comp_dir.mkdir(parents=True, exist_ok=True)
        figname=f'CCFs{suffix}' if 'retr_obj2' not in kwargs else 'comparison/CCFs_both'

    custom_legend=False
    if legend_labels is not None:
        custom_legend=True
        leg_lines=[]
        retr_obj2=kwargs.get('retr_obj2')
        retr_objects=[retr_obj,retr_obj2]
        for r,l in zip(retr_objects,legend_labels):
            leg_lines.append(Line2D([0], [0], color=r.color,linewidth=2,linestyle='-',label=l))

    if axes is None:
        fig,axes = plt.subplots(nrows,ncols,figsize=figsize,dpi=200,sharex=True)
    for j,species_i in enumerate(ccf_species):
        if len(ccf_species)==1:
            ax=axes
            ccf_species=[ccf_species] if isinstance(ccf_species, list)==False else ccf_species
        else:
            ax=axes[j//2,j%2]
        CCF_norm,ACF_norm,SNR = retr_obj.ccf_acf_dict[species_i]
        if j%2==1:
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()

        ax.axvspan(-noiserange,noiserange,color='k',alpha=0.05)
        ax.set_xlim(-xlim,xlim)
        ax.axvline(x=0,color='k',lw=0.6,alpha=0.3)
        ax.axhline(y=0,color='k',lw=0.6,alpha=0.3)
        ax.plot(RVs,CCF_norm,color=retr_obj.color,label='CCF',lw=lw,alpha=alpha)
        if show_ACF:
            ax.plot(RVs,ACF_norm,color=retr_obj.color,label='ACF',linestyle='dashed',alpha=0.5)
            if j==0:
                ax.legend(loc='upper right',fontsize=fs)
        mathtext_label = retr_obj.species_info.loc[species_i,'mathtext_name']
        if 'retr_obj2' in kwargs: 
            retr_obj2=kwargs.get('retr_obj2')
            CCF_norm2,_,SNR2 = retr_obj2.ccf_acf_dict[species_i]
            ax.plot(RVs,CCF_norm2,color=retr_obj2.color,label='CCF',ls=ls,lw=lw,alpha=alpha)
            species_label=f'{mathtext_label}'
            if custom_legend:
                leg = axes[0,0].legend(handles=leg_lines,loc='upper center',fontsize=fs,ncol=2,bbox_to_anchor=(1,1.4))
            else:
                lines = [Line2D([0], [0], color=retr_obj.color, linewidth=2,label=retr_obj.target.name),
                        Line2D([0], [0], color=retr_obj2.color, linewidth=2,label=retr_obj2.target.name)]
                leg=axes[0,0].legend(handles=lines,fontsize=11,ncol=2,bbox_to_anchor=(1,1.4),loc='upper center')
            leg.get_frame().set_linewidth(0.0)
            leg.get_frame().set_alpha(None)
            leg.get_frame().set_facecolor((0, 0, 0, 0))
            leg.get_frame().set_edgecolor((0, 0, 0, 0))
        else:
            species_label=f'{mathtext_label}\nS/N={np.round(SNR,decimals=1)}'
        text = ax.text(0.05, 0.9, species_label,transform=ax.transAxes,fontsize=10,verticalalignment='top')
        text.set_path_effects([pe.Stroke(linewidth=2, foreground='white', alpha=0.8),
                                pe.Normal()])

    # in case of odd number, remove last plot
    if len(ccf_species)%2==1 and len(ccf_species)!=1:
        axes[-1,-1].axis('off')

    if savefig:
        fig.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        fig.supxlabel(r'$v_{\rm rad}$ [km/s]', y=-0.04,fontsize=fs)
        fig.supylabel('S/N', x=-0.04,fontsize=fs)
        fig.savefig(f'{retr_obj.output_dir}/{figname}.pdf', bbox_inches='tight')
        if save_jpg:
            fig.savefig(f'{retr_obj.output_dir}/{figname}.jpg', bbox_inches='tight')
        plt.close()
    else:
        return# axes

def residuals_species(retr_obj,check_species=[],use_equ=True):

    from retrieval import Retrieval
    from parameters import Parameters

    if check_species==[]:
        #check_species = list(retr_obj.species_info.index) # all species, first column
        leave_out= ['H2','He','13CO','C18O','C17O','H2(18)O','H2(17)O','13CH4']
        for species_i in list(retr_obj.species_info.index.values):
            if species_i not in retr_obj.species_names + leave_out:
                check_species.append(species_i)
        print('Checking ', check_species)
    if isinstance(check_species, list)==False:
        check_species=[check_species]

    retr=retr_obj
    residuals=(retr.data_flux-retr.model_flux)

    for species_i in check_species:
        
        # create retrieval object containing only species at equibilrium abundance
        hill_i = retr_obj.species_info.loc[species_i,'Hill_notation']
        equ_table = pathlib.Path(f'{path_tables}/{hill_i}.hdf5')
        parameters_spec = retr_obj.params_dict
        if equ_table.exists() and use_equ:
            suffix=' equ'
            usechem= 'equchem'
            parameters_spec.update({'C/O': retr_obj.params_dict['C/O'],
                            'Fe/H': retr_obj.params_dict['C/H']})
            ratios_free,ratios_equ = get_ratios(retr_obj,equ_too=True)
            for r,e in zip(ratios_free,ratios_equ):
                parameters_spec.update({e: retr_obj.params_dict[r]})
        else:
            suffix = ' logVMR=-5'
            usechem= 'freechem'
            for other_spec_i in retr_obj.species_names:
                parameters_spec.pop(f'log_{other_spec_i}', None)
            parameters_spec[f'log_{species_i}']=-5 # manually set abundance

        parameters_spec = Parameters({}, parameters_spec)
        parameters_spec.param_priors['log_l']=[-3,0]
        retr_spec = Retrieval(target=retr_obj.target,parameters=parameters_spec, 
                                species_names=retr_obj.species_names,Nlive=retr_obj.Nlive,
                                evtol=retr_obj.evtol,chemistry=usechem,
                                PT_type=retr_obj.PT_type,cloud_mode=retr_obj.cloud_mode)
        retr_spec.species_names = [species_i]
        retr_spec.species_pRT, retr_spec.species_hill =retr_spec.get_pRT_hill(retr_spec.species_names)
        retr_spec.radtrans_objects = retr_spec.get_radtrans_objects(for_species=species_i)
        species_flux=pRT_spectrum(retr_spec).make_spectrum()
        # folder created when initializing retrieval object, delete afterwards
        if os.path.isdir(retr_spec.output_dir) and not os.listdir(retr_spec.output_dir):  # Check if folder exists and is empty
            os.rmdir(retr_spec.output_dir)  # Remove empty folder

        figs=[]
        for part in range(retr_obj.n_parts):

            if np.nansum(residuals[part])==0: # skip empty orders
                continue

            fig,ax=plt.subplots(1,1,figsize=(6,2.5),dpi=200)
            sp_flux = species_flux[part]-np.nanmedian(species_flux[part])

            ax.plot(retr.data_wave[part],residuals[part],lw=0.8,alpha=1,c='k')
            ax.set_xlim(np.nanmin(retr.data_wave[part]),np.nanmax(retr.data_wave[part]))
            ax.set_ylim(np.nanmin([np.nanmin(residuals[part]),np.nanmin(sp_flux)]),
                        np.nanmax([np.nanmax(residuals[part]),np.nanmax(sp_flux)]))
            ax.plot(retr.data_wave[part],np.zeros_like(retr.data_wave[part]),lw=0.8,alpha=0.5,c='k')
            ax.set_ylabel('Residuals')
            ax.set_xlabel('Wavelength [nm]')
            label = f"{retr_obj.species_info.loc[species_i,'mathtext_name']}{suffix}"
            ax.plot(retr.data_wave[part],sp_flux,lw=0.8,c='orange',label=label)
            ax.legend()
            fig.tight_layout()
            figs.append(fig)

        res_dir = pathlib.Path(f'{retr_obj.output_dir}/residuals')
        if use_equ:
            res_dir = pathlib.Path(f'{retr_obj.output_dir}/residuals_equ')
        res_dir.mkdir(parents=True, exist_ok=True)
        with PdfPages(f'{retr_obj.output_dir}/residuals/residuals_{species_i}.pdf') as pdf:
            for fig in figs:
                plt.figure(fig.number)
                pdf.savefig()
                plt.close()

def plot_rk(retr_obj,show_spectrum=True,save_jpg=False): # plot function of veiling rk
    wl = retr_obj.data_wave.flatten()
    fl = retr_obj.data_flux.flatten()
    mfl = retr_obj.model_flux.flatten()
    wl_mid = np.median(retr_obj.data_wave)
    fig,ax=plt.subplots(2,1,figsize=(4.5,4),dpi=200)
    ax[0].set_ylabel(retr_obj.ylabel)
    c_veil='magenta'

    if show_spectrum:
        sf = ''
        # show spectrum without veiling for comparison
        retr_obj2 = copy.deepcopy(retr_obj) 
        #retr_obj2.parameters.params.pop('log_k_rk')
        if 'log_k_rk' in retr_obj2.parameters.params:
            retr_obj2.parameters.params['log_k_rk']=-6
            retr_obj2.parameters.params['d_rk']=0
            rk = retr_obj.params_dict['d_rk']
        elif 'T_disk' in retr_obj2.parameters.params:
            retr_obj2.parameters.params['phi_disk']=0
            rk = retr_obj.params_dict['phi_disk']
        model_wo_veiling = pRT_spectrum(retr_obj2)
        mfl_wo_veil = model_wo_veiling.make_spectrum().flatten()
        mct = model_wo_veiling.model_continuum.flatten()
        wave_range = ((wl>2321.596)&(wl<2337.568))

        alph = 0.7
        lw=0.95
        rk = np.round(rk,decimals=2)
        ax[0].plot(wl[wave_range],(fl*mct)[wave_range],c='k',label='Data',lw=lw)
        ax[0].plot(wl[wave_range],(mfl_wo_veil*mct)[wave_range],c=c_veil,label='$r_k=0$',alpha=alph,lw=lw)
        ax[0].plot(wl[wave_range],(mfl*mct)[wave_range],c=retr_obj.color,label=f'$r_k={rk}$',alpha=0.8,lw=lw)
        #fig.autofmt_xdate()
        #from matplotlib.ticker import MaxNLocator
        #ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        from matplotlib.ticker import MultipleLocator
        ax[0].xaxis.set_major_locator(MultipleLocator(3))
        leg=ax[0].legend(ncol=3,bbox_to_anchor=(0.47,1.25),loc='upper center')
        leg.get_frame().set_linewidth(0.0)
        leg.get_frame().set_facecolor('none')    # makes it fully transparent
        leg.get_frame().set_alpha(0)             # just in case

    # veiling as linear function
    if 'log_k_rk' in retr_obj.params_dict:
        rk_k = 10**retr_obj.params_dict['log_k_rk']
        rk_d = retr_obj.params_dict['d_rk']
        rk = rk_k*(wl-wl_mid)+rk_d
        k = np.round(retr_obj.params_dict['log_k_rk'],decimals=2)
        d = np.round(retr_obj.params_dict['d_rk'],decimals=2)
        slopes = 10**retr_obj.posterior['log_k_rk'][0]
        intercepts = retr_obj.posterior['d_rk'][0]
        y_med,ym1,yp1,ym2,yp2,ym3,yp3 = get_sigma123_lin_func(wl,slopes,intercepts,wl_mid)
        
        a=0.15
        ax[1].plot(wl,rk,color=c_veil)
        for (m,p) in zip([ym1,ym2,ym3],[yp1,yp2,yp3]):
            ax[1].fill_between(wl, m, p, color=c_veil, alpha=a)
        ax[1].text(0.05, 0.95, f'log $k$ = {k}\n$d$ = {d}', transform=ax.transAxes,
                ha='left', va='top')
        ax[1].set_xlabel('Wavelength [nm]')
        ax[1].set_ylabel('Veiling factor r$_k$')
        ax[1].set_xlim(np.min(wl),np.max(wl))   
    
    # veiling as blackbody
    elif 'T_disk' in retr_obj.params_dict:
        c_veil= retr_obj.color
        T_disk = retr_obj.params_dict['T_disk']
        phi_disk = retr_obj.params_dict['phi_disk']
        BB_0 = planck_lambda_um(T_disk,wl_mid*1e-3)
        BB_veil = planck_lambda_um(T_disk,wl*1e-3)
        rk = phi_disk*BB_veil/BB_0

        phi_dist = retr_obj.posterior['phi_disk'][0]
        T_dist = retr_obj.posterior['T_disk'][0]
        BB_dist = planck_lambda_um(T_dist,wl*1e-3)
        phi_dist = phi_dist[:, None].T
        rk_dist = phi_dist.T*BB_dist/BB_0
        
        # rk_dist: shape (N_samples, N_wl)
        p = np.percentile(rk_dist,
                  [50-34.13, 50+34.13, 
                   50-47.72, 50+47.72, 
                   50-49.87, 50+49.87],
                  axis=0)
        (ym1, yp1, ym2, yp2, ym3, yp3) = p

        if show_spectrum==False:
            sf = '_BB'
            wl_large = np.linspace(1,5,1000) # um
            BB_plot= phi_disk*planck_lambda_um(T_disk,wl_large)/BB_0
            ax[0].axvspan(min(wl*1e-3),max(wl*1e-3),color='k',alpha=0.05)
            ax[0].plot(wl_large,BB_plot,color='k',linestyle='dashed')
            ax[0].plot(wl*1e-3,rk,color=c_veil,lw=1.1,label='Retrieved')
            str_disk = r'$T_\mathrm{disk}$'
            #ax[0].text(0.05, 0.95, f'{str_disk} = {int(T_disk)}K', 
                    #transform=ax[0].transAxes, ha='left', va='top')
            ax[0].set_xlabel('Wavelength [um]')

            if retr_obj.name=='test_ROXs12B':
                BB_0 = planck_lambda_um(1000,wl_mid*1e-3)
                BB_veil = planck_lambda_um(1000,wl*1e-3)
                #rk_in = 0.2*BB_veil/BB_0
                wl_large = np.linspace(1,5,1000) # um
                rk_in= 0.2*planck_lambda_um(1000,wl_large)/BB_0
                ax[0].plot(wl_large,rk_in,color='r',linestyle='dashed',label='Input')
                ax[0].legend()
                
        ax[1].plot(wl,rk,color=c_veil,label='Retrieved')
        if retr_obj.name=='test_ROXs12B':
            ax[1].plot(wl_large*1e3,rk_in,color='r',linestyle='dashed',label='Input')
            ax[1].legend()
        #a=0.15
        #for (m, p) in [(ym1, yp1), (ym2, yp2), (ym3, yp3)]:
            #ax[1].fill_between(wl, m, p, color=c_veil, alpha=a)
        ax[1].fill_between(wl, ym1, yp1, color=c_veil, alpha=0.2)
        ax[1].set_xlabel('Wavelength [nm]')
        ax[1].set_ylabel('Veiling factor r$_k$')
        ax[1].set_xlim(np.min(wl),np.max(wl))   

    fig.tight_layout()
    prefix = retr_obj.callback_label if retr_obj.callback_label=='live_' else ''
    fig.savefig(f'{retr_obj.output_dir}/{prefix}veiling{sf}.pdf', bbox_inches='tight')
    if save_jpg:
        fig.savefig(f'{retr_obj.output_dir}/{prefix}veiling{sf}.jpg', bbox_inches='tight')
    plt.close()
    return

def get_sigma123_lin_func(x,slopes,intercepts,x0=0):
    y_samples = np.array([m * (x-x0) + b for m, b in zip(slopes, intercepts)])
    y_med = np.percentile(y_samples, 50, axis=0)
    ym1 = np.percentile(y_samples, 16, axis=0) # m=minus (1 sigma)
    yp1 = np.percentile(y_samples, 84, axis=0) # p=plus (1 sigma)
    ym2 = np.percentile(y_samples, 2.5, axis=0)
    yp2 = np.percentile(y_samples, 97.5, axis=0)
    ym3 = np.percentile(y_samples, 0.15, axis=0)
    yp3 = np.percentile(y_samples, 99.85, axis=0)
    return y_med,ym1,yp1,ym2,yp2,ym3,yp3

def plot_scaled_abunds(retr_obj): # plot abundances of scaled equchem
    
    fig,ax=plt.subplots(1,1,figsize=(4.5,3),dpi=200)
    spec = []
    scale,scale2 = [],[]
    merr,perr=[],[]
    me1,pe1,me2,pe2,me3,pe3 = [],[],[],[],[],[]
    for species_i in retr_obj.species_names:
        if species_i not in ['13CO','C17O','C18O','H2(18)O']:
            spec.append(retr_obj.species_info.loc[species_i,'mathtext_name'])
            scale2.append(retr_obj.params_dict[f'log_a_{species_i}'])
            m,p = retr_obj.params_dict[f'log_a_{species_i}_err']
            merr.append(abs(m))
            perr.append(p)
            m3,m2,m1,median,p1,p2,p3 = np.percentile(retr_obj.posterior[f'log_a_{species_i}'][0],[0.2,2.3,15.9,50.0,84.1,97.7,99.8])
            scale.append(median)
            me1.append(abs(m1-median))
            pe1.append(abs(p1-median))
            me2.append(abs(m2-median))
            pe2.append(abs(p2-median))
            me3.append(abs(m3-median))
            pe3.append(abs(p3-median))
        
    for i in range(0, len(spec), 2):
        ax.axvspan(i - 0.5, i + 0.5, color='lightgray', alpha=0.2, edgecolor=None)
    plt.xlim(-0.5,len(spec)+0.5)    

    j=1
    for m,p in zip([me3,me2,me1],[pe3,pe2,pe1]):
        plt.errorbar(spec,scale,yerr=[m,p],color=retr_obj.color,
                    fmt='o',markersize=0,elinewidth=4,alpha=0.1*j)
        j+=1

    plt.xticks(rotation=90) 
    plt.xlabel('Species')
    ax.axhline(0,c='k',linestyle='dashed',alpha=0.3)
    plt.ylabel(r'log$_{10}$ scaling wrt. solar')
    prefix = retr_obj.callback_label if retr_obj.callback_label=='live_' else ''
    fig.savefig(f'{retr_obj.output_dir}/{prefix}scaled_abunds.pdf', bbox_inches='tight')
    plt.close()
    return

def plot_contribution_per_species(retr_obj,fs=10,plot_legend=True,add_data=True,
                                    wH2He = False, show_continuum=False,**kwargs):

    radtrans = retr_obj.radtrans_objects[0]
    species_names = retr_obj.species_names
    species_info = retr_obj.species_info
    n_atm_layers = retr_obj.n_atm_layers
    #wl_um = retr_obj.data_wave.flatten()
    temperature = retr_obj.model_object.temperature
    gravity = 10**retr_obj.params_dict['log_g']

    mo = retr_obj.model_object
    prm = retr_obj.parameters.params
    common_params = {'temperatures':temperature,
                    'mean_molar_masses': mo.MMW,
                    'reference_gravity': gravity,
                    'return_contribution':False,
                    'additional_absorption_opacities_function': mo.give_absorption_opacity,
                    'eddy_diffusion_coefficients':mo.Kzz, # array of eddy diffusion coefficients for each pressure layer (constant)
                    'cloud_f_sed':mo.cloud_f_sed, # dictionary of f_sed for each cloud species
                    'cloud_particle_radius_distribution_std': mo.cloud_particle_size_std,
                    'cloud_fraction':prm.get('cloud_fraction', 1.0)}

    if retr_obj.chemistry=='freechem':
        VMRs = {'He':0.15*np.ones(n_atm_layers)}
        VMRs_wo_H2 = 0.15
        for species_i in species_names:
            vmr = 10**retr_obj.params_dict[f"log_{species_i}"]
            VMRs[species_i] = vmr*np.ones(n_atm_layers)
            VMRs_wo_H2 += vmr
        VMRs['H2'] = (1-VMRs_wo_H2)*np.ones(n_atm_layers)
    elif retr_obj.chemistry in ['equchem','quequchem','flexequ','varchem']:
        VMRs = retr_obj.VMR_dict
        VMR_med = {}
        for species_i, VMR_i in VMRs.items():
            VMR_med[species_i] = np.median(np.array(VMR_i), axis=0)
        VMRs = VMR_med

    def VMR_to_MF(VMR_dict):
        MMW = np.zeros(n_atm_layers)
        for species_i, VMR_i in VMR_dict.items():
            mass_i = species_info.loc[species_i,'mass']
            MMW += mass_i * VMR_i

        mass_fractions = {'MMW': MMW * np.ones(n_atm_layers)}
        for species_i, VMR_i in VMR_dict.items():  
            species_pRT_i = species_info.loc[species_i,retr_obj.pRT_key]
            mass_i = species_info.loc[species_i,'mass']
            mass_fractions[species_pRT_i] = (VMR_i * mass_i)/MMW
        return mass_fractions, MMW
    
    mf, mmw = VMR_to_MF(VMRs)
    bb_alpha = 0.5
    tot_alpha=1

    def contribution_by_species_single(radtrans, mass_fractions):

        # Full spectrum (for reference)
        if prm['emission_or_transmission']=='emission':
            wl_cm, flux, _ = radtrans.calculate_flux(mass_fractions=mass_fractions,
                                                    frequencies_to_wavelengths=True,
                                                    **common_params)
            flux *= u.erg/(u.cm**2*u.s*u.cm)

        elif prm['emission_or_transmission']=='transmission':
            R_p = ensure_quantity(prm['R_p'], u.R_jup).to(u.cm).value
            wl_cm, transit_radii_cm, _ = radtrans.calculate_transit_radii(mass_fractions=mass_fractions,
                                                            planet_radius= R_p,
                                                            reference_pressure=prm['ref_pressure'],
                                                            **common_params)
            
            R_s = ensure_quantity(prm['R_s'], u.R_sun).to(u.cm).value
            flux = (transit_radii_cm / R_s)**2 * 100 # in  %

        base_spec = flux
        contributions = {}

        for sp in mass_fractions.keys():
            # build mass fractions with ONLY this species
            mf = {}
            for k in mass_fractions:
                if k == sp:
                    mf[k] = mass_fractions[k]
                else:
                    mf[k] = 0.0 * mass_fractions[k]

            if prm['emission_or_transmission']=='emission':
                wl_cm, flux, _ = radtrans.calculate_flux(mass_fractions=mf,
                                                        frequencies_to_wavelengths=True,
                                                        **common_params)
                flux *= u.erg/(u.cm**2*u.s*u.cm)

            elif prm['emission_or_transmission']=='transmission':
                R_p = ensure_quantity(prm['R_p'], u.R_jup).to(u.cm).value
                wl_cm, transit_radii_cm, _ = radtrans.calculate_transit_radii(mass_fractions=mf,
                                                                planet_radius= R_p,
                                                                reference_pressure=prm['ref_pressure'],
                                                                **common_params)
                
                R_s = ensure_quantity(prm['R_s'], u.R_sun).to(u.cm).value
                flux = (transit_radii_cm / R_s)**2 * 100 # in  %

            if prm['flux_unit'] is not None:
                if isinstance(prm['flux_unit'], u.Quantity):
                    flux = flux.to(prm['flux_unit']).value
                elif prm['flux_unit']=='photons':
                    flux = mo.pRT_to_photon_flux(radtrans)
            contributions[sp] = flux

        wl_cm *= u.cm
        wl = wl_cm.to(prm['wavelength_unit']).value

        # only continuum flux without any species
        for sp in mass_fractions.keys():
            mf[sp] = 0.0 * mass_fractions[sp]
        if prm['emission_or_transmission']=='emission':
            wl_cm, flux, _ = radtrans.calculate_flux(mass_fractions=mf,
                                                    frequencies_to_wavelengths=True,
                                                    **common_params)
            flux *= u.erg/(u.cm**2*u.s*u.cm)

        elif prm['emission_or_transmission']=='transmission':
            R_p = ensure_quantity(prm['R_p'], u.R_jup).to(u.cm).value
            wl_cm, transit_radii_cm, _ = radtrans.calculate_transit_radii(mass_fractions=mf,
                                                            planet_radius= R_p,
                                                            reference_pressure=prm['ref_pressure'],
                                                            **common_params)
            
            R_s = ensure_quantity(prm['R_s'], u.R_sun).to(u.cm).value
            flux = (transit_radii_cm / R_s)**2 * 100 # in  %

        if prm['flux_unit'] is not None:
            if isinstance(prm['flux_unit'], u.Quantity):
                flux = flux.to(prm['flux_unit']).value
            elif prm['flux_unit']=='photons':
                flux = mo.pRT_to_photon_flux(radtrans)
        continuum_flux = flux

        return wl, base_spec, contributions, continuum_flux
    
    wl, tot, contribs, bb = contribution_by_species_single(radtrans, mf)

    if 'ax' in kwargs:
        ax=kwargs.get('ax')
    else:
        fig,ax =plt.subplots(1,1,figsize=(5.5,3.5),dpi=200)

    ax.set_ylabel(retr_obj.ylabel,fontsize=fs)
    wl_unit = prm['wavelength_unit'].to_string('latex')  # astropy quantity
    ax.set_xlabel(rf'Wavelength [{wl_unit}]', fontsize=fs)
    ax.set_xlim(np.min(wl),np.max(wl))

    if add_data:
        lam = retr_obj.data_wave.flatten()
        fl = retr_obj.data_flux.flatten()
        err = retr_obj.data_err.flatten()
        data_size=1.3
        elw=data_size/6
        data_alpha = 0.4
        ax.errorbar(lam,fl,yerr=err,fmt='o',markersize=data_size,elinewidth=elw,
                    markerfacecolor=retr_obj.color,markeredgecolor='black',
                    markeredgewidth=elw,alpha=data_alpha)

    dp_max, dp_min = [], []
    leg_y = 1.25
    additional = 2 if wH2He else 0
    ncol= (len(species_names) + 1 + additional) // 2 # max 2 in column
    def plot_contribs(tot,contribs,bb,axi):
        lines = []
        for sp, dp in contribs.items():
            if sp in ['H2','He'] and wH2He==False:
                continue
            # find matching label in species_names via pRT_name column
            match = species_info[species_info[retr_obj.pRT_key] == sp]
            if match.empty:
                continue  # sp not in table

            label = match.index[0]        # this is in species_names
            c = match.loc[label, 'color'] # or match['color'].iloc[0]
            mathtext = species_info.loc[label,'mathtext_name']
            lines.append(Line2D([0],[0],color=c,
                        linewidth=2,label=mathtext))

            axi.plot(wl, dp, label=label, c=c,lw=0.8)
            dp_max.append(np.nanmax(dp))
            dp_min.append(np.nanmin(dp))

        axi.plot(wl, tot, c='k',alpha=tot_alpha,lw=0.8)
        if show_continuum:
            axi.plot(wl, bb, c='k',linestyle='dotted',lw=3,alpha=bb_alpha)

        return lines

    lines = plot_contribs(tot,contribs,bb,ax)
    handles1 = [Line2D([0], [0], color='k',linewidth=2,alpha=tot_alpha,label='Total')]

    if add_data:
        handles1.append(Line2D([0], [0], marker='o', linestyle='none', markersize=4,
                    markerfacecolor=retr_obj.color, markeredgecolor='black', markeredgewidth=4/6,
                    color='k',alpha=data_alpha,label='Data'))
    
    if show_continuum:
        handles1.append(Line2D([0], [0], color='k',linewidth=3,linestyle='dotted',alpha=bb_alpha,
                       label='Continuum'))#r'$F_{\lambda}^{\mathrm{cont}}$')
    leg1 = ax.legend(handles=handles1, loc='upper left',
                    handlelength=1.7, frameon=False,fontsize=fs*0.9)
    if plot_legend:
        species_leg = ax.legend(handles=lines,bbox_to_anchor=(0.5, leg_y),ncol=ncol,
                                loc='upper center',frameon=False,fontsize=fs*0.9)
        ax.add_artist(leg1)

    if 'ax' not in kwargs:
        fig.tight_layout()
        name = 'contribution_species' if retr_obj.callback_label=='final_' else f'{retr_obj.callback_label}contribution_species'
        fig.savefig(f'{retr_obj.output_dir}/{name}.pdf', bbox_inches='tight')
        plt.close()
    else:
        return

def plot_opacities(retr_obj,species=None,
                        wave_range=[],lw=0.7,top_n=None,alph=0.6,
                        include_total=False,include_cont_species=False,
                        ax=None,fill_below=False):

    mode = retr_obj.parameters.params['emission_or_transmission']
    common_params = {'mode': mode,
                    'temperatures': retr_obj.model_object.temperature,
                    'mass_fractions': retr_obj.model_object.mass_fractions,
                    'mean_molar_masses': retr_obj.model_object.MMW,
                    'reference_gravity': retr_obj.model_object.gravity}

    if mode=='transmission':
        common_params['planet_radius'] = (retr_obj.params_dict['R_p']*u.R_jup).to(u.cm).value
        common_params['reference_pressure'] = retr_obj.parameters.params['ref_pressure']

    if len(retr_obj.radtrans_objects)==1:
        radtrans = retr_obj.radtrans_objects[0]
    # if in sections, calc anew, for it to be continiuous in this plot
    else:
        radtrans = Radtrans(pressures=retr_obj.pressure,
                            line_species=retr_obj.line_species,
                            rayleigh_species= ['H2', 'He'],
                            gas_continuum_contributors=retr_obj.CIA,
                            wavelength_boundaries=retr_obj.wlen_range_um, # microns
                            cloud_species=retr_obj.cloud_species_pRT,
                            line_opacity_mode=retr_obj.parameters.params['opa_mode'],
                            line_by_line_opacity_sampling=retr_obj.lbl_opacity_sampling)
        
    opacity_contributions = radtrans.calculate_contribution_spectra(**common_params)
    species_contributions = {}
    # Extract wavelength grid (same for all)
    lam, total_op, _ = opacity_contributions['Total']

    for key in ['line_species', 'rayleigh_species', 'gas_continuum_contributors']:
        if key in opacity_contributions:
            for spec, (lam, spec_op, contr) in opacity_contributions[key].items():
                spec_op = spec_op - total_op
                opacity_contributions[key][spec] = (lam, spec_op, contr)

    # Line species
    for spec, (spec_lam, spec_op, _) in opacity_contributions['line_species'].items():
        # sanity check: lam should match spec_lam; if not, interpolate
        if not np.allclose(spec_lam, lam):
            spec_op = np.interp(lam, spec_lam, spec_op)
        species_contributions[spec] = np.max(spec_op) #np.trapz(spec_op, lam)

    if species is None:
        if top_n is not None: # plot n most relevant species
            species = sorted(species_contributions, key=species_contributions.get, reverse=True)[:top_n]
        else: # plot all
            species = retr_obj.species_names

    if ax is None:
        return_ax=False
    else:
        return_ax = True

    include_contributions = []
    if include_total:
        include_contributions.append('Total')
    if 'H2' in species and 'He' in species and include_cont_species:
        include_contributions.extend(['H2 (Rayleigh)', 'He (Rayleigh)', 'H2--H2','H2--He'])
    for species_i in species:
        include_contributions.append(retr_obj.species_info.loc[species_i,retr_obj.pRT_key])

    line_species_colors = {}
    rayleigh_species_colors = {}
    mathtext_labels = []
    for spec in species:
        include_contributions.append(spec)
        if spec in ['H2','He']:
            rayleigh_species_colors[spec] = retr_obj.species_info.loc[spec,'color']
        else:
            prt_name = retr_obj.species_info.loc[spec,retr_obj.pRT_key]
            mathtext_labels.append(retr_obj.species_info.loc[spec,'mathtext_name'])
            line_species_colors[prt_name] = retr_obj.species_info.loc[spec,'color']
    
    _ = plot_opacity_contributions(
            radtrans,
            include=include_contributions,
            colors={'Total': 'k',
                'line_species': line_species_colors,
                'rayleigh_species': rayleigh_species_colors},
            line_styles={'Total': ':',
                'line_species': '-',
                'rayleigh_species': '--'},
            opacity_contributions=opacity_contributions,
            fill_below=fill_below,
            show_legend=False,
            ax=ax,
            #y_axis_scale='log',
            **common_params)
    
    fig = plt.gcf()
    if return_ax==False:
        ax = fig.axes[0]

    for line in ax.get_lines():
        line.set_linewidth(lw)
        line.set_alpha(alph)

    for a in fig.axes:
        if a.legend_ is not None:
            a.legend_.remove()
    for leg in fig.get_children():
        if isinstance(leg, matplotlib.legend.Legend):
            leg.remove()

    if return_ax==False:
        fig.set_size_inches(5.5, 4)
        handles, labels = ax.get_legend_handles_labels()
        ncols = (len(species) + 1) // 2 # max 2 in column
        leg = ax.legend(handles, mathtext_labels, ncol=ncols, loc="upper center",
                        bbox_to_anchor=(0.5, 1.4),fontsize=10)
        for legline in leg.get_lines():
            legline.set_linewidth(2)
            legline.set_alpha(1.0)
        leg.get_frame().set_linewidth(0.0)
    
    for line in ax.get_lines():

        # x data in meters -> to wavelength unit
        wl_unit = retr_obj.parameters.params['wavelength_unit']
        x = line.get_xdata()*u.m
        line.set_xdata(x.to(wl_unit).value)

        #if mode=='transmission':
            # y data in transit radii
            #Rstar_m = ensure_quantity(retr_obj.parameters.params['R_s'], u.R_sun).to(u.m).value
            #Rp_m = line.get_ydata()  # already in meters
            #depth = 100.0 * (Rp_m / Rstar_m)**2
            #line.set_ydata(depth)
            # need the baseline Rp (total!)
            #Rp_total = total_op  # in meters

            #for line in ax.get_lines():
            #    dRp = line.get_ydata()  # this is ΔRp
            #    delta_depth = 2 * Rp_total * dRp / (Rstar_m**2) * 100
            #    line.set_ydata(delta_depth)

    if return_ax==False:
        wl_unit = wl_unit.to_string('latex')  # astropy quantity
        ax.set_xlabel(rf'Wavelength [{wl_unit}]')
        if mode=='transmission':
            ax.set_ylabel("Transit depth [%]")

    if wave_range!=[]:
        ax.set_xlim(wave_range[0],wave_range[1])
    else:
        ax.set_xlim(np.nanmin(retr_obj.data_wave), np.nanmax(retr_obj.data_wave))

    if return_ax==False: 
        fig.tight_layout()  
        name = 'opacity_contributions' if retr_obj.callback_label=='final_' else f'{retr_obj.callback_label}opacity_contributions'
        plt.savefig(f'{retr_obj.output_dir}/{name}.pdf',dpi=200)
        plt.close()