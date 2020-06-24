import os
import numpy as np
import scipy.stats
from sympy import *
from linetools.spectralline import AbsLine
from astropy import units as u
from glob import glob
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from linetools.spectra.xspectrum1d import XSpectrum1D
#----------------------------------------------------
#plot parameters
plt.rcParams["font.family"] = "Times New Roman"
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)

#========================================================================
###FILE IMPORTS
#========================================================================
#redshift data
dtype_z = [('Ha',float),('Na',float)]
#redshifts with visually confirmed abs lines
z = np.genfromtxt('/Users/chris/Documents/ResearchSDSU/zNEW.txt',dtype=dtype_z)

#all redshifts (including non-absorption targets & absorption targets) 
zALL = np.genfromtxt('/Users/chris/Documents/ResearchSDSU/esi_gotoq/zHa_zNa.txt',dtype=dtype_z)

#gotoq spectra
gotoqfiles = glob('/Users/chris/Documents/ResearchSDSU/gotoq/*F.fits')
gotoqfiles = np.sort(gotoqfiles)
e_gotoqfiles = glob('/Users/chris/Documents/ResearchSDSU/gotoq/*E.fits')
e_gotoqfiles = np.sort(e_gotoqfiles)

#voigt profile models
'GOTOQ___Fmodel.fits'
modelspecfile = glob('/Users/chris/Documents/ResearchSDSU/gotoq/bestfit_modelspec/*')
modelspecfile = np.sort(modelspecfile) 

#voigt profile parameters
'J___VP.dat'
paramfiles = glob('/Users/chris/Documents/ResearchSDSU/gotoq/bestfit_params/*')
paramfiles = np.sort(paramfiles)

#format 'gotoq info' file
names = ['raQSO','decQSO','zqso','zgal','b','logM','rmagQSO','lam(NaI)']
dtype = []
dtype.append( ('raQSO','U20') )
dtype.append( ('decQSO','U20') )
for name in names[2:]:
    dtype.append( (name,float))

gotoqinfo = np.genfromtxt('/Users/chris/Documents/ResearchSDSU/esi_gotoq/gotoqinfo.txt',dtype=dtype)
gotoqinfo = gotoqinfo[1:]

#all proj sep
IP = gotoqinfo['b']

#proj sep with visually confirmed abs lines
IPdetections = np.genfromtxt('/Users/chris/Documents/ResearchSDSU/IPdetections.txt',dtype=[('spec','U40'),('IP',float)])

#emission line gaussian parameters
emissiondata = np.genfromtxt('/Users/chris/Documents/ResearchSDSU/emissiondata.txt',delimiter='|',
                             dtype=[('target','U100'),('tau0',float),('bD',float),('line','U10')])

#========================================================================
#STRAKA
#========================================================================

#SFR from straka
dtype_sfr = [('ra','U100'),('dec','U100'),('distance',float),('IP',float),('flux',float),('e_flux',float),('SFRHa_Acorr',float),('logM',float),
             ('elogM',float),('ElogM',float)]
SFRstraka = np.genfromtxt('/Users/chris/Documents/ResearchSDSU/esi_gotoq/strakaSFR.tsv',delimiter='|',dtype=dtype_sfr)

#EW from straka 
dtype_ew = [('ra','U20'),('dec','U20'),('pair','U10'),('IP',float),('flagCaIIK','U10'),('EWCaIIK',float),('e_EWCaIIK',float),('flagCaIIH','U10'),('EWCaIIH',float),
            ('e_EWCaIIH',float),('flagNaID2','U10'),('EWNaID2',float),('e_EWNaID2',float),('flagNaID1','U10'),('EWNaID1',float),
            ('e_EWNaID1',float)]
EWstraka = np.genfromtxt('/Users/chris/Documents/ResearchSDSU/esi_gotoq/strakaEW.tsv',delimiter='|',dtype=dtype_ew)

#========================================================================
#RUPKE
#========================================================================

#ew and sfr from rupke
dtype_ewRupke = [('target','U100'),('z',float),('SFR',float),('EWNaI',float)]
EWrupke = np.genfromtxt('/Users/chris/Documents/ResearchSDSU/esi_gotoq/rupke_ewSFR.txt',usecols=(0,2,7,-1),dtype=dtype_ewRupke)

#----------------------------------------------------
#speed of light
c=2.99792458e5

#========================================================================
###MASTER LINES ARRAY
#========================================================================
#absortion lines
CaII1 = 3934.777
CaII2 = 3969.591
NaI1 = 5891.5833
NaI2 = 5897.5581
#emission lines
Halpha = 6564.61
OIII = 5008.24
#format lines
line_label = ['Ca II1','Ca II2','Na I1','Na I2','O III',r'$H \alpha$']
line_wave =  [ CaII1  , CaII2  , NaI1  , NaI2  , OIII  , Halpha]
line_abs =   ['yes'  ,'yes'  ,'yes'   ,'yes'   ,   'no','no']
linedtype = [('wave',float),('element','U20'),('abs?','U10')]
lines = []
for l in range(len(line_label)):
    row = (line_wave[l],line_label[l],line_abs[l])
    lines.append(row)
lines = np.array(lines,dtype=linedtype)
#========================================================================
###GENERAL FUNCTIONS
#========================================================================
def get_repeats(x):
    'find repeats in a given array'
    'x: array'
            
    _size = len(x) 
    repeated = [] 
    for i in range(_size): 
        k = i + 1
        for j in range(k, _size): 
            if x[i] == x[j] and x[i] not in repeated: 
                repeated.append(x[i]) 
    return repeated    
    
def get_nearest_value(array,value):
    'get the nearest value in a corresponding array'
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def get_gaussian(wave,tau0,zline,bD):
    'Fit gaussian for emission lines'
    c = 3e5
    g = tau0*np.exp( -(wave - zline)**2 / (zline*bD/c)**2 )
    return g

def get_tau(wave,tau0,zline,bD):
    'Fit gaussian for absorption lines. Get line intensity'
    c = 3e5
    Cf= 1
    tau = tau0*np.exp( -(wave - zline)**2 / (zline*bD/c)**2 )
    Intensity = (1 - Cf + Cf*np.exp(-tau))
    return Intensity

def get_velocity_sep(z1,z2):
    'get the relativistic velocity separation between z1,z2'
    c=2.99792458e5
    return c*((1.+z1)**2-(1.+z2)**2)/((1.+z1)**2+(1.+z2)**2)
    
def get_sigvel_sep(z1,z2,sigz1,sigz2):
    'get uncertainty in the relativistic velocity separation'
    c=2.99792458e5
    difz1=2.*c*(1.+z1)*((1.+z1)**2+(1.+z2)**2)**-1 + -2.*c*(1.+z1)*((1.+z1)**2+(1.+z2)**2)**-2
    difz2=-2.*c*(1.+z2)*((1.+z1)**2+(1.+z2)**2)**-1 + -2.*c*(1.+z2)*((1.+z1)**2+(1.+z2)**2)**-2
    return np.sqrt((difz1*sigz1)**2+(difz2*sigz2)**2)

def get_vel_trans(redshift,waves,line):
    'From JOEBVP. Transform wavelength into velocity space centered on some line'
    c=299792.458
    if (isinstance(line,int)|isinstance(line,float)):
        transline =c*(waves-(1+redshift)*line)/line/(1+redshift)
    else:
        transline=[]
        for ll in line: transline.append(c*(waves-(1+redshift)*ll)/ll/(1+redshift))
    return transline 

#========================================================================
###S/N RATIO FOR ABS LINES
#========================================================================
def get_continuum_plot_forSN(zline,z,wave,flux,continuum,lower_width,upper_width,axs,label):
    'plot continuum for S/N measurements'
    axs.plot(wave,flux,'black',drawstyle='steps-mid')
    axs.plot(wave,continuum,'red')
    
    axs.vlines(x=zline,ymin=min(flux),ymax=max(flux),color='r',linestyles='dashed',linewidth=1.5)
    axs.axvspan(zline+lower_width,zline+upper_width,color='r', alpha=0.1)    
    axs.set_xlim(zline+lower_width-15,zline+upper_width+10)
    axs.set_ylabel('Flux',size=13)
    #find y limits for continuum plot
    flux_range = flux[(zline - 5< wave)&(wave < zline + 5)]
    ymin = 0.9*abs(min(flux_range))
    ymax = 1.1*abs(max(flux_range))
    axs.set_ylim(ymin,ymax)          
    
    axs.text(0.85,0.93,str(label[:-1]), horizontalalignment='center',verticalalignment='center', transform=axs.transAxes,size=13)
    axs.grid()

def get_SNratio(z,modelspecfile,lines):
    'get the SN ratio near absorption lines'
    
    'z : REDSHIFTS'
    'modelspecfile : VOIGT SPECTRUM PROFILE FOR SYSTEM'
    'lines : MASTER LINES ARRAY'
    
    SN = []
    for i in range(len(modelspecfile)):
        
        #spectra 
        #-----------------------------------------
        targetname = os.path.split(modelspecfile[i])[1][:17]
        #F.fits
        specfile = '/Users/chris/Documents/ResearchSDSU/gotoq/' + targetname +'F.fits'
        spec = XSpectrum1D.from_file(specfile)
        spec_wave = spec.wavelength.value
        spec_flux = spec.flux.value
        spec_co = spec.co
        #E.fits
        e_specfile = '/Users/chris/Documents/ResearchSDSU/gotoq/' + targetname +'E.fits'
        e_spec = XSpectrum1D.from_file(e_specfile)
        e_spec_flux = e_spec.flux.value
        
        f, ax = plt.subplots(1,4,figsize=(15,4),sharex=False,sharey=False)
        f.text(0.5,0.95, str(targetname),horizontalalignment='center',verticalalignment='center',size=13)
        
        #empty arrays
        CaII_SN = []
        NaI_SN = []       
        
        for line,axs in zip(lines,ax):
            #reshift lines               
            zline = line['wave']*(1+z['Na'][i])
            #main limits for SN calculation
            lower_width = 10
            upper_width = 25
            #targets/lines that need limit adjustments
            if i==7:
                if line['element']=='Na I1':
                    lower_width = 12 
                    upper_width = 23
                if line['element']=='Na I2':
                    lower_width = 6
                    upper_width = 17
                if line['element']=='Ca II1':
                    lower_width = -30
                    upper_width = -15
            if i==9:
                if line['element']=='Ca II2':
                    lower_width = -20
                    upper_width = -5
                
            #plot pixel range for SN ratio calculation
            #plot_continuum = get_continuum_plot_forSN(zline,z['Na'][i],spec_wave,spec_flux,spec_co,lower_width,upper_width,axs,line['element'])
            
            #SN calculation
            #-----------------------------------------
            #set window for SN calculation (~15 angstrom)
            SN_lowerlim = get_nearest_value(spec_wave,zline + lower_width)
            SN_upperlim = get_nearest_value(spec_wave,zline + upper_width)
            SNwindow = (SN_lowerlim <= spec_wave)&(spec_wave <= SN_upperlim)
            
            #signal and noise window
            signal = spec_flux[SNwindow]
            noise = e_spec_flux[SNwindow]
            SNratio = signal/noise
            
            #add SNratio array to the respective element array
            if line['element']=='Ca II1' or line['element']=='Ca II2':
                CaII_SN.append(SNratio)
            if line['element']=='Na I1' or line['element']=='Na I2':
                NaI_SN.append(SNratio)
            
        CaII_SN = np.concatenate( (CaII_SN[0],CaII_SN[1]) )
        CaII_SN = np.median(CaII_SN)
        
        NaI_SN = np.concatenate( (NaI_SN[0],NaI_SN[1]) )
        NaI_SN = np.median(NaI_SN)   
        row = (targetname,CaII_SN,NaI_SN)
        SN.append(row)
    
    SN = np.array(SN,dtype=[('target','U40'),('CaII',float),('NaI',float)])
    return SN

'==================='
#SN ratio     
#SN = get_SNratio(z,modelspecfile,lines)
#print('Average CaII S/N:',np.mean(SN['CaII']))
#print('Average NaI  S/N:',np.mean(SN['NaI']))
'===================' 

#========================================================================
###EW MEASUREMENTS
#========================================================================
def get_data(line,Efit,lower_width,lowerMEANwidth,upper_width,upperMEANwidth,zline,wave,flux,continuum,z,IP,element,target,axs):
    'MEASURE EW FOR ABS LINES'

    'lower_width : WAVELENGTH DIFFERENCE BETWEEN LINE CENTER AND LEFT BOUND OF ABSORPTION'
    'lower_MEANwidth : AVERAGE WAVELENGTH DIFFERENCE BETWEEN LINE CENTER AND LEFT BOUND OF ABSORPTION FROM DIRECT DETECTIONS'
    'upper_width : WAVELENGTH DIFFERENCE BETWEEN LINE CENTER AND RIGHT BOUND OF ABSORPTION'
    'upper_MEANwidth : AVERAGE WAVELENGTH DIFFERENCE BETWEEN LINE CENTER AND RIGHT BOUND OF ABSORPTION FROM DIRECT DETECTIONS'
    'z : REDSHIFT OF THE ABSORPTION SYSTEM'
    'IP : PROJECTED SEPARATIONS FOR SYSTEMS'
    'element : NAME OF LINE'

    #abs lines 
    if lower_width > 0 and upper_width > 0:
        lowerlimit = zline - lower_width
        upperlimit = zline + upper_width
        
    #non abs lines. use average width from detections
    if lower_width== 0 and upper_width== 0:    
        lowerlimit = zline - lowerMEANwidth
        upperlimit = zline + upperMEANwidth
        
    #get wavelength,flux,continuum pixels for abs lines
    absRange = (lowerlimit <= wave) & (wave <= upperlimit)
    wave_absRange = wave[absRange]
    wave_absRange = wave_absRange/(1+z)
    flux_absRange = flux[absRange]
    continuum_absRange = continuum[absRange]

    #find flux weighted centroid
    tops = []
    bottoms = []    
    for i in range(len(wave_absRange)):
        top = (1 - (flux_absRange[i]/continuum_absRange[i]))*(wave_absRange[i]*(1+z))
        tops.append(top)
        
        bottom = (1 - (flux_absRange[i]/continuum_absRange[i]))
        bottoms.append(bottom)    
    flux_weighted_centroid = np.sum(tops)/np.sum(bottoms)
    
    line_deredshift = zline/(1+z)
    
    #find y limits for continuum plot 
    flux_range = flux[(line_deredshift - 5< wave/(1+z))&(wave/(1+z) < line_deredshift + 5)]
    ymin = 0.9*abs(min(flux_range))
    ymax = 1.1*abs(max(flux_range))
    axs.set_ylim(ymin,ymax)  
    
    #measure ew and sig_ew over line
    'ew'
    ew_rest = [ (1 - (flux_absRange[j]/continuum_absRange[j]))*(wave_absRange[j+1] - wave_absRange[j]) for j in range(len(wave_absRange)-1)]
    ew_rest = np.sum(ew_rest)
    
    'error ew'
    eflux = Efit/continuum
    eflux_absRange = eflux[absRange]
    variance_ew = [ ((wave_absRange[k+1] - wave_absRange[k])**2)*eflux_absRange[k]**2 for k in range(len(wave_absRange)-1)]
    variance_ew = np.sum(variance_ew)
    ew_rest_err= np.sqrt(variance_ew)
    
    #record in master array if there is obvious absorption
    if lower_width > 0 and upper_width > 0:  
        absorption_presence = 'yes'        
    if lower_width== 0 and upper_width== 0:  
        absorption_presence = 'no'
        
    #targets with absorption contamination. assign np.nan for ew and its error
    if target=='J0851+0719a' and element=='Na I 2':
        ew_rest = np.nan
        ew_rest_err = np.nan
    if target=='J1241+6332a' and element=='Ca II 2':
        ew_rest = np.nan
        ew_rest_err = np.nan

    row = (target,IP,flux_weighted_centroid,ew_rest,ew_rest_err,element,absorption_presence)
    return row

def get_continuum_plot(zline,z,wave,flux,continuum,lower_width,upper_width,axs,label):
    'get flux/continuum plot'
    axs.yaxis.set_ticks_position('both')
    axs.xaxis.set_ticks_position('both')
    axs.tick_params(direction='in',labelsize=15)
    
    line_deredshift = zline/(1+z)
    axs.plot(wave/(1+z),flux,'black',drawstyle='steps-mid')
    axs.plot(wave/(1+z),continuum,'red')
    axs.set_ylabel('Flux',size=13)    
    axs.axvspan(line_deredshift-lower_width,line_deredshift+upper_width,color='r',alpha = 0.2)        
    axs.set_xlim(line_deredshift-4,line_deredshift+4)
    axs.vlines(x=line_deredshift,ymin=min(flux),ymax=max(flux),color='r',linestyles='dashed',linewidth=1.5)
    axs.text(0.85,0.93,str(label[:-1]), horizontalalignment='center',verticalalignment='center', transform=axs.transAxes,size=13)
    
    #find y limits for continuum plot 
    flux_range = flux[(line_deredshift - 5< wave/(1+z))&(wave/(1+z) < line_deredshift + 5)]
    ymin = 0.9*abs(min(flux_range))
    ymax = 1.1*abs(max(flux_range))
    axs.set_ylim(ymin,ymax)                    

def get_EW(z,info,einfo):
    'create array for EW measurements'
    
    'info : LIST OF TARGET F.FITS'
    'einfo: LIST OF TARGET E.FITS'
    'z    : REDSHIFT OF ABSORPTION SYSTEM'
    
    #absorption lines
    NaI1 = 5891.5833
    NaI2 = 5897.5581
    CaII1 = 3934.777
    CaII2 = 3969.591

    lines = [CaII1,CaII2,NaI1,NaI2]
    
    ew_array = []
    for i in range(len(info)):
        #main data 
        sp = XSpectrum1D.from_file(info[i])
        target_name = os.path.split(info[i])[1][5:16]
        continuum = sp.co
        wave = sp.wavelength.value
        flux = sp.flux.value
        
        #Efits
        e_sp = XSpectrum1D.from_file(einfo[i])
        Efits = e_sp.flux.value
        
        f, ax = plt.subplots(4,1,figsize=(4,13),sharex=False,sharey=False)
        f.subplots_adjust(wspace=0.35)
        f.text(0.5,0.9, str(target_name)[:-1],horizontalalignment='center',verticalalignment='center',size=13)
        
        #assign lower and upper bounds for each absorption line (by eye)
        for line,axs in zip(lines,ax):
            zline = line*(1+z[i])
            
            if round(line)==3935:   #CaII1        
                lower_width = [1.4,0,0,0, 1.6,0, 1.3,0,1.3, 0.5, 2.3, 1.0, 0.7, 0.7, 0.4,0,0,0, 0.6,0, 1.0, 0.9]
                upper_width = [1.2,0,0,0, 0.7,0, 0.6,0,1.3, 0.5, 0.9, 0.6, 1.0, 0.6, 0.6,0,0,0, 0.5,0, 0.8, 0.9]
                label = 'Ca II 1'
            if round(line)==3970:   #CaII2  
                lower_width = [0.9,0,0,0, 0.7,0, 0.4,0,0, 0.5, 2.0,0, 0.5, 0.4, 0.4, 0.8, 0,0,0,0, 0.8, 1.1]
                upper_width = [1.1,0,0,0, 0.5,0, 0.4,0,0, 0.6, 0.6,0, 1.2, 0.4, 0.4, 0.8, 0,0,0,0, 0.6, 0.7]
                label = 'Ca II 2'
            if round(line)==5892:   #NaI1    
                lower_width = [1.4,0, 1.8, 0.5, 2.3,0,0,0,0, 0.7, 1.1, 2.0, 0.7, 0.4, 0.7, 1.1,0,0, 1.0,0, 1.0, 1.2]
                upper_width = [1.2,0, 1.8, 1.0, 1.1,0,0,0,0, 1.2, 1.1, 1.0, 0.7, 0.7, 0.7, 1.0,0,0, 0.6,0, 1.2, 1.0]
                label = 'Na I 1'
            if round(line)==5898:   #NaI2
                lower_width = [1.7,0, 1.2, 0.7, 1.3,0,0,0,0, 0.8, 1.1, 1.5, 0.7,0, 1.2,0,0,0, 1.0,0, 0.7, 1.4]
                upper_width = [0.9,0, 1.7, 0.7, 1.9,0,0,0,0, 0.7, 1.1, 1.1, 0.8,0, 0.5,0,0,0, 1.4,0, 0.9, 0.8]
                label = 'Na I 2'
                axs.set_xlabel('$\lambda_{rest} (\AA)$',size=13)
                
            #find the average width from direction detections
            lowerMEANwidth = np.mean(np.array(lower_width)[np.array(lower_width)!=0])
            upperMEANwidth = np.mean(np.array(upper_width)[np.array(upper_width)!=0])
                
            data_row = get_data(line,Efits,lower_width[i],lowerMEANwidth,upper_width[i],upperMEANwidth,zline,wave,
                                flux,continuum,z[i],IP[i],label,target_name,axs)
            ew_array.append(data_row)
    
            #plot continuum around absorption lines for each target
            #plot_continuum = get_continuum_plot(zline,z[i],wave,flux,continuum,lower_width[i],upper_width[i],axs,label)

    ew_array = np.array(ew_array,dtype=[('target','U100'),('IP',float),('flux_weight_centroid',float),('ew',float),('e_ew',float),('element','U10'),('abs?','U10')])
    return ew_array
        
'==================='
#EW data
#EWdata = get_EW(zALL['Na'],gotoqfiles,e_gotoqfiles)
'==================='
    
def get_EWplot(data,EWstraka):
    'get the EW plot for the absorption lines'
    
    'data     : EQUIVALENT WIDTH DATA'
    'EWstraka : EQUIVALENT WIDTH DATA FROM STRAKA'
    #absorption lines
    CaII1 = 3934.77
    CaII2 = 3969.58
    NaI1 = 5891.58
    NaI2 = 5897.55
    lines = [CaII1,CaII2,NaI1,NaI2]
    
    f, ax = plt.subplots(2,1,figsize=(8,10),sharex=True)
    ax[0].set_ylabel('$W_{CaII\lambda3934}$' + ' [$\AA$]',size=25)
    ax[0].yaxis.set_ticks_position('both')
    ax[0].xaxis.set_ticks_position('both')
    ax[0].tick_params(direction='in',labelsize=15)
    
    ax[0].vlines(x=6,ymin=-0.2,ymax=1.2,color='black',linestyles='dashed',linewidth=1,alpha=0.5)
    ax[0].hlines(y=0.2,xmin=0,xmax=14,color='black',linestyles='dashed',linewidth=1,alpha=0.5)
    ax[0].set_xlim(0,13)
    ax[0].set_ylim(-0.1,1)
    
    ax[1].set_ylabel('$W_{CaII\lambda3970}$' + ' [$\AA$]',size=25)
    ax[1].yaxis.set_ticks_position('both')
    ax[1].xaxis.set_ticks_position('both')
    ax[1].tick_params(direction='in',labelsize=15)
    ax[1].set_xlabel('Projected Separation [kpc]',size=25)
    
    ax[1].vlines(x=6,ymin=-0.2,ymax=1.2,color='black',linestyles='dashed',linewidth=1,alpha=0.5)
    ax[1].hlines(y=0.2,xmin=0,xmax=14,color='black',linestyles='dashed',linewidth=1,alpha=0.5)
    ax[1].set_xlim(0,13)
    ax[1].set_ylim(-0.01,0.5)
    
    f.subplots_adjust(hspace=0)
    f1, ax1 = plt.subplots(2,1,figsize=(8,10),sharex=True)
    ax1[0].set_ylabel('$W_{NaI\lambda5891}$' + ' [$\AA$]',size=25)
    ax1[0].yaxis.set_ticks_position('both')
    ax1[0].xaxis.set_ticks_position('both')
    ax1[0].tick_params(direction='in',labelsize=15)
    
    ax1[0].vlines(x=6,ymin=-0.1,ymax=0.8,color='black',linestyles='dashed',linewidth=1,alpha=0.5)
    ax1[0].hlines(y=0.1,xmin=0,xmax=14,color='black',linestyles='dashed',linewidth=1,alpha=0.5)
    ax1[0].set_xlim(0,13)
    ax1[0].set_ylim(-0.03,0.8)
    
    ax1[1].set_ylabel('$W_{NaI\lambda5898}$' + ' [$\AA$]',size=25)
    ax1[1].yaxis.set_ticks_position('both')
    ax1[1].xaxis.set_ticks_position('both')
    ax1[1].tick_params(direction='in',labelsize=15)
    ax1[1].set_xlabel('Projected Separation [kpc]',size=25)
    
    ax1[1].vlines(x=6,ymin=-0.1,ymax=0.8,color='black',linestyles='dashed',linewidth=1,alpha=0.5)
    ax1[1].hlines(y=0.1,xmin=0,xmax=14,color='black',linestyles='dashed',linewidth=1,alpha=0.5)
    ax1[1].set_xlim(0,13)
    ax1[1].set_ylim(-0.02,0.65)
    
    f1.subplots_adjust(hspace=0)
    
    n_sigma = 3
     
    CaII1EWdata = data[data['element']=='Ca II 1']
    CaII2EWdata = data[data['element']=='Ca II 2']
    
    #plot direct detections and 3 sigma upper limits 
    for i in range(len(CaII1EWdata)):
        'DIRECT DETECTIONS'
        if CaII1EWdata[i]['ew']>n_sigma*CaII1EWdata[i]['e_ew']:
            ax[0].errorbar(CaII1EWdata[i]['IP'],CaII1EWdata[i]['ew'],yerr=CaII1EWdata[i]['e_ew'],fmt='s',color='purple')
            
        'UPPER LIMITS'
        if CaII1EWdata[i]['ew']<n_sigma*CaII1EWdata[i]['e_ew']:
            ax[0].errorbar(CaII1EWdata[i]['IP'],n_sigma*CaII1EWdata[i]['e_ew'],yerr=0.05,uplims=True,fmt='s',markerfacecolor='none',color='purple')
            
    for j in range(len(CaII2EWdata)):
        'DIRECT DETECTIONS'
        if CaII2EWdata[j]['ew']>n_sigma*CaII2EWdata[j]['e_ew']:
            ax[1].errorbar(CaII2EWdata[j]['IP'],CaII2EWdata[j]['ew'],yerr=CaII2EWdata[j]['e_ew'],fmt='s',color='purple')
        
        'UPPER LIMITS'
        if CaII2EWdata[j]['ew']<n_sigma*CaII2EWdata[j]['e_ew']:
            ax[1].errorbar(CaII2EWdata[j]['IP'],n_sigma*CaII2EWdata[j]['e_ew'],yerr=0.05,uplims=True,fmt='s',markerfacecolor='none',color='purple')
    f.savefig('/Users/chris/Documents/ResearchSDSU/plots/ew/'+'CaIIew.pdf',bbox_inches='tight')
    
    NaI1EWdata = data[data['element']=='Na I 1']
    NaI2EWdata = data[data['element']=='Na I 2']
    
    for k in range(len(NaI1EWdata)):
        'DIRECT DETECTIONS'
        if NaI1EWdata[k]['ew']>n_sigma*NaI1EWdata[k]['e_ew']:
            ax1[0].errorbar(NaI1EWdata[k]['IP'],NaI1EWdata[k]['ew'],yerr=NaI1EWdata[k]['e_ew'],fmt='s',color='red')
        
        'UPPER LIMITS'
        if NaI1EWdata[k]['ew']<n_sigma*NaI1EWdata[k]['e_ew']:
            ax1[0].errorbar(NaI1EWdata[k]['IP'],n_sigma*NaI1EWdata[k]['e_ew'],yerr=0.05,uplims=True,fmt='s',markerfacecolor='none',color='red')
    
    for l in range(len(NaI2EWdata)):
        'DIRECT DETECTIONS'
        if NaI2EWdata[l]['ew']>n_sigma*NaI2EWdata[l]['e_ew']:
            ax1[1].errorbar(NaI2EWdata[l]['IP'],NaI2EWdata[l]['ew'],yerr=NaI2EWdata[l]['e_ew'],fmt='s',color='red')
            
        'UPPER LIMITS'
        if NaI2EWdata[l]['ew']<n_sigma*NaI2EWdata[l]['e_ew']:
            ax1[1].errorbar(NaI2EWdata[l]['IP'],n_sigma*NaI2EWdata[l]['e_ew'],yerr=0.05,uplims=True,fmt='s',markerfacecolor='none',color='red')
    f1.savefig('/Users/chris/Documents/ResearchSDSU/plots/ew/'+'NaIew.pdf',bbox_inches='tight')
        
    
    'compare with straka data'
    EWstraka = EWstraka[(EWstraka['pair']=='*')&(EWstraka['IP']>0)]
    #for s in range(len(EWstraka)):
        #CaIIH upper lims
    #    if EWstraka[s]['flagCaIIH']=='<':
    #        ax[0].errorbar(EWstraka[s]['IP'],EWstraka[s]['EWCaIIH'],yerr=0.05,uplims=True,
    #          fmt='d',markerfacecolor='none',color='purple',alpha=0.5)
        #CaIIH detections
    #    if EWstraka[s]['flagCaIIH']==' ':
    #        ax[0].errorbar(EWstraka[s]['IP'],EWstraka[i]['EWCaIIH'],yerr=EWstraka[s]['e_EWCaIIH'],
    #          fmt='d',color='purple',alpha=0.5)            
        #CaIIK upper lims
    #    if EWstraka[s]['flagCaIIK']=='<':
    #        ax[1].errorbar(EWstraka[s]['IP'],EWstraka[s]['EWCaIIK'],yerr=0.05,uplims=True,
    #          fmt='d',markerfacecolor='none',color='purple',alpha=0.5)
        #CaIIK detections
    #    if EWstraka[s]['flagCaIIK']==' ':
    #        ax[1].errorbar(EWstraka[s]['IP'],EWstraka[i]['EWCaIIK'],yerr=EWstraka[s]['e_EWCaIIK'],
    #          fmt='d',color='purple',alpha=0.5)
            
        #NaI1 upper lims
    #    if EWstraka[s]['flagNaID1']=='<':
    #        ax1[0].errorbar(EWstraka[s]['IP'],EWstraka[s]['EWNaID1'],yerr=0.05,uplims=True,
    #          fmt='d',markerfacecolor='none',color='red',alpha=0.5)
        #NaI1 detections
    #    if EWstraka[s]['flagNaID1']==' ':
    #        ax1[0].errorbar(EWstraka[s]['IP'],EWstraka[i]['EWNaID1'],yerr=EWstraka[s]['e_EWNaID1'],
    #          fmt='d',color='red',alpha=0.5) 
        #NaI2 upper lims
    #    if EWstraka[s]['flagNaID2']=='<':
    #        ax1[1].errorbar(EWstraka[s]['IP'],EWstraka[s]['EWNaID2'],yerr=0.05,uplims=True,
    #          fmt='d',markerfacecolor='none',color='red',alpha=0.5)
        #NaI2 detections
    #    if EWstraka[s]['flagNaID2']==' ':
    #        ax1[1].errorbar(EWstraka[s]['IP'],EWstraka[i]['EWNaID2'],yerr=EWstraka[s]['e_EWNaID2'],
    #          fmt='d',color='red',alpha=0.5)
    #f.legend()
        
        
'==================='
#EW plots
#EWplots = get_EWplot(EWdata,EWstraka)
'==================='

#========================================================================
###VELOCITY OFFSET FOR ZSYS
#========================================================================  
  
def get_zandv_data(z,modelspecfile,lines,IP,EWdata):
    'Returns redshift/voffset data'
    
    'z : REDSHIFTS FOR ABSORPTION SYSTEMS'
    'modelspecfile : VOIGT SPECTRUM PROFILE FOR SYSTEM'
    'IP : PROJECTED SEPARATIONS FOR SYSTEMS'
    'EWdata : EQUIVALENT WIDTH DATA'
    
    zmaster_array = []
    voffsetZSYS_data = []
    for i in range(len(modelspecfile)):
        #spectra 
        #-----------------------------------------
        targetname = os.path.split(modelspecfile[i])[1][:17]
        'match ew data to targets with voigt profiles'
        EWdata_match = EWdata[EWdata['target']==targetname[5:16]]

        #F.fits
        specfile = '/Users/chris/Documents/ResearchSDSU/gotoq/' + targetname +'F.fits'
        spec = XSpectrum1D.from_file(specfile)
        #E.fits
        e_specfile = '/Users/chris/Documents/ResearchSDSU/gotoq/' + targetname +'E.fits'
        e_spec = XSpectrum1D.from_file(e_specfile)
        
        #spec data
        spec_wave = spec.wavelength.value
        spec_flux = spec.flux.value
        e_spec_flux = e_spec.flux.value
        
        spec_cont = spec.co
        spec_norm = spec_flux/spec_cont
        
        #model spec data
        modelspec = XSpectrum1D.from_file(modelspecfile[i])
        modelspec_wave = modelspec.wavelength.value
        modelspec_norm = modelspec.flux.value
        
        #empty arrays
        zarray = []
            
        for line in lines:
            #reshift lines
            if line['abs?']=='yes':                
                #use flux weighted centroids to get redshift (based on NaI)
                zNaI1 = EWdata_match[EWdata_match['element']=='Na I 1']['flux_weight_centroid']/lines[lines['element']=='Na I1']['wave']-1
                zNaI2 = EWdata_match[EWdata_match['element']=='Na I 2']['flux_weight_centroid']/lines[lines['element']=='Na I2']['wave']-1
                redshift = np.mean((zNaI1,zNaI2))
                
                if i==4 or i==5 or i==10:
                    #use flux weighted centroids to get redshift (based on CaII)
                    zCaII1 = EWdata_match[EWdata_match['element']=='Ca II 1']['flux_weight_centroid']/lines[lines['element']=='Ca II1']['wave']-1
                    zCaII2 = EWdata_match[EWdata_match['element']=='Ca II 2']['flux_weight_centroid']/lines[lines['element']=='Ca II2']['wave']-1
                    redshift = np.mean((zCaII1,zCaII2))
        
            if line['abs?']=='no':
                redshift = z['Ha'][i]
            zline = line['wave']*(1+redshift)  

            #GAUSSIAN FITTING ON LINES
            #======================================
            #gaussian parameter arrays (initial guesses)
            if line['element']=='Ca II1':   #CaII1     
                tau0 = [0.7,0,0, 0.800, 0.430, 0.3,1.3000, 0.910, 1.400, 1.200, 0.480, 0,0,0, 0.640, 1.110]
                bD =   [6e4,0,0, 3.8e4, 3.4e4, 5e4,1.75e4, 3.4e4, 3.5e4, 2.5e4, 3.0e4, 0,0,0, 3.5e4, 2.8e4]
                label = 'Ca II'
                         
            if line['element']=='Ca II2':   #CaII2                                      
                tau0 = [0,0,0, 0.780, 0.3, 0,0.6, 0.500, 0, 0.550, 0.460, 0, 0.320,0,0.400, 0.680]
                bD =   [0,0,0, 2.6e4, 2e4, 0,3e4, 3.5e4, 0, 1.5e4, 2.0e4, 0, 4.0e4,0,3.0e4, 2.5e4]
                label = 'Ca II'                
                
            if line['element']=='Na I1':   #NaI1
                tau0 = [0.390, 0.210, 0.270, 0.300,0,0, 1.400, 0.9000, 1.1700, 0.580, 0.12, 0.210, 0.1900, 0.299, 0.400, 0.870]
                bD =   [4.8e4, 7.3e4, 2.0e4, 3.5e4,0,0, 2.5e4, 3.50e4, 3.50e4, 2.0e4, 2e4, 2.2e4,  4.50e4, 1.8e4, 2.5e4, 2.9e4]
                label = 'Na I'                
                
            if line['element']=='Na I2':   #NaI2
                tau0 = [0.280, 0.140, 0.260, 0.200,0,0, 0.600, 0.700, 0, 0.4000,0, 0.320, 0, 0.140, 0.180, 0.760]
                bD =   [3.7e4, 7.7e4, 2.1e4, 4.0e4,0,0, 1.9e4, 3.35e4,0, 2.35e4,0, 1.8e4, 0, 2.3e4, 2.5e4, 2.5e4]
                label = 'Na I'                
                
            if line['element']=='O III':   #OIII
                tau0 = [0, 1.21e-14, 5.20,0,0, 0,2e-16 ,0,0,0, 4.4e-16,0,0, 1e-15, 4.5e-16, 1.8e-16]
                bD =   [0, 6e4     , 13e4,0,0, 0,3e4   ,0,0,0, 3e4    ,0,0, 3.6e4, 3e4    , 2.5e4  ]
                label = 'O III'                
                
            if line['element']==r'$H \alpha$':   #Halpha
                tau0 = [0.93e-15, 1.41e-14, 8.00,  12.4, 2.60, 4   , 1.3e-16 , 2.6e-16, 1.4e-16, 1.4e-16, 3.8e-16, 1.65e-16, 3.40, 1.35e-15, 0.87e-15, 2e-16]
                bD =   [6.6e4   , 5.3e4   , 12e4,  11e4, 12e4, 10e4,2.5e4   , 5e4    , 5e4    , 3.5e4  , 3e4    , 2.5e4   , 11e4, 3e4     , 3e4     , 4e4  ] 
                label = r'H$\alpha$'
                
            #fit a gaussian to emission/abs lines
            #-----------------------------------------
            #these are the intial guess parameters for ESI emission/absorption
            p_initial = [tau0[i],zline,bD[i]/1e3] 

            if line['abs?']=='no':            
                'this uses SDSS spectra when ESI has no emission lines'
                if i==2 or i==3 or i==4 or i==5 or i==12:
                    #spectra data
                    specfile = '/Users/chris/Documents/ResearchSDSU/SDSSspectra/new/'+'sdss'+targetname+'F.fits'
                    spec = XSpectrum1D.from_file(specfile)
                    spec_wave = spec.wavelength.value
                    spec_flux = spec.flux.value
                    spec_cont = spec.co
                    #these are the intial guess parameters for SDSS emission
                    p_initial = [tau0[i],zline,bD[i]/1e3]  
                                      
            #use the respective gaussian function for abs/emission lines
            if line['abs?']=='yes':
                use_function = get_tau
                ydata = spec_flux/spec_cont
            if line['abs?']=='no':
                use_function = get_gaussian
                ydata = spec_flux - spec_cont
                
            #optimization for curve fit
            'create data window around lines (ESI)'
            gaussian_datafilter = (zline-5<spec_wave)&(spec_wave<zline+5)
            if line['abs?']=='no':  
                if i==2 or i==3 or i==4 or i==5 or i==12:
                    'create data window around emission lines (SDSS)'
                    gaussian_datafilter = (zline-30<spec_wave)&(spec_wave<zline+30)            
            
            #restrict gaussian center so it is close to our measured redshift
            gaussiancenter_bounds = (min(tau0),zline-1e-4,min(bD)/1e3), (max(tau0),zline+1e-4,max(bD)/1e3)  

            popt,pcov = curve_fit(use_function,spec_wave[gaussian_datafilter],ydata[gaussian_datafilter],
                                  p0=p_initial,bounds=gaussiancenter_bounds)
            perr = np.sqrt(np.diag(pcov))
            
            #best fit gaussian parameters/uncertainties
            err_tau0,err_waveOBS,err_bD = perr[0],perr[1],perr[2]
            
            #uncertainty in redshift
            err_z = np.sqrt((err_waveOBS/line['wave'])**2 + 0)     
            row = (targetname[:-1],line['element'],redshift,err_z,line['wave'],err_waveOBS)
            zarray.append(row)
            
        #this array containts z,e_z,lambdaOBS and e_lamdaOBS        
        zarray = np.array(zarray,dtype=[('target','U40'),('element','U40'),('z',float),('e_z',float),('wave',float),('e_waveOBS',float)])    
        zmaster_array.append(zarray)
        
        #voffset for zsys
        #======================================
        #absorption line redshift uncertainties
        #------------------------------------
        abs_zdata = zarray[:4]
        
        #use z/e_z from NaI b/c they are easily detected/less complex
        zABS = np.array( [float(abs_zdata[abs_zdata['element']=='Na I1']['z']),float(abs_zdata[abs_zdata['element']=='Na I2']['z'])] )            
        e_zABS = np.array( [float(abs_zdata[abs_zdata['element']=='Na I1']['e_z']),float(abs_zdata[abs_zdata['element']=='Na I2']['e_z'])] )            
        e_zABS = e_zABS[e_zABS!=np.inf]    #this filters the inf values 
        
        if i==4 or i==5 or i==7 or i==9:
            'this if statement uses zCaII measurements for J1044,J1158,J1328,J1241. They are detected better than NaI'
            zABS = np.array( [float(abs_zdata[abs_zdata['element']=='Ca II1']['z']),float(abs_zdata[abs_zdata['element']=='Ca II2']['z'])] )
            e_zABS = np.array( [float(abs_zdata[abs_zdata['element']=='Ca II1']['e_z']),float(abs_zdata[abs_zdata['element']=='Ca II2']['e_z'])] )
            e_zABS = e_zABS[e_zABS!=np.inf]
            
        'average flux weighted centroids from NaI1&NaI2 to get 1 redshift.' 
        zABS = np.mean(zABS)  
        
        'average redshift uncertainties from NaI1&NaI2 to get 1 e_redshift.' 
        e_zABS = np.mean(e_zABS)  
        
        #emission line redshift uncertainties
        #------------------------------------
        em_zdata = zarray[4:6]
        zEM = em_zdata[em_zdata['element']=='$H \\alpha$']['z']
        e_zEM = em_zdata[em_zdata['element']=='$H \\alpha$']['e_z']
        
        if i==1:
            'this target Ha is saturated. Use OIII instead'
            zEM = em_zdata[em_zdata['element']=='O III']['z']
            e_zEM = em_zdata[em_zdata['element']=='O III']['e_z']
            
        #voffset and its uncertainty 
        #-----------------------------------
        voffset_zsys = get_velocity_sep(zABS,zEM)
        e_voffset_zsys = get_sigvel_sep(zABS,zEM,e_zABS,e_zEM)
        
        row_vZSYS = (os.path.split(specfile)[-1],IP[i],zABS,e_zABS,zEM,e_zEM,voffset_zsys,e_voffset_zsys)
        
        voffsetZSYS_data.append(row_vZSYS)
        
    #build voffsetHa array
    voffsetZSYS_data = np.array(voffsetZSYS_data,dtype=[('target','U50'),('IP',float),('zsys',float),('e_zsys',float),('zem',float),('e_zem',float),('vsys',float),('e_vsys',float)])    
    return voffsetZSYS_data,zmaster_array 
        
'==================='
#redshift/voffset_zsys data     
vZSYSdata, zdata = get_zandv_data(z,modelspecfile,lines,IPdetections['IP'],EWdata)
'==================='
        
#========================================================================
###VELOCITY OFFSETS BY COMPONENT
#========================================================================

def get_voffset_values(v_zsys,ev_zsys,vcomp,e_vcomp):
    'get voffset for absorption components'
    
    'v_zsys : VELOCITY OFFSET OF ABSORPTION SYSTEM'
    'ev_zsys : VELOCITY OFFSET UNCERTAINTY OF ABSORPTION SYSTEM'
    'vcomp : VELOCITY OF THE ABSORPTION COMPONENT FROM VOIGT PROFILE'
    'e_vcomp : VELOCITY UNCERTAINTY OF THE ABSORPTION COMPONENT FROM VOIGT PROFILE'
    
    voffset_total = v_zsys + vcomp
    e_voffset_total = np.sqrt( (ev_zsys)**2 + (e_vcomp)**2)    
    return voffset_total, e_voffset_total

def get_IP_vs_voffset_plot(vdata,ylabel,color):
    'plot IP vs voffset'
    find_repeats = get_repeats(vdata['IP'])
    #multi comps
    indexNcomps = np.in1d(vdata['IP'],find_repeats,invert=False)
    vdataNcomp = vdata[indexNcomps]
    
    #1comp
    index1comp = np.in1d(vdata['IP'],find_repeats,invert=True)
    vdata1comp = vdata[index1comp]
    #plot
    plt.figure(figsize=(10,5))
    plt.errorbar(vdataNcomp['IP'],vdataNcomp['voffset'],yerr=vdataNcomp['e_voffset'],color=color,fmt='s',markerfacecolor='none')
    plt.errorbar(vdata1comp['IP'],vdata1comp['voffset'],yerr=vdata1comp['e_voffset'],color=color,fmt='s',markerfacecolor=color)
    plt.hlines(0,xmin=min(vdataNcomp['IP'])-3,xmax=max(vdataNcomp['IP'])+3,color='black',linestyle=':',alpha=1)
    plt.tick_params(direction='in',labelsize=20)
    plt.xlabel('Projected Separation (kpc)',size=25)
    plt.ylabel(ylabel,size=25)
    plt.xlim(0,13)
    plt.savefig('/Users/chris/Documents/ResearchSDSU/plots/voffset/'+'voffset'+ylabel[20:24]+'.pdf',bbox_inches='tight')

def get_voffset_component(paramfiles,zdata,vZSYS):
    'get velocity offsets from absorption components'
    
    'paramfiles : ABSORPTION PARAMETERS FROM VOIGT PROFILES'
    'zdata : REDSHIFT DATA'
    'vZSYS : VELOCITY OFFSET DATA'
    
    #set up dtype for parameter files
    dtype_paramfiles = [('specfile','U40'),('restwave',float),('zsys',float),('col',float),('sigcol',float),('bval',float),
                     ('sigbval',float),('vel',float),('sigvel',float),('zcomp',float),('trans','U10')]
    CaIIvtotal = []
    NaIvtotal = []
    
    for i in range(len(paramfiles)):
        #parameters from voigt profile
        paramdata = np.genfromtxt(paramfiles[i],delimiter='|',dtype=dtype_paramfiles,usecols=(0,1,2,3,4,5,6,7,8,18,19))[1:]
        
        #filter out data where it is 0
        'only care about the 1st transition for abs lines'
        paramdata = paramdata[paramdata['sigcol']!=0]        
        
        #filter parameter data by element
        CaIIparamdata = paramdata[paramdata['trans']=='CaII']
        NaIparamdata = paramdata[paramdata['trans']=='NaI']       
        
        v_zsys = vZSYS[i]['vsys']
        ev_zsys = vZSYS[i]['e_vsys']
        
        #velocity components for each element
        CaIIvcomp = CaIIparamdata['vel']
        CaIIe_vcomp = CaIIparamdata['sigvel']
        
        NaIvcomp = NaIparamdata['vel']
        NaIe_vcomp = NaIparamdata['sigvel']  
        
        #redshift component for each element
        CaIIzcomp = CaIIparamdata['zcomp']
        NaIzcomp  = NaIparamdata['zcomp']
        
        'this is to get the minimum velocity difference between kinematic components'
        #if len(CaIIvcomp)==2:
        #    print(CaIIvcomp[1]-CaIIvcomp[0])
            
        #if len(NaIvcomp)==2:
        #    print(NaIvcomp[1]-NaIvcomp[0])
        
        #get voffset total for CaII/NaI abs components. 
        CaIIvoffset_comp,CaIIevoffset_comp = get_voffset_values(v_zsys,ev_zsys,CaIIvcomp,CaIIe_vcomp)    
        for j in range(len(CaIIvoffset_comp)):
            rowj = (paramdata['specfile'][0],CaIIparamdata['restwave'][j],vZSYS[i]['IP'],CaIIparamdata[j]['zcomp'],
                    CaIIparamdata[j]['vel'],CaIIparamdata[j]['sigvel'],CaIIvoffset_comp[j],CaIIevoffset_comp[j])
            CaIIvtotal.append(rowj)
            
        NaIvoffset_comp,NaIevoffset_comp = get_voffset_values(v_zsys,ev_zsys,NaIvcomp,NaIe_vcomp)
        for k in range(len(NaIvoffset_comp)):
            rowk = (paramdata['specfile'][0],NaIparamdata['restwave'][k],vZSYS[i]['IP'],NaIparamdata[k]['zcomp'],
                    NaIparamdata[k]['vel'],NaIparamdata[k]['sigvel'],NaIvoffset_comp[k],NaIevoffset_comp[k])
            NaIvtotal.append(rowk)
            
    #voffset total for each element
    dtype_vtotal = [('specfile','U40'),('restwave',float),('IP',float),('zcomp',float),('vel',float),('e_vel',float),('voffset',float),('e_voffset',float)]    
    CaIIvtotal = np.array(CaIIvtotal,dtype=dtype_vtotal)
    NaIvtotal = np.array(NaIvtotal,dtype=dtype_vtotal)
      
    #plot CaII/NaI velocity offsets
    #plotCaII_IPvVoffset = get_IP_vs_voffset_plot(CaIIvtotal,'$v_{offset,z_{comp},CaII}$ (km s$^{-1}$)','purple')
    #plotNaI_IPvVoffset = get_IP_vs_voffset_plot(NaIvtotal,'$v_{offset,z_{comp},NaI}$ (km s$^{-1}$)','red')
    
    voffset_total = np.concatenate((CaIIvtotal,NaIvtotal))
    voffset_total = np.sort(voffset_total,order='specfile')
    return voffset_total

'==================='
#velocity data
#vZCOMPdata = get_voffset_component(paramfiles,zdata,vZSYSdata)
'==================='

def get_emission_params(wave,tau0,bD,zline,ydata):
    'get best fit bD and tau0 from gaussian emission fits'
    
    'ydata : FNORM FOR ABS LINES, F-FC FOR EMISSION LINES'
    
    p_initial = [tau0,zline,bD[0]]     
    popt,pcov = curve_fit(get_gaussian,wave,ydata,p0=p_initial)
    
    tau0fit = popt[0]
    bDfit = popt[2]
    
    return tau0fit,bDfit
        
#========================================================================
###VEL VS F_NORM PLOT
#========================================================================
    
def get_vel_vs_fnorm_plot(paramfiles,modelspecfile,lines,vZSYSdata,vZCOMPdata,emissiondata):
    'plot velocity vs normalized flux for systems with voigt profiles'
    
    'paramfiles : ABSORPTION PARAMETERS FROM VOIGT PROFILES'
    'modelspecfile : VOIGT SPECTRUM PROFILE FOR SYSTEM'
    'vZSYSdata : VELOCITY OFFSET DATA'
    'vZCOMPdata : ABSORPTION VELOCITY COMPONENT DATA'
    'emissiondata : EMISSION (OIII/HA) DATA FROM TARGETS'
    
    dtype_paramfiles = [('specfile','U40'),('restwave',float),('zsys',float),('zcomp',float),('trans','U10')]
    
    for i in range(len(paramfiles)):
        #parameter data
        paramdata = np.genfromtxt(paramfiles[i],delimiter='|',dtype=dtype_paramfiles,usecols=(0,1,2,18,19))[1:]
        
        #spectra import
        specfile = paramdata['specfile'][0]   
        'there are two F.fits files for gotoqj0902. use gotoqj0902b'
        if specfile=='GOTOQJ0902+1414a_F.fits':
            continue
        spec = XSpectrum1D.from_file('/Users/chris/Documents/ResearchSDSU/gotoq/'+specfile)
        spec_wave = spec.wavelength.value
        spec_flux = spec.flux.value
        spec_co = spec.co
        spec_norm = spec_flux/spec_co
        f_minus_fc = spec_flux - spec_co
        
        #model spectra import
        model_spec = XSpectrum1D.from_file(modelspecfile[i])
        model_wave = model_spec.wavelength.value
        model_norm = model_spec.flux.value
        
        #redshifts
        zsys = vZSYSdata['zsys'][i]
        
        #voffset data 
        spec_zsys = vZSYSdata[i]['target']
        vZCOMPdata1 = vZCOMPdata[vZCOMPdata['specfile']==specfile]
        zem = vZSYSdata[i]['zem']
        
        #emission data match
        emission = emissiondata[emissiondata['target']==specfile]
        
        Haemission = emission[emission['line']==r'$H \alpha$']
        OIIIemission = emission[emission['line']=='O III']
        
        f, ax = plt.subplots(6,1,figsize=(4,10),sharex=True,sharey=False)
        f.suptitle(str(specfile[:15])+' (R$_{\perp}$= '+str('%.2f'%vZCOMPdata1['IP'][0])+')'+'\n'+r'z$_{em}$ = '+str('%.5f'%zem)+
                   '\n'+'z$_{sys}$ = '+str('%.5f'%zsys)+
                   '\n'+'v$_{offset,sys}$ = '+str('%.3f'%vZSYSdata['vsys'][i])+' $\pm$ '+ str('%.3f'%vZSYSdata['e_vsys'][i])+' (km s$^{-1}$)',
                   horizontalalignment='left',verticalalignment='top',y=1.00,size=14,fontweight='bold',x=0.1)
        
        f.subplots_adjust(hspace=0)
        for j in range(len(ax)):
            
            #set up bounds/limits for abs lines
            if lines['abs?'][j]=='yes':
                z = zsys
                zline = lines['wave'][j]*(1+z)
                wavemin = zline-15
                wavemax = zline+15
                vmin = -250
                vmax = 250
                ydata = spec_norm
            
            #set up bounds/limits for emission lines
            if lines['abs?'][j]=='no':
                z = zem
                zline = lines['wave'][j]*(1+z)
                wavemin = zline-15
                wavemax = zline+15
                vmin = -250
                vmax = 250
                ylabel = 'f - f$_c$'
                
                #call in SDSS emission data. set up larget bounds/limits for emission lines 
                if vZSYSdata['target'][i][:4]=='sdss':
                    spec = XSpectrum1D.from_file('/Users/chris/Documents/ResearchSDSU/SDSSspectra/new/'
                                                 +spec_zsys)
                    
                    spec_wave = spec.wavelength.value
                    spec_flux = spec.flux.value
                    spec_co = spec.co
                    spec_norm = spec_flux/spec_co
                    f_minus_fc = spec_flux - spec_co
                    
                    wavemin = zline-25
                    wavemax = zline+25
                    vmin = -420
                    vmax = 420
                    ylabel = 'f - f$_c$ (SDSS)'
                
                ydata = f_minus_fc
                ax[4].set_ylabel(ylabel,size=15)
        
            #plot spec/model/lines  
            #----------------------
            #create window for each panel
            window = (wavemin<=spec_wave)&(spec_wave<=wavemax)
            
            #get the vcomp for each line
            vcomp_match = vZCOMPdata[vZCOMPdata['restwave']==lines['wave'][j]]
            
            #this is to plot the vcomps in the panels with the 2nd line transition
            if lines['wave'][j]==3934.777 or lines['wave'][j]==3969.591:
                vcomp_match = vZCOMPdata1[vZCOMPdata1['restwave']==3934.777]
            if lines['wave'][j]==5891.5833 or lines['wave'][j]==5897.5581:
                vcomp_match = vZCOMPdata1[vZCOMPdata1['restwave']==5891.5833]
            
            ax[j].vlines(vcomp_match['voffset'],min(ydata[window]),max(ydata[window]),'red',linestyle=':',alpha=0.5)
            
            #check saturation on abs lines for N lower limits 
            ###
            #ax[j].hlines(0.4,vmin,vmax,'green',linestyle=':')
            ###
            
            #convert line to velocity space (relative to emission redshift)
            'redshift, waves, line'
            wave_to_velocitySPEC = get_vel_trans(zem,spec_wave[window],lines['wave'][j])    

            if lines['abs?'][j]=='no':
                'plot guassian fits for emission lines'
                OIIItau0 =  OIIIemission['tau0']
                Hatau0 = Haemission['tau0']
                
                OIIIbD = OIIIemission['bD']
                HabD = Haemission['bD']
                
                spec_wavewindow = spec_wave[window]
                
                #fit gaussian to sdss emission
                '============='
                if i==2:
                    spec_wavewindow = np.linspace(min(spec_wave[window]),max(spec_wave[window]),500)
                    OIIItau0guess = 5
                    OIIItau0,OIIIbD = get_emission_params(spec_wave[window],OIIItau0guess,OIIIbD
                                                          ,zline,ydata[window])
                    
                    Hatau0guess = 8
                    Hatau0,HabD = get_emission_params(spec_wave[window],Hatau0guess,HabD
                                                          ,zline,ydata[window])                    
                if i==3:
                    spec_wavewindow = np.linspace(min(spec_wave[window]),max(spec_wave[window]),500)
                    OIIItau0guess = 0
                    OIIItau0,OIIIbD = get_emission_params(spec_wave[window],OIIItau0guess,OIIIbD
                                                          ,zline,ydata[window])
                    Hatau0guess = 13.5
                    Hatau0,HabD = get_emission_params(spec_wave[window],Hatau0guess,HabD
                                                          ,zline,ydata[window])               
                if i==4:
                    spec_wavewindow = np.linspace(min(spec_wave[window]),max(spec_wave[window]),500)
                    OIIItau0guess = 0
                    OIIItau0,OIIIbD = get_emission_params(spec_wave[window],OIIItau0guess,OIIIbD
                                                          ,zline,ydata[window])
                    Hatau0guess = 2.5
                    Hatau0,HabD = get_emission_params(spec_wave[window],Hatau0guess,HabD
                                                          ,zline,ydata[window])
                if i==5:
                    spec_wavewindow = np.linspace(min(spec_wave[window]),max(spec_wave[window]),500)
                    OIIItau0guess = 0
                    OIIItau0,OIIIbD = get_emission_params(spec_wave[window],OIIItau0guess,[0]
                                                          ,zline,ydata[window])
                    Hatau0guess = 2.5
                    Hatau0,HabD = get_emission_params(spec_wave[window],Hatau0guess,HabD
                                                          ,zline,ydata[window])
                if i==12:
                    spec_wavewindow = np.linspace(min(spec_wave[window]),max(spec_wave[window]),500)
                    OIIItau0guess = 5
                    OIIItau0,OIIIbD = get_emission_params(spec_wave[window],OIIItau0guess,OIIIbD
                                                          ,zline,ydata[window])
                    Hatau0guess = 5
                    Hatau0,HabD = get_emission_params(spec_wave[window],Hatau0guess,HabD
                                                          ,zline,ydata[window])
                '============='
                #create model velocity values for gaussian plot
                modelx_gauss = spec_wavewindow
                gauss_wave_to_velocitySPEC = get_vel_trans(zem,modelx_gauss,lines['wave'][j])    
                
                'gaussian emissions'
                OIII_gauss = get_gaussian(modelx_gauss,OIIItau0,zline,OIIIbD)
                Ha_gauss = get_gaussian(modelx_gauss,Hatau0,zline,HabD)                
                
                if lines['element'][j]=='O III':                                            
                    ax[4].plot(gauss_wave_to_velocitySPEC,OIII_gauss,'blue')
                
                if lines['element'][j]==r'$H \alpha$':                         
                    ax[5].plot(gauss_wave_to_velocitySPEC,Ha_gauss,'blue')
        
            #plot spectrum velocity space vs ydata
            ax[j].plot(wave_to_velocitySPEC,ydata[window],'black',drawstyle='steps-mid')
            
            #velocity offset value
            ax[j].vlines(vZSYSdata['vsys'][i],min(ydata[window]),max(ydata[window]),'black',linestyle='dashed')
            ax[j].set_xlim(vmin,vmax)
            ax[j].yaxis.set_ticks_position('both')
            ax[j].xaxis.set_ticks_position('both')
            
            if lines['element'][j]=='Ca II1':
                labels = '$CaII\lambda3934$'
            if lines['element'][j]=='Ca II2':
                labels = '$CaII\lambda3969$'
                
            if lines['element'][j]=='Na I1':
                labels = '$NaI\lambda5891$'
            if lines['element'][j]=='Na I2':
                labels = '$NaI\lambda5897$'
                
            if lines['abs?'][j]=='no':
                labels = lines['element'][j]
            
            if i==0:
                ax[j].annotate(labels, xy=(5,30),xycoords='axes points',size=11, ha='left', va='top',bbox=dict(boxstyle='square', fc='w'))
                    
            #plot model velocity space vs ydata
            if lines['abs?'][j]=='yes':
                wave_to_velocityMODEL = get_vel_trans(zem,model_wave[window],lines['wave'][j])
                ax[j].plot(wave_to_velocityMODEL,model_norm[window],'red')
            
            ax[j].tick_params(direction='in',labelsize=11)
            ax[5].set_xlabel('Velocity (km s$^{-1}$)',size=15)
            ax[0].set_ylabel('f$_{norm}$',size=15)
            f.savefig('/Users/chris/Documents/ResearchSDSU/plots/velocities/'+specfile[:-5]+'_vel.pdf',bbox_inches='tight')
            
'==================='
#plot velocity vs fnorm
#vel_fnorm_plot = get_vel_vs_fnorm_plot(paramfiles,modelspecfile,lines,vZSYSdata,vZCOMPdata,emissiondata)
'==================='
    
#========================================================================
###VELOCITY WIDTH OF ABS LINE
#========================================================================

def get_velocity_width(paramfiles,zdata,lines,modelspecfile,vdata,IPdetections):
    'find velocity width of absorption line.'
    
    'paramfiles : ABSORPTION PARAMETERS FROM VOIGT PROFILES'
    'zdata : REDSHIFT DATA'
    'modelspecfile : VOIGT SPECTRUM PROFILE FOR SYSTEM'
    'vdata : VELOCITY DATA'
    'IPdetections: PROJECTED SEPARATION OF SYSTEMS WITH ABS LINES'
    
    dtype_paramfiles = [('specfile','U40'),('restwave',float),('zsys',float),('col',float),('sigcol',float),('bval',float),
                     ('sigbval',float),('vel',float),('sigvel',float),('zcomp',float),('trans','U10')]
    
    vlimdata = []
    for i in range(len(modelspecfile)):  
        #impact parameter
        IP = IPdetections[i]['IP']
        
        #parameter data
        paramdata = np.genfromtxt(paramfiles[i],delimiter='|',dtype=dtype_paramfiles,usecols=(0,1,2,3,4,5,6,7,8,18,19))[1:]
        
        #spectra import
        specfile = paramdata[0]['specfile']
        spec = XSpectrum1D.from_file('/Users/chris/Documents/ResearchSDSU/gotoq/'+specfile)
        spec_wave = spec.wavelength.value
        spec_norm = spec.flux.value/spec.co   
        #model spectra import
        model_specfile = modelspecfile[i]
        model_spec = XSpectrum1D.from_file(model_specfile)
        model_wave = model_spec.wavelength.value
        model_norm = model_spec.flux.value
        
        #get absorption line data from master line array
        lines = lines[:4]
        
        for line in lines:   
            
        #f, ax = plt.subplots(4,1,figsize=(4,12),sharex=False,sharey=False)
        #f.suptitle(x=0.5,y=0.9,t=specfile+' i:'+str(i)+':'+str(IP),size=15)
        #for axs,line in zip(ax,lines):
        
            zline = line['wave']*(1+paramdata['zsys'][0])
            
            #create window for spec/mdodel
            window = (zline-3<model_wave)&(model_wave<zline+3)
            
            specnorm_window = spec_norm[window]
            specwave_window = spec_wave[window]
            
            modelnorm_window = model_norm[window]
            modelwave_window = model_wave[window]
            
            #split model/spec window array in half
            modelnorm_firsthalf = modelnorm_window[:len(modelnorm_window)//2]
            modelnorm_secondhalf = modelnorm_window[len(modelnorm_window)//2:]
            
            #find values >= ~0.95 of the continuum level
            percent_level = 0.945
            
            #find corresponding wave limits on each end of the continuum level
            #------------------------------
            #model wave limits
            flux_firsthalf = modelnorm_firsthalf[modelnorm_firsthalf>=percent_level]
            flux_secondhalf = modelnorm_secondhalf[modelnorm_secondhalf>=percent_level]
            
            lowerpoint,upperpoint = min(flux_firsthalf),min(flux_secondhalf)
         
            #make limit adjustments for J1238
            if specfile=='GOTOQJ1238+6448a_F.fits':
                if line['element']=='Ca II1':
                    lowerpoint = np.sort(flux_firsthalf)[1]    
                if line['element']=='Ca II2':    
                    lowerpoint = np.sort(flux_firsthalf)[1]    
                            
            #match the norm flux values with its wavelength       
            #------------------------------
            #model wave 
            wave1,wave2 = modelwave_window[modelnorm_window==lowerpoint],modelwave_window[modelnorm_window==upperpoint]            
            
            #this target returned 2 wave1's. This uses the wavelength that is smaller (aka lower limit)
            if specfile=='GOTOQJ1238+6448a_F.fits':
                if line['element']=='Ca II1' or line['element']=='Ca II2':
                    wave1 = wave1[0]
            
            #skip targets where there is no abs. assign np.nan
            if specfile=='GOTOQJ0902+1414a_F.fits' or specfile=='GOTOQJ0902+1414b_F.fits':
                if line['element']=='Ca II1'or line['element']=='Ca II2':
                    wave1,wave2 = np.nan,np.nan                    
            if specfile=='GOTOQJ1044+0518a_F.fits':
                if line['element']=='Na I1' or line['element']=='Na I2':
                    wave1,wave2 = np.nan,np.nan                  
            if specfile=='GOTOQJ1241+6332a_F.fits':
                if line['element']=='Ca II2':
                    wave1,wave2 = np.nan,np.nan

            #find the redshift for the lower/upper limits
            z1 = (wave1/line['wave']) - 1
            z2 = (wave2/line['wave']) - 1
            
            'test wave limits'
            #axs.vlines(wave1,0,2,'red',linestyle='dashed')
            #axs.vlines(wave2,0,2,'red',linestyle='dashed')            
            #axs.plot(spec_wave[window],spec_norm[window],'black',drawstyle='steps-mid')
            #axs.axhline(percent_level,color='gray')
            #axs.plot(modelwave_window,modelnorm_window,'red')
            #ymin = 0.6*min(modelnorm_window)
            #ymax =1.3*max(modelnorm_window)
            #axs.set_ylim(ymin,ymax)
            
            
            #find vel seperation of limits, using zHa as a reference            
            v1 = get_velocity_sep(z1,vdata[i]['zem'])
            v2 = get_velocity_sep(z2,vdata[i]['zem'])            
            delta_v = v2 - v1
            
            row = (specfile,line['wave'],float(z1),float(v1),float(z2),float(v2),float(delta_v),line['element'],IP)
            vlimdata.append(row)
                        
    vlimdata = np.array(vlimdata,dtype=[('specfile','U50'),('restwave',float),('z1',float),('v1',float),('z2',float),
                                            ('v2',float),('delta_v',float),('element','U10'),('IP',float)])
    return vlimdata
        
'==================='
#velocity width measurements
#vel_widths = get_velocity_width(paramfiles,zdata,lines,modelspecfile,vZSYSdata,IPdetections) 
'==================='
    
###PLOT IP VS V_WIDTH
#--------------------------------------
def get_deltaV_plot(widthdata):
    'plot IP vs velocity width'
    'widthdata : VELOCITY WIDTH DATA'
   
    #CaII data
    CaIIwidths = widthdata[widthdata['element']=='Ca II1']
    #NaI data
    NaIwidths = widthdata[widthdata['element']=='Na I1']
        
    #plot setup
    #-------------------
    f,ax = plt.subplots(1,1,figsize=(10,8))
    'CaII data'    
    ax.scatter(CaIIwidths['IP'],CaIIwidths['delta_v'],color='purple',marker='s')
    ax.tick_params(direction='in',labelsize=20)
    ax.set_xlabel('Projected Separation [kpc]',size=25)
    ax.set_ylabel('$\Delta v_{CaII\lambda3934}$ [km s$^{-1}$]',size=25)
    ax.legend(prop={'size':13})
    f.savefig('/Users/chris/Documents/ResearchSDSU/plots/voffset/'+'delta_vCaII.pdf',bbox_inches='tight')
    
    'NaI data'    
    f1,ax1 = plt.subplots(1,1,figsize=(10,8))
    ax1.scatter(NaIwidths['IP'],NaIwidths['delta_v'],color='red',marker='s')
    ax1.tick_params(direction='in',labelsize=20)
    ax1.set_xlabel('Projected Separation [kpc]',size=25)
    ax1.set_ylabel('$\Delta v_{NaI\lambda5891}$ [km s$^{-1}$]',size=25)
    ax1.legend(prop={'size':13})
    f1.savefig('/Users/chris/Documents/ResearchSDSU/plots/voffset/'+'delta_vNaII.pdf',bbox_inches='tight')
    
'==================='
#velocity width plot
#deltaVplot = get_deltaV_plot(vel_widths)
'==================='

#========================================================================
###COLUMN DENSITY
#========================================================================

###PLOT IP VS N
#--------------------------------------
def get_IPvsNplot(paramfiles,IPdetections,EWdata):
    'plot IP vs column densities'

    'paramfiles : ABSORPTION PARAMETERS FROM VOIGT PROFILES'
    'IPdetections: PROJECTED SEPARATION OF SYSTEMS WITH ABS LINES'
    'EWdata : EQUIVALENT WIDTH DATA FOR ABS LINES'

    dtype_paramfiles = [('specfile','U40'),('restwave',float),('zsys',float),('col',float),('sigcol',float),('trans','U10')]
    
    CaIINdata = []
    NaINdata = []
    for i in range(len(paramfiles)):
        paramdata = np.genfromtxt(paramfiles[i],delimiter='|',dtype=dtype_paramfiles,usecols=(0,1,2,3,4,19))[1:]
        target = paramdata['specfile'][0]
        
        'We only care about the 1st transition for abs lines'
        paramdata = paramdata[paramdata['sigcol']!=0]  
        
        IP = IPdetections['IP'][i]
        #filter parameter data by element & create new array to plot
        CaIIparamdata = paramdata[paramdata['trans']=='CaII']
        for c in range(len(CaIIparamdata)):
            crow = (CaIIparamdata[c]['specfile'],IP,CaIIparamdata[c]['col'],CaIIparamdata[c]['sigcol'],CaIIparamdata[c]['trans'])
            CaIINdata.append(crow)
        
        NaIparamdata = paramdata[paramdata['trans']=='NaI']
        for n in range(len(NaIparamdata)):
            nrow = (NaIparamdata[n]['specfile'],IP,NaIparamdata[n]['col'],NaIparamdata[n]['sigcol'],NaIparamdata[n]['trans'])
            NaINdata.append(nrow)
            
    dtype_N = [('specfile','U40'),('IP',float),('N',float),('e_N',float),('trans','U10')]
    CaIINdata = np.array(CaIINdata,dtype=dtype_N)
    NaINdata = np.array(NaINdata,dtype=dtype_N)
    
    CaIIEWdata = EWdata[(EWdata['element']=='Ca II 1')]
    NaIEWdata  = EWdata[(EWdata['element']=='Na I 1')]
    
    f,ax = plt.subplots(2,1,figsize=(10,8),sharex=True)
    f.subplots_adjust(hspace=0)
    nsig=3
    
    #place lower/upper limits on CaII/NaI
    #-------------------------------------
    'CAII'
    for ca in CaIINdata:
        targ = ca['specfile'][5:10]
        #place lower limits on saturated lines
        'targets with saturation. 1 absorption component'
        if targ=='J0013' or targ=='J1220' or targ=='J1241' or targ=='J1429' or targ=='J1605':
            ax[0].errorbar(ca['IP'],ca['N'],yerr=0.1,lolims=True,fmt='s',
                         markerfacecolor='none',color='purple')
        'targets with saturation. 2 absorption components. place lower limits on the saturated components'
        if targ=='J1248' or targ=='J1717':
            if ca['N']==12.928 or ca['N']==13.111:
                ax[0].errorbar(ca['IP'],ca['N'],yerr=0.1,lolims=True,fmt='s',
                         markerfacecolor='none',color='purple')
            
            else:
                ax[0].errorbar(ca['IP'],ca['N'],yerr=ca['e_N'],fmt='s',color='purple')                    
   
        if targ!='J0013' and targ!='J1220' and targ!='J1241' and targ!='J1429' and targ!='J1605' and targ!='J1248' and targ!='J1717':
            ax[0].errorbar(ca['IP'],ca['N'],yerr=ca['e_N'],fmt='s',color='purple')    
    
    ax[0].vlines(x=6,ymin=11,ymax=14,color='black',linestyles='dashed',linewidth=1,alpha=0.5)
    ax[0].hlines(y=12.5,xmin=0,xmax=14,color='black',linestyles='dashed',linewidth=1,alpha=0.5)
    ax[0].set_xlim(0,13)
    ax[0].set_ylim(11.75,13.5)
    
    ax[0].tick_params(direction='in',labelsize=15)
    ax[0].set_ylabel('$N_{CaII}$ [cm$^{-2}$]',size=25)
    ax[0].yaxis.set_ticks_position('both')
    ax[0].xaxis.set_ticks_position('both')
    
    CaIIabsline = AbsLine('CaII 3934')
    
    saveN = []    
    for j in range(len(CaIIEWdata)):
        'upper limit for CaIIN'
        if CaIIEWdata['target'][j]=='J0902+1414a':
            CaIIN_3sig = CaIIabsline.get_N_from_Wr(nsig*CaIIEWdata['e_ew'][j]*u.AA)*u.cm*u.cm
            CaIIN_3sig = np.log10(CaIIN_3sig)
        
        if CaIIEWdata['ew'][j]<nsig*CaIIEWdata['e_ew'][j]:
            CaIIN_3sig = CaIIabsline.get_N_from_Wr(nsig*CaIIEWdata['e_ew'][j]*u.AA)*u.cm*u.cm
            CaIIN_3sig = np.log10(CaIIN_3sig)
            
            row = ('GOTOQ'+CaIIEWdata['target'][j],CaIIEWdata['IP'][j],float(CaIIN_3sig),float(CaIIN_3sig),'CaII')
            saveN.append(row)
            ax[0].errorbar(CaIIEWdata['IP'][j],CaIIN_3sig,yerr=0.1,uplims=True,fmt='s',
                         markerfacecolor='none',color='purple')
                   
    'NAI'
    for na in NaINdata:
        targ = na['specfile'][5:10]
        #place lower limits on saturated lines
        if targ=='J1220':
            ax[1].errorbar(na['IP'],na['N'],yerr=0.1,lolims=True,fmt='s',
                         markerfacecolor='none',color='red')
            
        if targ=='J1241':
            if na['N']==12.781:
                ax[1].errorbar(na['IP'],na['N'],yerr=0.1,lolims=True,fmt='s',
                         markerfacecolor='none',color='red')
            else:
                ax[1].errorbar(na['IP'],na['N'],yerr=na['e_N'],fmt='s',color='red') 
        if targ!='J1220' and targ!='J1241':
            
            ax[1].errorbar(na['IP'],na['N'],yerr=na['e_N'],fmt='s',color='red')                       
    
    ax[1].vlines(x=6,ymin=11,ymax=13.3,color='black',linestyles='dashed',linewidth=1,alpha=0.5)
    ax[1].hlines(y=11.65,xmin=0,xmax=14,color='black',linestyles='dashed',linewidth=1,alpha=0.5)
    ax[1].set_xlim(0,13)
    ax[1].set_ylim(11.25,13.25)
    
    ax[1].tick_params(direction='in',labelsize=15)
    ax[1].set_ylabel('$N_{NaI}$ [cm$^{-2}$]',size=25)
    ax[1].yaxis.set_ticks_position('both')
    ax[1].xaxis.set_ticks_position('both')
    ax[1].set_xlabel('Projected Separation [kpc]',size=25)
    
    NaIabsline = AbsLine(5891.5833*u.AA)
    
    for k in range(len(NaIEWdata)):
        'upper limit for NaIN'  
        if NaIEWdata['ew'][k]<nsig*NaIEWdata['e_ew'][k]:
            NaIN_3sig = NaIabsline.get_N_from_Wr(nsig*NaIEWdata['e_ew'][k]*u.AA)*u.cm*u.cm
            NaIN_3sig = np.log10(NaIN_3sig)
            
            row = ('GOTOQ'+NaIEWdata['target'][k],NaIEWdata['IP'][k],float(NaIN_3sig),float(NaIN_3sig),'NaI')
            saveN.append(row)
            ax[1].errorbar(NaIEWdata['IP'][k],NaIN_3sig,yerr=0.1,uplims=True,fmt='s',
                         markerfacecolor='none',color='red')
        
    f.savefig('/Users/chris/Documents/ResearchSDSU/plots/N/'+'N.pdf',bbox_inches='tight')
    #saveN = np.array(saveN,dtype=[('target','U100'),('IP',float),('N',float),('e_N',float),('element','U10')])
    #return saveN

'==================='
#column density plot
Nplot = get_IPvsNplot(paramfiles,IPdetections,EWdata)
'==================='

#========================================================================
###PLOTS TO INTERPRET RESULTS
#========================================================================    

#star formation rate vs delta v
#--------------------------------------
def get_SFRvsVel_plot(veldata,sfr):
    'GET SFR VS DELTA V'
    f,ax = plt.subplots(2,1,figsize=(8,11),sharex=True)
    f.subplots_adjust(hspace=0)
    
    CaIIspearmen = []
    NaIspearmen = []
    
    for i in range(len(sfr)):
        #match sfr data to velocity data
        if i==1 or i==61:
            'incorrect matches. Skip these iterations'
            continue
        matchto_vdata = veldata[veldata['IP']==sfr['IP'][i]]
        
        if len(matchto_vdata)!=0:
            sfr_data = sfr[i]
        
            #Ha flux
            Haflux = sfr_data['flux']*1e-20
            Haflux = Haflux*10000000 
            e_Haflux = sfr_data['e_flux']*1e-20   #W/m^2
            e_Haflux = e_Haflux*10000000   #ergs/s /m^2
            
            #SFR
            SFR = sfr_data['SFRHa_Acorr']
            logSFR = np.log10(SFR)            
            sigSFR = (1/(Haflux*np.log(10)))*e_Haflux
            
            #get CaII/NaI velocity data
            CaIIdata = matchto_vdata[matchto_vdata['element']=='Ca II1']
            NaIdata = matchto_vdata[matchto_vdata['element']=='Na I1']            
            
            #filter out targets with no velocity widths
            if CaIIdata['delta_v'][0]>0:
                CaIIspearmen.append( (logSFR,CaIIdata['delta_v'][0]) )
            if NaIdata['delta_v'][0]>0:
                NaIspearmen.append( (logSFR,NaIdata['delta_v'][0]))
                
            #plot
            for C in CaIIdata:
                ax[0].errorbar(x=logSFR,y=C['delta_v'],xerr=sigSFR,color='purple',fmt='s')
                ax[0].set_ylabel(r'$\Delta v_{CaII\lambda3934}$ [km s$^{-1}$]',size=25)
                ax[0].tick_params(direction='in',labelsize=15)
                ax[0].yaxis.set_ticks_position('both')
                ax[0].xaxis.set_ticks_position('both')

            for N in NaIdata:
                ax[1].errorbar(x=logSFR,y=N['delta_v'],xerr=sigSFR,color='red',fmt='s')
                ax[1].set_ylabel(r'$\Delta v_{NaI\lambda5891}$ [km s$^{-1}$]',size=25)
                ax[1].set_xlabel(r'log SFR$_{H \alpha,CORR}$ [$M_{\odot}$ yr$^{-1}$]',size=25)
                ax[1].tick_params(direction='in',labelsize=15)
                ax[1].yaxis.set_ticks_position('both')
                ax[1].xaxis.set_ticks_position('both')

    #spearmen correlation values between SFR/delta_v for CaII/NaI
    CaIIspearmen = np.array(CaIIspearmen,dtype=[('SFR',float),('v',float)])    
    CaIIpearson = scipy.stats.pearsonr(CaIIspearmen['SFR'],CaIIspearmen['v'])

    NaIspearmen = np.array(NaIspearmen,dtype=[('SFR',float),('v',float)])
    NaIpearson = scipy.stats.pearsonr(NaIspearmen['SFR'],NaIspearmen['v'])

    f.savefig('/Users/chris/Documents/ResearchSDSU/plots/voffset/'+'v_SFR.pdf',bbox_inches='tight') 
    
'==================='
#VEL vs SFR
#plot_vel_vs_SFR = get_SFRvsVel_plot(vel_widths,SFRstraka)
'==================='

#Halo mass
#--------------------------------------                
def get_Mhalo(logMs_Msun):
    'use eq(2) from moster et al 2010 to predict halo mass'
    
    'logMs_Msun : STELLAR MASSES FROM STRAKA ET AL 2015'    

    #constants from moster et al. 
    Msun = 1.989e30
    N = 0.02817
    beta = 1.068
    gamma = 0.611
    M1 = Msun * 10**(11.899)
    
    #stellar mass
    Mstellar = Msun*10**(logMs_Msun)
    
    #numerically solve for Mhalo in eq(2)
    M = Symbol('M',real=True)
    m_equation = Eq(2*N*M*( (M/M1)**(-beta) + (M/M1)**gamma )**(-1),Mstellar)
    
    Mhalo = nsolve( [m_equation],[M],[1e40],verify=False)
    Mhalo = float(np.asarray(Mhalo[0]))
    return Mhalo/Msun

def get_vesc_voffset_comparison(vel_widths,vZSYSdata,vZCOMPdata,IP,gotoinfo):
    'plot halo mass vs velocity offsets'
    
    'vel_widths : VELOCITY WIDTH DATA'
    'vZSYSdata : VELOCITY OFFSET DATA'
    'vZCOMPdata : ABSORPTION COMPONENT VELOCITY OFFSET DATA'
    'IP : PROJECTED SEPARATIONS FROM TARGETS'
    'gotoqinfo : GENERAL INFO FROM EACH GOTOQ SYSTEM'
    
    f,ax = plt.subplots(2,1,figsize=(10,10),sharex=True)
    f.subplots_adjust(hspace=0)
    ax[0].tick_params(direction='in',labelsize=15)
    ax[1].tick_params(direction='in',labelsize=15)
    
    Msun = 1.989e30
    G = 6.67e-11    
    #get velocity width data
    NaIwidths = vel_widths[vel_widths['element']=='Na I1']
    NaIwidthmean = np.mean(NaIwidths['delta_v'][~np.isnan(NaIwidths['delta_v'])])

    CaIIwidths = vel_widths[vel_widths['element']=='Ca II1']
    CaIIwidthmean = np.mean(CaIIwidths['delta_v'][~np.isnan(CaIIwidths['delta_v'])])
    
    sortedIP = np.sort(gotoqinfo['b'])
    
    #get escape velocity curves
    #--------------------------
    #stellar mass model
    model_log10Mstellar_Msun = np.arange(min(gotoinfo['logM'])-1,max(gotoinfo['logM'])+1,0.1)
    model_Mhalo_Msun = [get_Mhalo(m) for m in model_log10Mstellar_Msun]
    model_Mhalo_Msun = np.asarray(model_Mhalo_Msun)
    
    vesc1 = np.sqrt(2*G*model_Mhalo_Msun*Msun/(13*3.085677e19))/1e3
    vesc2 = np.sqrt(2*G*model_Mhalo_Msun*Msun/(7*3.085677e19))/1e3
    vesc3 = np.sqrt(2*G*model_Mhalo_Msun*Msun/(5*3.085677e19))/1e3
    
    ax[0].plot(np.log10(model_Mhalo_Msun),vesc1,'--',color='black',label='$R_{\perp}$ = '+str(13))
    ax[0].plot(np.log10(model_Mhalo_Msun),-vesc1,'--',color='black')
    ax[0].plot(np.log10(model_Mhalo_Msun),vesc2,'-.',color='black',label='$R_{\perp}$ = '+str(7))
    ax[0].plot(np.log10(model_Mhalo_Msun),-vesc2,'-.',color='black')
    ax[0].plot(np.log10(model_Mhalo_Msun),vesc3,':',color='black',label='$R_{\perp}$ = '+str(5))
    ax[0].plot(np.log10(model_Mhalo_Msun),-vesc3,':',color='black')
    ax[0].yaxis.set_ticks_position('both')
    ax[0].xaxis.set_ticks_position('both')
    
    ax[1].plot(np.log10(model_Mhalo_Msun),vesc1,'--',color='black')
    ax[1].plot(np.log10(model_Mhalo_Msun),-vesc1,'--',color='black')
    ax[1].plot(np.log10(model_Mhalo_Msun),vesc2,'-.',color='black')
    ax[1].plot(np.log10(model_Mhalo_Msun),-vesc2,'-.',color='black')
    ax[1].plot(np.log10(model_Mhalo_Msun),vesc3,':',color='black')
    ax[1].plot(np.log10(model_Mhalo_Msun),-vesc3,':',color='black')
    ax[1].yaxis.set_ticks_position('both')
    ax[1].xaxis.set_ticks_position('both')

    for i in range(len(vZSYSdata)):
        #match vzsys data to gotoqinfo 
        gotoqinfo_match = gotoinfo[vZSYSdata['IP'][i]==gotoqinfo['b']][0]

        #get halo mass
        Mhalo_Msun = get_Mhalo(gotoqinfo_match['logM'])
        log10Mhalo_Msun = np.log10(Mhalo_Msun)
            
        #plot special targets
        if gotoqinfo_match['raQSO']=='11:58:23':
            ax[0].errorbar(log10Mhalo_Msun,vZSYSdata[i]['vsys'],yerr=CaIIwidths[i]['delta_v'],marker='D',
              color='black',markerfacecolor='none',markersize=10)
            ax[1].errorbar(log10Mhalo_Msun,vZSYSdata[i]['vsys'],yerr=NaIwidths[i]['delta_v'],marker='D',
              color='black',markerfacecolor='none',markersize=10,label='GOTOQJ1158+3907',linestyle = 'None')
            
        if gotoqinfo_match['raQSO']=='12:38:47':
            ax[0].errorbar(log10Mhalo_Msun,vZSYSdata[i]['vsys'],yerr=CaIIwidths[i]['delta_v'],marker='p',
              color='blue',markerfacecolor='none',markersize=12)
            ax[1].errorbar(log10Mhalo_Msun,vZSYSdata[i]['vsys'],yerr=NaIwidths[i]['delta_v'],marker='p',
              color='blue',markerfacecolor='none',markersize=12,label='GOTOQJ1238+6448',linestyle = 'None')
        
        #plot all other systems
        if gotoqinfo_match['raQSO']!='11:58:23' and gotoqinfo_match['raQSO']!='12:38:47':
            ax[0].errorbar(log10Mhalo_Msun,vZSYSdata[i]['vsys'],yerr=CaIIwidths[i]['delta_v'],marker='o',color='purple')
            ax[0].axhline(0,color='black',linestyle=':',alpha=0.1)
            ax[0].set_ylabel('$\Delta v_{CaII\lambda3934}$ [km/s]',size=25)
            ax[0].set_xlim(10,12.5)
            ax[0].set_ylim(-650,650)
            ax[0].legend(prop={'size':15})
            
            ax[1].errorbar(log10Mhalo_Msun,vZSYSdata[i]['vsys'],yerr=NaIwidths[i]['delta_v'],marker='o',color='red')
            ax[1].axhline(0,color='black',linestyle=':',alpha=0.1)
            ax[1].set_xlabel('$log_{10} (M_H/M_{\odot})$',size=25)
            ax[1].set_ylabel('$\Delta v_{NaI\lambda5891}$ [km/s]',size=25)
            ax[1].set_xlim(10,12.5)
            ax[1].set_ylim(-650,650)
            
            ax[1].legend(prop={'size':15})
    f.savefig('/Users/chris/Documents/ResearchSDSU/plots/voffset/'+'vesc.pdf',bbox_inches='tight')
    
'==================='
#Compare voffset_zsys with escape velocities of halo mass
#vesc = get_vesc_voffset_comparison(vel_widths,vZSYSdata,vZCOMPdata,IP,gotoqinfo)
'==================='

#gather EW measurements from previous studies
def get_SFRvsEW(EWdata,EWrupke,sfr):
    'compare SFR and EW with this sample and rupke et al. 2005'
    
    'EWdata : EQUIVALENT WIDTH DATA'
    'EWrupke : EQUIVALENT WIDTH DATA FROM RUPKE ET AL. 2005'
    'sfr : STAR FORMATION RATE DATA FROM STRAKA ET AL. 2015'
    
    f,ax = plt.subplots(1,1,figsize=(10,10),sharex=True)
    ax.plot(np.log10(EWrupke['SFR']),EWrupke['EWNaI'],'s',color='black',markerfacecolor='none',label='Rupke (et al. 2005)')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.set_xlabel(r'log SFR [$M_{\odot}$ yr$^{-1}$]',size=25)
    ax.set_ylabel('$W_{Na 5891}$' + ' [$\AA$]',size=25)
    ax.tick_params(direction='in',labelsize=15)
    
    spearmen = []
    for i in range(len(sfr)):
        #match sfr data to EWdata
        if i==1 or i==61:
            'incorrect matches. Skip these iterations'
            continue
        matchto_EWdata = EWdata[EWdata['IP']==sfr['IP'][i]]
        
        nsig = 3
        if len(matchto_EWdata)!=0:
            logSFR = np.log10(sfr[i]['SFRHa_Acorr'])
            NaIEWdata = matchto_EWdata[matchto_EWdata['element']=='Na I 1'][0]
            
            #place upper limits on our NaI EW
            if NaIEWdata['ew']<nsig*NaIEWdata['e_ew']:
                NaIEW3sig = nsig*NaIEWdata['e_ew']
                ax.errorbar(logSFR,NaIEW3sig,yerr=0.1,uplims=True,fmt='s',
                             markerfacecolor='none',color='red')
                spearmen.append( (logSFR,NaIEW3sig) )
                
            if NaIEWdata['ew']>nsig*NaIEWdata['e_ew']:
                ax.errorbar(logSFR,NaIEWdata['ew'],yerr=NaIEWdata['e_ew'],fmt='s',
                             markerfacecolor='red',color='red')    
                spearmen.append( (logSFR,NaIEWdata['ew']) )
    ax.legend(prop={'size':20},frameon=False)
    f.savefig('/Users/chris/Documents/ResearchSDSU/plots/ew/'+'EWvsSFR.pdf',bbox_inches='tight')
    
    rupke = []
    for r in EWrupke:
        rupke.append( (np.log10(r['SFR']),r['EWNaI']) )
    rupke = np.array(rupke,dtype=[('SFR',float),('ew',float)])    
    thisdata = np.array(spearmen,dtype=[('SFR',float),('ew',float)])    
    #find spearman correlation values 
    spearmen = np.concatenate( (rupke,thisdata) )
    spearmen = scipy.stats.spearmanr(spearmen['SFR'],spearmen['ew'])
    spearmen_ourdata = scipy.stats.pearsonr(thisdata['SFR'],thisdata['ew'])
    spearmen_rupke = scipy.stats.pearsonr(rupke['SFR'],rupke['ew'])
    
'==================='
#compare these NaEW with Rupke NaEW
#plotSFRvsEW = get_SFRvsEW(EWdata,EWrupke,SFRstraka)
'==================='

#========================================================================
###COVERING FRACTIONS
#======================================================================== 

def get_wilson_score(z,p,n):
    'get binomial wilson score'
    
    'z : Z-SCORE OF THE CONFIDENCE INTERVAL'
    'p : THE COVERING FRACTION'
    'n : TOTAL NUMBER OF MEASUREMENTS'
    
    denominator = 1 + (z**2)/n
    centre_adjusted_probability = p + (z**2)/(2*n)
    adjusted_standard_deviation = np.sqrt((p*(1 - p)/n + (z**2) / (4*n**2)))
    
    lower_bound = (centre_adjusted_probability - z*adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z*adjusted_standard_deviation) / denominator
    return (lower_bound, upper_bound)

def get_ecf(z,p,n):
    'get covering fraction uncertainties'
    
    'z : Z-SCORE OF THE CONFIDENCE INTERVAL'
    'p : THE COVERING FRACTION'
    'n : TOTAL NUMBER OF MEASUREMENTS'
    
    #the location of the error bars
    lower_upperBounds = np.asarray(get_wilson_score(z,p,n))
    
    elower_cf = p-lower_upperBounds[0]
    eupper_cf = lower_upperBounds[1]-p
    e_cf = np.asarray( (elower_cf,eupper_cf) )
    e_cf.shape = (2,1)
    return e_cf

def get_significance(below,e_below,above,e_above):
    'get statistical significance between covering fractions'
    
    'below : COVERING FRACTION AT R < 6 KPC'
    'e_below : COVERING FRACTION ERROR AT R < 6 KPC'
    'above : COVERING FRACTION AT R > 6 KPC'
    'e_above : COVERING FRACTION ERROR AT R > 6 KPC'
    
    sigtot = np.sqrt(e_below**2 + e_above**2)
    return (above - below)/sigtot

def get_cf(IP):
    'plot covering fractions for EW and N'
    f,ax = plt.subplots(2,1,figsize=(5,8),sharex=True)
    f.subplots_adjust(hspace=0)
    
    #z-score for 68% confidence interval
    zalpha_2 = 0.9945 
    
    #total data points
    N = 22
    
    #covering fractions in bins R < 6kpc, R > 6kpc
    'WCaII'
    EW_CaIIcf_below6 = 8/13
    eEW_CaIIcf_below6 = get_ecf(zalpha_2,EW_CaIIcf_below6,N)    
    
    EW_CaIIcf_above6 = 6/6
    eEW_CaIIcf_above6 = get_ecf(zalpha_2,EW_CaIIcf_above6,N) 
    signifWCaII = get_significance(EW_CaIIcf_below6,eEW_CaIIcf_below6[1],EW_CaIIcf_above6,eEW_CaIIcf_above6[0])
    
    'WNaI'
    EW_NaIcf_below6 = 8/16
    eEW_NaIcf_below6 = get_ecf(zalpha_2,EW_NaIcf_below6,N)    
    
    EW_NaIcf_above6 = 5/6
    eEW_NaIcf_above6 = get_ecf(zalpha_2,EW_NaIcf_above6,N)
    signifWNaI  =get_significance(EW_NaIcf_below6,eEW_NaIcf_below6[1],EW_NaIcf_above6,eEW_NaIcf_above6[0])   

    'NCaII'
    N_CaII_below6 = 8/16
    eN_CaII_below6 =  get_ecf(zalpha_2,N_CaII_below6,N)    
    
    N_CaII_above6 = 7/8
    eN_CaII_above6 =  get_ecf(zalpha_2,N_CaII_above6,N)        
    signifNCaII = get_significance(N_CaII_below6,eN_CaII_below6[1],N_CaII_above6,eN_CaII_above6[0])
    
    'NNaI'
    N_NaI_below6 = 10/18
    eN_NaI_below6 = get_ecf(zalpha_2,N_NaI_below6,N)    
    
    N_NaI_above6 = 8/8
    eN_NaI_above6 = get_ecf(zalpha_2,N_NaI_above6,N)        
    signifNNaI = get_significance(N_NaI_below6,eN_NaI_below6[1],N_NaI_above6,eN_NaI_above6[0])
   
    #average the projected separations in each bin
    avg_IPbelow6 = np.mean(IPdetections['IP'][IPdetections['IP']<=6])
    avg_IPabove6 = np.mean(IPdetections['IP'][IPdetections['IP']>=6])    
    
    'cfW plot'
    #below 6 cf
    ax[0].errorbar(avg_IPbelow6,EW_CaIIcf_below6,yerr=eEW_CaIIcf_below6,color='purple',fmt='s',label='$CaII\lambda3934$')
    ax[0].errorbar(avg_IPbelow6,EW_NaIcf_below6,yerr=eEW_NaIcf_below6,color='red',fmt='s',label='$NaI\lambda5891$')
    
    ax[0].tick_params(direction='in',labelsize=15)
    ax[0].yaxis.set_ticks_position('both')
    ax[0].xaxis.set_ticks_position('both')
    ax[0].legend(prop={'size':15})
    
    #above 6 cf
    ax[0].errorbar(avg_IPabove6,EW_CaIIcf_above6,yerr=eEW_CaIIcf_above6,color='purple',fmt='s')
    ax[0].errorbar(avg_IPabove6,EW_NaIcf_above6,yerr=eEW_NaIcf_above6,color='red',marker='s')
    ax[0].set_ylabel('$c_{f,W}$',size=25)
    
    'cfN plot'
    #below 6 cf
    ax[1].errorbar(avg_IPbelow6,N_CaII_below6,yerr=eN_CaII_below6,color='purple',fmt='s')
    ax[1].errorbar(avg_IPbelow6,N_NaI_below6 ,yerr=eN_NaI_below6,color='red',fmt='s')
    
    #above 6 cf
    ax[1].errorbar(avg_IPabove6,N_CaII_above6,yerr=eN_CaII_above6,color='purple',fmt='s')
    ax[1].errorbar(avg_IPabove6,N_NaI_above6 ,yerr=eN_NaI_above6,color='red',fmt='s')
    
    ax[1].yaxis.set_ticks_position('both')
    ax[1].xaxis.set_ticks_position('both')
    ax[1].tick_params(direction='in',labelsize=15)
    ax[1].set_xlabel('$R_{\perp}$ [kpc]',size=25)
    ax[1].set_ylabel('$c_{f,N}$',size=25)
    f.savefig('/Users/chris/Documents/ResearchSDSU/plots/'+'cf.pdf',bbox_inches='tight')
    
'==================='
#plot covering fractions
#cf = get_cf(IPdetections)
'==================='
    