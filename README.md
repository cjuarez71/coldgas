The file 'measure_coldgas.py' contains all functions to make the measurements and their respective plots. The Voigt profile models for each target is in 'Voightmodels'. Absorption parameter initial guesses and results are in 'Voigtparams' with file names 'GOTOQJ--param.dat' and 'J--VP.dat', respectively. All spectra (including SDSS) are in 'fits'. These are the main functions to make the measurements and plots:

get_data(line,Efit,lower_width,lowerMEANwidth,upper_width,upperMEANwidth,zline,wave,flux,continuum,z,IP,element,target,axs):
    'MEASURE EW FOR ABS LINES'
    
get_EW(z,info,einfo):
    'create array for EW measurements'
    
get_EWplot(data,EWstraka):
    'get the EW plot for the absorption lines'
    
get_zandv_data(z,modelspecfile,lines,IP,EWdata):
    'Returns redshift/voffset data'
    
get_voffset_values(v_zsys,ev_zsys,vcomp,e_vcomp):
    'get voffset for absorption components'
    
get_vel_vs_fnorm_plot(paramfiles,modelspecfile,lines,vZSYSdata,vZCOMPdata,emissiondata):
    'plot velocity vs normalized flux for systems with voigt profiles'
    
get_velocity_width(paramfiles,zdata,lines,modelspecfile,vdata,IPdetections):
    'find velocity width of absorption lines.'
    
get_IPvsNplot(paramfiles,IPdetections,EWdata):
    'plot IP vs column densities'
    
get_SFRvsVel_plot(veldata,sfr):
    'GET SFR VS DELTA V'
    
get_vesc_voffset_comparison(vel_widths,vZSYSdata,vZCOMPdata,IP,gotoinfo):
    'plot halo mass vs velocity offsets'
    
get_SFRvsEW(EWdata,EWrupke,sfr):
    'compare SFR and EW with this sample and rupke et al. 2005'
   
get_cf(IP):
    'plot covering fractions for EW and N'

Data from Straka et al. 2015 and Rupke et al. 2005 are in 'strakarupke_data'. It contains SFR, stellar masses, and WNaI. 

'IPdetections.txt' is the projected separations for visually confirmed absorption lines. 

'emissiondata.txt' is the gaussian absoprtion parameter initial guesses for emission lines.

'gotoqinfo.txt' contains general info for the targets (coordinates, projected separations, QSO/gal redshifts, etc.)

'zHa_zNa.txt' has absorption (based on NaI or CaII) and emission (Ha or OIII) redshifts for all targets. Targets with the same absorption and emission redshift have no absorption lines. 

'zNEW.txt' has absorption (based on NaI or CaII) and emission (Ha or OIII) redshifts for targets with visually confirmed absorption lines. 
