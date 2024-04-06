import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dust_extinction.parameter_averages import CCM89, F99
from specutils.spectra import Spectrum1D, SpectralRegion
from specutils.fitting import fit_generic_continuum
from astropy import units as u
import warnings
warnings.filterwarnings("ignore")

#plt.plot(xfit,yfit)
#plt.plot(data_new[:,0],data_new[:,1])
filename=sys.argv[1]
redshift=float(sys.argv[2])
lambda_pm=int(sys.argv[5])
Rv=3.1
Ebv=float(sys.argv[3])
ext = F99(Rv=Rv)
def get_data(filename):
    if filename[-5:]=='ascii' or filename[-3:]=='txt' or filename[-3:]=='dat':
        data=np.loadtxt(filename)
        wavelength=data[:,0]/(1+redshift)
        flux=data[:,1]
        dim=np.shape(data)[1]
        if dim>2:
            error=data[:,2]
        else:
            error=0.05*flux
    elif filename[-4:]=='fits':
        f = fits.open(filename)  
        specdata = f[0].data 
        wavelength=specdata[0]/(1+redshift)
        flux=specdata[1]
        error=specdata[2]
    elif filename[-3:]=='csv':
        data=pd.read_csv(filename)
        wavelength=np.empty(0)
        flux=np.empty(0)
        error=np.empty(0)
        for i in range(len(data)):
            wavelength=np.append(wavelength,data.iloc[:,0][i]/(1+redshift))
            flux=np.append(flux,data.iloc[:,1][i])
            error=np.append(error,data.iloc[:,2][i])
    return wavelength,flux,error
def get_wavelength_vals(filename):
    l,f,e=get_data(filename)
    l_ext=f/ext.extinguish(l/1e4,Ebv=Ebv)
    line_value=int(sys.argv[4])
    xfit=l[np.where((l>line_value-lambda_pm)&(l<line_value+lambda_pm))]
    yfit=f[np.where((l>line_value-lambda_pm)&(l<line_value+lambda_pm))]
    region=SpectralRegion((line_value-(lambda_pm/5))*u.nm, (line_value+(lambda_pm/5))*u.nm)
    spectrum = Spectrum1D(flux=yfit*u.Jy, spectral_axis=xfit*u.nm)
    g1_fit = fit_generic_continuum(spectrum,exclude_regions=region)
    y_continuum_fitted = g1_fit(xfit*u.nm)
    ynew=np.array(y_continuum_fitted)
    y=yfit-ynew
    x=3e5*(xfit-line_value)/(line_value)
    yerr=e[np.where((l>line_value-lambda_pm)&(l<line_value+lambda_pm))]   
    return x,y,yerr    
x,y,e=get_wavelength_vals(filename)
plt.plot(x,y)
plt.show()

