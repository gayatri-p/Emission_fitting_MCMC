#importing libraries
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dust_extinction.parameter_averages import CCM89, F99
from specutils.spectra import Spectrum1D, SpectralRegion
from specutils.fitting import fit_generic_continuum
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
import warnings
from scipy.optimize import leastsq
import argparse
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='Fit spectral lines')
#Needed arguments
parser.add_argument('filename', help='filename of spectrum that you want to fit')
#Default values
parser.add_argument('-z', '--redshift', default=0, type=float,
                    help='redshift of object to be used to correct wavelengths to rest wavelengths')
parser.add_argument('--correction', default=0,type=float,
                    help='Dust correction term')
parser.add_argument('--pm', default=0,type=float,
                    help='Range from rest wavelength over which to perform fitting')
parser.add_argument('--wavelength', default=0,type=float,
                    help='Rest wavelength of line')
parser.add_argument('--guess',help='Best guess file')
parser.add_argument('--plot_limits', default=10000,type=int,
                    help='Limit of plots')
parser.add_argument('--continuum_sub', default=True,
                    help='Do continuum_sub')
parser.add_argument('--lorentzian', default=False,
                    help='Input True if Lorentzian guess')
args = parser.parse_args()
def model_4gauss(theta):
    '''
    Fit 4 Gaussian model
    Inputs:
    Parameters
    Outputs:
    4-gaussian Model
    '''
    a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12 = theta
    model = a1*np.exp(-(x-a2)**2/(2*a3**2))+a4*np.exp(-(x-a5)**2/(2*a6**2))+a7*np.exp(-(x-a8)**2/(2*a9**2))+a10*np.exp(-(x-a11)**2/(2*a12**2))
    return model
def model_3gauss(theta):
    '''
    Fit 3 Gaussian model
    Inputs:
    Parameters
    Outputs:
    3-gaussian Model
    '''
    a1,a2,a3,a4,a5,a6,a7,a8,a9 = theta
    model = a1*np.exp(-(x-a2)**2/(2*a3**2))+a4*np.exp(-(x-a5)**2/(2*a6**2))+a7*np.exp(-(x-a8)**2/(2*a9**2))
    return model
def model_2gauss(theta):
    '''
    Fit 2 Gaussian model
    Inputs:
    Parameters
    Outputs:
    2-gaussian Model
    '''
    a1,a2,a3,a4,a5,a6= theta
    model = a1*np.exp(-(x-a2)**2/(2*a3**2))+a4*np.exp(-(x-a5)**2/(2*a6**2))
    return model
def model_1gauss(theta):
    '''
    Fit 1 Gaussian model
    Inputs:
    Parameters
    Outputs:
    1-gaussian Model
    '''
    a1,a2,a3 = theta
    model = a1*np.exp(-(x-a2)**2/(2*a3**2))
    return model
def model_lorentzian(theta):
    '''
    Fit Lorentzian model
    Inputs:
    Parameters
    Outputs:
    Lorentzian Model
    '''
    a,b,c=theta
    model=(a/np.pi)*(c/((x-b)**2+c**2))
    return model
def three_gaussian_fit( params ):#functions for least squares minimization for initial guesses
    '''
    Least Squares 3-gauss minimization function
    Inputs:
    Fit Parameters
    Outputs:
    Fit-y
    '''
    fit = model_3gauss( params )
    return (fit - y)
def two_gaussian_fit( params ):
    '''
    Least Squares 2-gauss minimization function
    Inputs:
    Fit Parameters
    Outputs:
    Fit-y
    '''
    fit = model_2gauss(params )
    return (fit - y)
def one_gaussian_fit( params ):
    '''
    Least Squares 1-gauss minimization function
    Inputs:
    Fit Parameters
    Outputs:
    Fit-y
    '''
    fit= model_1gauss(params )
    return (fit - y)
def four_gaussian_fit( params ):
    '''
    Least Squares 4-gauss minimization function
    Inputs:
    Fit Parameters
    Outputs:
    Fit-y
    '''
    fit = model_4gauss(params )
    return (fit - y)
def lorentzian_fit(params):
    '''
    Least Squares lorentzian minimization function
    Inputs:
    Fit Parameters
    Outputs:
    Fit-y
    '''
    fit = model_lorentzian(params)
    return (fit - y)
def get_specdata(filename,redshift):
    '''
    Read in file depending on what format it is in and output wavelength/flux/error
    Inputs:
    File in format ascii,csv,fits file,redshift
    Output:
    wavelength,flux,error arrays-error assumed to be 0.1*flux if not supplied
    '''
    if filename[-3:]=='rtf' or filename[-5:]=='ascii' or filename[-3:]=='txt' or filename[-3:]=='dat':#check txt/ascii files first
        data=np.loadtxt(filename)
        wavelength=data[:,0]/(1+redshift)
        flux=data[:,1]
        dim=np.shape(data)[1]
        if dim>2:
            error=data[:,2]
        else:#if no error in data
            error=0.1*flux
    elif filename[-4:]=='fits':# check fits files
        f = fits.open(filename)  
        if len(f[0].data)==3 or len(f[0].data)==2:#checking for the normal types of fits files
            specdata = f[0].data 
            wavelength=specdata[0]/(1+redshift)
            flux=specdata[1]
            if len(specdata)==3:
                error=specdata[2]
            else:#if no error in data
                error=flux*0.1
        else:#more abnormal fits files
            data = f[0].data
            header = f[0].header
            flux=data[0]
            wavelength = np.arange(int(header['NAXIS1']))*header['CD1_1'] + header['CRVAL1']#use wavelength solution
            wavelength/=(1+redshift)
            indices=np.where(np.isnan(flux)==False)#check for nans-some fits files with certain spectrographs are nasty!
            flux=flux[indices]
            wavelength=wavelength[indices]
            if len(data)>5:
                error=data[1]
            else:
                error=0.05*flux #if no error in data
    elif filename[-3:]=='csv':#check csv files
        data=pd.read_csv(filename)
        wavelength=np.empty(0)
        flux=np.empty(0)
        error=np.empty(0)
        for i in range(len(data)):
            wavelength=np.append(wavelength,data.iloc[:,0][i]/(1+redshift))
            flux=np.append(flux,data.iloc[:,1][i])
            error=np.append(error,data.iloc[:,2][i])
    return wavelength,flux,error
def get_wavelength_vals(filename,lambda_pm):
    '''
    Read out data and convert to the desired units
    Takes in a file and region over which to fit and gives x in velocity, y in flux(in 1e-15 ergs/s/cm2/A) and y errors in the same unit
    Inputs:
    Spectral File
    Outputs:
    Velocity,continuum-subtracted flux,flux error
    '''
    l,f,e=get_specdata(args.filename,args.redshift)
    #get lambda, flux,error from data file
    if args.continuum_sub==True:#check that continuum subtraction is desired
        Rv=3.1
        Ebv=args.correction
        ext = F99(Rv=Rv)#set extinction with model from Fitzpatrick 1999
        f/=ext.extinguish(l/1e4,Ebv=Ebv)# do dust extinction
        line_value=args.wavelength#get relevant wavelength
        xfit=l[np.where((l>(line_value-lambda_pm))&(l<(line_value+lambda_pm)))]#get a subset of the wavelength around the line-don't fit the whole spectrum
        yfit=f[np.where((l>(line_value-lambda_pm))&(l<(line_value+lambda_pm)))]#get a subset of the flux around the line
        region=SpectralRegion((line_value-(lambda_pm/5))*u.nm, (line_value+(lambda_pm/5))*u.nm)# start fitting continuum with astropy
        spectrum = Spectrum1D(flux=f*u.Jy, spectral_axis=l*u.nm)
        g1_fit = fit_generic_continuum(spectrum,exclude_regions=region)#fit the whole spectrum-but exclude the relevant emission line
        y_continuum_fitted = g1_fit(l*u.nm)
        ynew=np.array(y_continuum_fitted)[np.where((l>line_value-lambda_pm)&(l<line_value+lambda_pm))]# now filter the continuum solution to relevant values
        x=3e5*(xfit-line_value)/(line_value)
        y=yfit-ynew
        yerr=e[np.where((l>line_value-lambda_pm)&(l<line_value+lambda_pm))]
        if abs(np.median(y))>0.01:
            y-=np.median(y)
        if abs(np.mean(y))<1e-10:
            y*=1e15#normalize to classic spectroscopy units to make fitting easier
            yerr*=1e15
    else:
        x=l
        y=f
        yerr=e
    return x,y,yerr    
x,y,e=get_wavelength_vals(args.filename,args.pm)
if args.guess is None:# if theres no guess lets try everything
    fit1 = leastsq( one_gaussian_fit, [1,0,100] )
    fit2 = leastsq( two_gaussian_fit, [0.1,10,100,0.1,10,100] )
    fit3 = leastsq( three_gaussian_fit, [0.1,10,100,0.1,10,100,0.1,10,100] )
    fit4 = leastsq( four_gaussian_fit, [0.1,10,1000,0.1,10,500,0.1,10,100,0.1,1000,100] )
    fitl = leastsq( lorentzian_fit, [0,10,100] )
    chi1=1/(len(y)-len(fit1[0]))*sum((y-model_1gauss(fit1[0]))**2/e**2)
    chi2=1/(len(y)-len(fit2[0]))*sum((y-model_2gauss(fit2[0]))**2/e*2)
    chi3=1/(len(y)-len(fit3[0]))*sum((y-model_3gauss(fit3[0]))**2/e**2)
    chi4=1/(len(y)-len(fit4[0]))*sum((y-model_4gauss(fit4[0]))**2/e**2)
    chi5=1/(len(y)-len(fitl[0]))*sum((y-model_lorentzian(fitl[0]))**2/e**2)
    chi_array=np.array([chi1,chi2,chi3,chi4,chi5])
    min_chisq=min(chi_array)# check greater than 1 assuming you have good errors-which you should!
    if np.argmin(chi_array)==0:
        print("Our initial least squares guess is a 1 component gaussian with parameters:",fit1[0])
        plt.plot(x,model_1gauss(fit1[0]),color='r',label='fit')
    elif np.argmin(chi_array)==1:
        print("Our initial least squares guess is a 2-component gaussian with parameters :",fit2[0])
        plt.plot(x,model_2gauss(fit2[0]),color='blue',label='fit')
        for i in range(3,9,3):
            plt.plot(x,model_1gauss(fit2[0][i-3:i]),color='r')
    elif np.argmin(chi_array)==2:
        print("Our initial least squares guess is a 3 component gaussian with parameters:",fit3[0])
        plt.plot(x,model_3gauss(fit3[0]),color='blue',label='fit')
        for i in range(3,12,3):
            plt.plot(x,model_1gauss(fit3[0][i-3:i]),color='r')
    elif np.argmin(chi_array)==3:
        print("Our initial least squares guess is a 4 component gaussian with parameters:",fit4[0])
        plt.plot(x,model_4gauss(fit4[0]),color='blue',label='fit')
        for i in range(3,15,3):
            plt.plot(x,model_1gauss(fit4[0][i-3:i]),color='r')
    elif np.argmin(chi_array)==4:
        print('Our initial least squares guess is a lorentzian with parameters:', fitl[0])
        plt.plot(x,model_lorentzian(fitl[0]),color='blue',label='fit')
    plt.plot(x,y,alpha=0.6,color='black',label='data')
    plt.legend()
    plt.xlabel('Velocity(km/s)')
    plt.ylabel('Flux')
    plt.show()
elif args.guess is not None:#If we have a guess read it in and do some fitting to maybe get a slightly better guess
    guess=np.loadtxt(args.guess)
    if len(guess)==6:
        fit = leastsq( two_gaussian_fit, guess)
        print("Final leastsq values=",fit[0])
        plt.plot(x,model_2gauss(fit[0]),color='blue',label='fit')
        for i in range(3,9,3):
            plt.plot(x,model_1gauss(fit[0][i-3:i]),color='r')
    elif len(guess)==9:
        fit = leastsq( three_gaussian_fit, guess)
        print("Final leastsq values=",fit[0])
        plt.plot(x,model_3gauss(fit[0]),color='blue',label='fit')
        for i in range(3,12,3):
            plt.plot(x,model_1gauss(fit[0][i-3:i]),color='r')
    elif len(guess)==12:
        fit = leastsq( four_gaussian_fit, guess)
        print("Final leastsq values=",fit[0])
        plt.plot(x,model_4gauss(fit[0]),color='blue',label='fit')
        for i in range(3,15,3):
            plt.plot(x,model_1gauss(fit[0][i-3:i]),color='r')
    elif len(guess)==3 and args.lorentzian==False:
        fit = leastsq( one_gaussian_fit, guess)
        print("Final leastsq values=",fit[0])
        plt.plot(x,model_1gauss(fit[0]),color='blue',label='fit')
    else:
        fit = leastsq(lorentzian_fit, guess)
        print("Final leastsq values=",fit[0])
        plt.plot(x,model_lorentzian(fit[0]),color='blue',label='fit')
    plt.plot(x,y,alpha=0.6,color='black',label='data')
    plt.legend()
    plt.xlim(-args.plot_limits,args.plot_limits)
    plt.xlabel('Velocity(km/s)')
    plt.ylabel('Flux')
    plt.show()

