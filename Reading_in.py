import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dust_extinction.parameter_averages import CCM89, F99
from specutils.spectra import Spectrum1D, SpectralRegion
from specutils.fitting import fit_generic_continuum
from astropy import units as u
import warnings
from scipy.optimize import leastsq
import argparse
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='Fit spectral lines')
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
args = parser.parse_args()
def model_4gauss(theta):
    '''
    Fit 4 Gaussian model
    '''
    a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12 = theta
    model = a1*np.exp(-(x-a2)**2/(2*a3**2))+a4*np.exp(-(x-a5)**2/(2*a6**2))+a7*np.exp(-(x-a8)**2/(2*a9**2))+a10*np.exp(-(x-a11)**2/(2*a12**2))
    return model
def model_3gauss(theta):
    '''
    Fit 3 Gaussian model
    '''
    a1,a2,a3,a4,a5,a6,a7,a8,a9 = theta
    model = a1*np.exp(-(x-a2)**2/(2*a3**2))+a4*np.exp(-(x-a5)**2/(2*a6**2))+a7*np.exp(-(x-a8)**2/(2*a9**2))
    return model
def model_2gauss(theta):
    '''
    Fit 2 Gaussian model
    '''
    a1,a2,a3,a4,a5,a6= theta
    model = a1*np.exp(-(x-a2)**2/(2*a3**2))+a4*np.exp(-(x-a5)**2/(2*a6**2))
    return model
def model_1gauss(theta):
    '''
    Fit 1 Gaussian model
    '''
    a1,a2,a3 = theta
    model = a1*np.exp(-(x-a2)**2/(2*a3**2))
    return model
def model_lorentzian(theta):
    '''
    Fit Lorentzian model
    '''
    a,b,c=theta
    model=(a/np.pi)*(c/((x-b)**2+c**2))
    return model
def three_gaussian_fit( params ):#functions for least squares minimization
    fit = model_3gauss( params )
    return (fit - y)
def two_gaussian_fit( params ):
    fit = model_2gauss(params )
    return (fit - y)
def one_gaussian_fit( params ):
    fit= model_1gauss(params )
    return (fit - y)
def four_gaussian_fit( params ):
    fit = model_4gauss(params )
    return (fit - y)
def lorentzian_fit(params):
    fit = model_lorentzian(params)
    return (fit - y)
filename=args.filename
redshift=args.redshift
lambda_pm=args.pm
Rv=3.1
Ebv=args.correction
ext = F99(Rv=Rv)
def get_data(filename):
    '''
    Read in file depending on what format it is in
    '''
    if filename[-3:]=='rtf' or filename[-5:]=='ascii' or filename[-3:]=='txt' or filename[-3:]=='dat':#check txt/ascii files first
        data=np.loadtxt(filename)
        wavelength=data[:,0]/(1+redshift)
        flux=data[:,1]
        dim=np.shape(data)[1]
        if dim>2:
            error=data[:,2]
        else:
            error=0.2*flux
    elif filename[-4:]=='fits':# check fits files
        f = fits.open(filename)  
        specdata = f[0].data #assume file is in 2nd header-could change this potentially
        wavelength=specdata[0]/(1+redshift)
        flux=specdata[1]
        error=specdata[2]
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
def get_wavelength_vals(filename):
    '''
    Read out data and convert to the desired units
    Gives x in velocity, y in flux(in 1e-15 ergs/s/cm2/A) and yerrors in the same unit
    '''
    l,f,e=get_data(filename)#get lambda, flux,error from data file
    l_ext=f/ext.extinguish(l/1e4,Ebv=Ebv)# do dust extinction
    line_value=args.wavelength#get relevant wavelength
    xfit=l[np.where((l>(line_value-lambda_pm))&(l<(line_value+lambda_pm)))]#get a subset of the wavelength around the line-don't fit the whole spectrum
    yfit=f[np.where((l>(line_value-lambda_pm))&(l<(line_value+lambda_pm)))]#get a subset of the flux around the line
    region=SpectralRegion((line_value-(lambda_pm/5))*u.nm, (line_value+(lambda_pm/5))*u.nm)# start fitting continuum with astropy
    spectrum = Spectrum1D(flux=f*u.Jy, spectral_axis=l*u.nm)
    g1_fit = fit_generic_continuum(spectrum,exclude_regions=region)#fit the whole spectrum-but exclude the relevant emission line
    y_continuum_fitted = g1_fit(l*u.nm)
    ynew=np.array(y_continuum_fitted)[np.where((l>line_value-lambda_pm)&(l<line_value+lambda_pm))]# now filter the continuum solution to relevant values
    y=yfit-ynew
    yerr=e[np.where((l>line_value-lambda_pm)&(l<line_value+lambda_pm))]   
    if abs(np.mean(y))<1e-10:
        y*=1e15
        yerr*=1e15
    if abs(np.median(y))>0.01:
        y-=np.median(y)
    x=3e5*(xfit-line_value)/(line_value)
    return x,y,yerr    
x,y,e=get_wavelength_vals(filename)
if args.guess is None:
    fit1 = leastsq( one_gaussian_fit, [0.1,10,100] )
    fit2 = leastsq( two_gaussian_fit, [0.1,10,100,0.1,10,100] )
    fit3 = leastsq( three_gaussian_fit, [0.1,10,100,0.1,10,100,0.1,10,100] )
    fit4 = leastsq( four_gaussian_fit, [0.1,10,1000,0.1,10,500,0.1,10,100,0.1,1000,100] )
    fit5 = leastsq( lorentzian_fit, [0.1,10,1000] )
    chi1=1/(len(y)-len(fit1[0]))*sum((y-model_1gauss(fit1[0]))**2/e**2)
    chi2=1/(len(y)-len(fit2[0]))*sum((y-model_2gauss(fit2[0]))**2/e*2)
    chi3=1/(len(y)-len(fit3[0]))*sum((y-model_3gauss(fit3[0]))**2/e**2)
    chi4=1/(len(y)-len(fit4[0]))*sum((y-model_4gauss(fit4[0]))**2/e**2)
    chi5=1/(len(y)-len(fit5[0]))*sum((y-model_lorentzian(fit5[0]))**2/e**2)
    chi_array=np.array([chi1,chi2,chi3,chi4,chi5])
    min_chisq=min(chi_array)
    print('Minimum chi^2 with rough guess', min_chisq)
    if np.argmin(chi_array)==0:
        print("Our initial least squares guess is a 1 component gaussian with parameters:",fit1[0])
        plt.plot(x,model_1gauss(fit1[0]),color='r')
    elif np.argmin(chi_array)==1:
        print("Our initial least squares guess is a 2-component gaussian with parameters :",fit2[0])
        plt.plot(x,model_2gauss(fit2[0]),color='r')
    elif np.argmin(chi_array)==2:
        print("Our initial least squares guess is a 3 component gaussian with parameters:",fit3[0])
        plt.plot(x,model_3gauss(fit3[0]),color='r')
    elif np.argmin(chi_array)==3:
        print("Our initial least squares guess is a 4 component gaussian with parameters:",fit4[0])
        plt.plot(x,model_4gauss(fit4[0]),color='r')
    elif np.argmin(chi_array)==4:
        print(fit5[0])
        plt.plot(x,model_lorentizan(fit5[0]),color='r')
    plt.plot(x,y,alpha=0.6)
    plt.show()
elif args.guess is not None:
    guess=np.loadtxt(args.guess)
    if len(guess)==6:
        fit = leastsq( two_gaussian_fit, guess)
        print("Final leastsq values=",fit[0])
        plt.plot(x,model_2gauss(fit[0]),color='blue')
        plt.plot(x,model_1gauss(fit[0][0:3]),color='r')
        plt.plot(x,model_1gauss(fit[0][3:6]),color='r')
    elif len(guess)==9:
        fit = leastsq( three_gaussian_fit, guess)
        print("Final leastsq values=",fit[0])
        plt.plot(x,model_3gauss(fit[0]),color='blue')
        plt.plot(x,model_1gauss(fit[0][0:3]),color='r')
        plt.plot(x,model_1gauss(fit[0][3:6]),color='r')
        plt.plot(x,model_1gauss(fit[0][6:9]),color='r')
    elif len(guess)==12:
        fit = leastsq( four_gaussian_fit, guess)
        print("Final leastsq values=",fit[0])
        plt.plot(x,model_4gauss(fit[0]),color='blue')
        plt.plot(x,model_1gauss(fit[0][0:3]),color='r')
        plt.plot(x,model_1gauss(fit[0][3:6]),color='r')
        plt.plot(x,model_1gauss(fit[0][6:9]),color='r')
        plt.plot(x,model_1gauss(fit[0][9:12]),color='r')
    elif len(guess)==3 and sys.argv[7]!='L':
        fit = leastsq( one_gaussian_fit, guess)
        print("Final leastsq values=",fit[0])
        plt.plot(x,model_1gauss(fit[0]),color='r')
    elif len(guess)==3 and sys.argv[7]=='L':
        fit = leastsq(lorentzian_fit, guess)
        print("Final leastsq values=",fit[0])
        plt.plot(x,model_lorentzian(fit[0]),color='r')
    plt.plot(x,y,alpha=0.6,color='black')
    plt.xlim(-args.plot_limits,args.plot_limits)
    plt.show()

