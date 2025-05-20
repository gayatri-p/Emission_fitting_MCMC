#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from specutils.spectra import Spectrum1D, SpectralRegion
from specutils.fitting import fit_generic_continuum
from astropy.io import fits
from astropy import units as u
import warnings
from scipy.optimize import leastsq
import argparse
import emcee
import corner

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
parser.add_argument('--output', default=0,type=str,
                    help='Details of the SN to be saved as the output file')
parser.add_argument('--guess',help='Best guess file')
parser.add_argument('--plot_limits', default=10000,type=int,
                    help='Limit of plots')
parser.add_argument('--continuum_sub', default=True,
                    help='Do continuum_sub')
parser.add_argument('--scaling', default=1,type=float,
                    help='Scaling factor for the flux')
parser.add_argument('--lorentzian', default=False,
                    help='Input True if Lorentzian guess')
args = parser.parse_args()

''' Relevant Models '''
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

def get_specdata(filename,redshift):
    '''
    Read in file depending on what format it is in and output wavelength/flux/error
    Inputs:
    File in format ascii,csv,fits file,redshift
    Output:
    wavelength,flux,error arrays-error assumed to be 0.1*flux if not supplied
    '''
    if filename[-3:]=='rtf' or filename[-5:]=='ascii' or filename[-3:]=='txt' or filename[-3:]=='dat' or filename[-3:]=='flm':#check txt/ascii files first
        data=np.loadtxt(filename)
        wavelength=data[:,0]/(1+redshift)
        flux=data[:,1]*args.scaling
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
    # print(x, y, yerr)
    return x,y,yerr    

def guess_fitting_params(x, y, e, guess_file):
    if guess_file is None:# if theres no guess lets try everything

        plt.figure(figsize=(5,5))
        fit1 = leastsq( one_gaussian_fit, [1,0,100] )
        chi1=1/(len(y)-len(fit1[0]))*sum((y-model_1gauss(fit1[0]))**2/e**2)
        print("Our initial least squares guess is a 1 component gaussian with parameters:",fit1[0])
        plt.plot(x,model_1gauss(fit1[0]),color='r',label='fit')
        plt.plot(x,y,alpha=0.6,color='black',label='data')
        plt.legend()
        # plt.xlim(-args.plot_limits,args.plot_limits)
        plt.xlabel('Velocity(km/s)')
        plt.ylabel('Flux')
        plt.gca().set_xlim(left=0)
        plt.savefig(f'outputs/{args.output}_fitted.png', bbox_inches = 'tight', format='png')
        plt.show()

    elif guess_file is not None: #If we have a guess read it in and do some fitting to maybe get a slightly better guess
        guess=np.loadtxt(guess_file)
        
        plt.figure(figsize=(5,5))
        fit = leastsq( one_gaussian_fit, guess)
        print("Final leastsq values=",fit[0])
        plt.plot(x,model_1gauss(fit[0]),color='blue',label='fit')

        plt.plot(x,y,alpha=0.6,color='black',label='data')
        plt.legend()
        # plt.xlim(-args.plot_limits,args.plot_limits)
        plt.xlabel('Velocity(km/s)')
        plt.ylabel('Flux')
        plt.show()
    
    return fit1[0]

x,y,e=get_wavelength_vals(args.filename,args.pm)
guessed_params = guess_fitting_params(x, y, e, args.guess)
continue_prompt = input('Continue? ')

''' Fitting to a Corner Plot'''

model='1g'

def lnlike(theta, x, y, yerr):
    '''
    log likelihood function-chi squared like-defining for different potential models'
    Inputs:
    parameters,x,y,yerr
    Outputs:
    Log Likelihood depending on preferred model
    '''
    lnl=-np.sum((y-model_1gauss(theta))**2/yerr**2)/2
    return lnl

def log_prior(theta):
        '''
        Find the log prior based on guess and model
        Inputs:
        Parameters
        Outputs:
        log prior-set the priors properly given +/- 5*guess buffers
        '''
        a,b,c= theta
        a0,b0,c0=guessed_params
        if -5*abs(a0)<a<5*abs(a0) and -5*abs(b0)<b<5*abs(b0) and -5*abs(c0)<c<5*abs(c0):
            return 0.0

        return -np.inf

def lnprob(theta, x,y,yerr):
    '''
    find maximimum likelihood-but only within prior bounds
    Inputs:
    Parameters,Data
    Outputs:
    Likelihood plus consideration of prior to not leave the relevant parameter space
    '''
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr) #recall if lp not -inf, its 0, so this just returns likelihood

def main(p0,nwalkers,niter,ndim,lnprob,data):
    '''
    running mcmc with set burn in, walkers and iterations
    Input:
    Priors,walkers,iterations,dimensions, log probability considering priors, data
    Output:
    Final samples and positions of posteriors to get final distributions out 
    '''
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)
    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0,1000)
    sampler.reset()

    print("Running production...")
    pos, prob, state = sampler.run_mcmc(p0, niter)
    return sampler, pos, prob, state

x,y,yerr=x,y,e
data = (x,y,yerr)
nwalkers=200
niter=20000#args.niter
ndim=len(guessed_params)
p0 = [np.array(guessed_params) + 1e-7 * np.abs(np.random.randn(ndim)) for i in range(nwalkers)]

sampler, pos, prob, state = main(p0,nwalkers,niter,ndim,lnprob,data)
samples=sampler.flatchain#get the final samples

try:
    tau = sampler.get_autocorr_time()# get autocorrelation times
    for i in range(len(tau)):
        if 40*tau[i]<niter:
            print('The chains converged for parameter',i+1)
        else:
            print('Convergence failed')
    theta=samples[np.argmax(sampler.flatlnprobability)]#get samples at min likelihood"
    
except Exception as e:
    print(e)

finally:

    def plotting(model):
        '''
        final plotting and read-out based on the preferred model set by the length of the guess
        Input:
        Preferred model:
        Output:
        Final plots with MCMC posteriors
        '''
        plt.figure(figsize=(5,5))
        chi_sq = 1/(len(x)-len(theta))*sum((y-model_1gauss(theta))**2/(yerr**2))
        plt.plot(x,y,color='black',alpha=0.5,label='Data')
        plt.plot(x, model_1gauss(theta),color='blue',label=f'Best-fit Model ($\chi^2$ = {round(chi_sq, 3)})')
        print('MCMC reduced chi squared=',chi_sq)
        plt.legend()
        plt.xlabel('Velocity (km/s)')
        plt.ylabel('Relative Flux')
        plt.gca().set_xlim(left=0)
        plt.savefig(f'outputs/{args.output}_fitted.png', bbox_inches = 'tight', format='png', dpi=300)
        plt.show()

    labels=['$A_{1}$','$\\mu_{1}$','$\\sigma_{1}$']
    corner.corner(samples,show_titles=True,labels=labels,plot_datapoints=True,quantiles=[0.16, 0.5, 0.84],title_fmt='.3f')
    plt.savefig(f'outputs/{args.output}_corner.png',format='png')
    plt.show()
    plt.cla()   # Clear axis
    plt.clf()   # Clear figure

    plotting(model) #call plotting function to output results after saving results above