import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Reading_in as r
import emcee
import corner
import warnings
warnings.filterwarnings("ignore")
filename=sys.argv[1]
x,y,yerr=r.get_wavelength_vals(filename)
def model_4gauss(theta):
    a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12 = theta
    model = a1*np.exp(-(x-a2)**2/(2*a3**2))+a4*np.exp(-(x-a5)**2/(2*a6**2))+a7*np.exp(-(x-a8)**2/(2*a9**2))+a10*np.exp(-(x-a11)**2/(2*a12**2))
    return model
def model_3gauss(theta):
    a1,a2,a3,a4,a5,a6,a7,a8,a9 = theta
    model = a1*np.exp(-(x-a2)**2/(2*a3**2))+a4*np.exp(-(x-a5)**2/(2*a6**2))+a7*np.exp(-(x-a8)**2/(2*a9**2))
    return model
def model_2gauss(theta):
    a1,a2,a3,a4,a5,a6= theta
    model = a1*np.exp(-(x-a2)**2/(2*a3**2))+a4*np.exp(-(x-a5)**2/(2*a6**2))
    return model
def model_1gauss(theta):
    a1,a2,a3 = theta
    model = a1*np.exp(-(x-a2)**2/(2*a3**2))
    return model
def model_lorentzian(theta):
    a,b,c=theta
    model=(a/np.pi)*(c/((x-b)**2+c**2))
    return model
def three_gaussian_fit( params ):
    fit = model_3gauss( x, params )
    return (fit - y)
def two_gaussian_fit( params ):
    fit = model_2gauss( x, params )
    return (fit - y)
def one_gaussian_fit( params ):
    fit =model_1gauss( x, params )
    return (fit - y)
def four_gaussian_fit( params ):
    fit = model_4gauss( x, params )
    return (fit - y)
def lorentzian_fit(params):
    fit = model_lorentzian(params)
    return (fit - y)
def lnlike(theta, x, y, yerr):
        if model=='1g':
            lnl=-np.sum((y-model_1gauss(theta))**2/yerr**2)/2
        elif model=='2g':
            lnl=-np.sum((y-model_2gauss(theta))**2/yerr**2)/2
        elif model=='3g':
            lnl=-np.sum((y-model_3gauss(theta))**2/yerr**2)/2
        elif model=='4g':
            lnl=-np.sum((y-model_4gauss(theta))**2/yerr**2)/2
        return lnl
def log_prior(theta):
        if model=='2g':
            a,b,c,d,e,f= theta
            if 0<a<5 and -500<b<500 and -1000<c<1000 and 0<d<5 and -1000<e<1000 and -1500<f<1500:
                return 0.0
        if model=='1g':
            a,b,c= theta
            if 0<a<5 and -500<b<500 and -1000<c<1000:
                return 0.0
        if model=='l':
            a,b,c= theta
            if 0<a<5 and -500<b<500 and -10000<c<10000:
                return 0.0
        if model=='3g':
            a,b,c,d,e,f,g,h,i= theta
            if 0<a<5 and -500<b<500 and -1000<c<1000 and 0<d<5 and -1000<e<1000 and -1500<f<1500 and 0<g<5 and -1500<h<1500 and -1e4<i<1e4:
                return 0.0
        if model=='4g':
            a,b,c,d,e,f,g,h,i,j,k,l= theta
            if 0<a<5 and -10000<b<10000 and -10000<c<10000 and 0<d<5 and -500<e<500 and -1500<f<1500 and 0<g<5 and -1000<h<1000 and -5e2<i<5e2 and 0<j<5 and -2e3<k<2e3 and -5e2<l<5e2:
                return 0.0
        return -np.inf
def lnprob(theta, x,y,yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr) #recall if lp not -inf, its 0, so this just returns likelihood
def main(p0,nwalkers,niter,ndim,lnprob,data):
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

        print("Running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0,4000)
        sampler.reset()
    
        print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter)

        return sampler, pos, prob, state
model=sys.argv[6]
def run_chains(model):
    data = (x,y,yerr)
    nwalkers=200
    niter=10000
    if model=='1g':
        initial = np.array([0.1,0,100] )
    elif model=='l':
        initial = np.array([0.3,0,1000] )
    elif model=='2g':
        initial = np.array([0.1,0,100,0.3,0,1000] )
    elif model=='3g':
        initial = np.array([1.5,-50,2000,0.2,-500,1000,0.8,0,50] )
    elif model=='4g':
        initial=[0.3,-5000,5000,0.8,100,950,0.3,50,100,0.2,1000,240]
    ndim=len(initial)
    p0 = [np.array(initial) + 1e-7 * np.random.randn(ndim) for i in range(nwalkers)]
    sampler, pos, prob, state = main(p0,nwalkers,niter,ndim,lnprob,data)
    samples=sampler.flatchain
    theta=samples[np.argmax(sampler.flatlnprobability)]
    labels=['a','b','c','d','e','f','g','h','i','j','k','l']
    corner.corner(samples,show_titles=True,labels=labels,plot_datapoints=True,quantiles=[0.16, 0.5, 0.84],title_fmt='.3f')
    plt.savefig('MCMC.png',format='png')
    plt.show()
    return theta
def plotting(model):
    theta=run_chains(model)
    if model=='2g':
        plt.plot(x,model_2gauss(theta),color='blue')
        print('MCMC reduced chi squared=',1/(len(x)-len(theta))*sum((y-model_2gauss(theta))**2/(yerr**2)))
    elif model=='3g':
        plt.plot(x,model_3gauss(theta),color='blue')
        print('MCMC reduced chi squared=',1/(len(x)-len(theta))*sum((y-model_3gauss(theta))**2/(yerr**2)))
    elif model=='1g':
        plt.plot(x,model_1gauss(theta),color='blue')
        print('MCMC reduced chi squared=',1/(len(x)-len(theta))*sum((y-model_1gauss(theta))**2/(yerr**2)))
    elif model=='4g':
        plt.plot(x,model_4gauss(theta),color='blue')
        print('MCMC reduced chi squared=',1/(len(x)-len(theta))*sum((y-model_4gauss(theta))**2/(yerr**2)))
    elif model=='l':
        plt.plot(x,model_lorentizan(theta),color='blue')
        print('MCMC reduced chi squared=',1/(len(x)-len(theta))*sum((y-model_lorentzian(theta))**2/(yerr**2)))
    plt.plot(x,y,color='black')
    plt.xlim(-10000,10000)
    plt.savefig('Fitted.png')
    plt.show()
plotting(model)    