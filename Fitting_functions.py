import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Reading_in as r
import emcee
import corner
import warnings
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
args = parser.parse_args()
filename=args.filename
redshift=args.redshift
lambda_pm=args.pm
guess=np.loadtxt(args.guess)
x,y,yerr=r.get_wavelength_vals(filename)
if len(guess)==6:
    model='2g'
elif len(guess)==9:
    model='3g'
elif len(guess)==12:
    model='4g'
elif len(guess)==3 and sys.argv[7]!='L':
    model='1g'
else:
    model='l'
def lnlike(theta, x, y, yerr):
        if model=='1g':
            lnl=-np.sum((y-r.model_1gauss(theta))**2/yerr**2)/2
        elif model=='2g':
            lnl=-np.sum((y-r.model_2gauss(theta))**2/yerr**2)/2
        elif model=='3g':
            lnl=-np.sum((y-r.model_3gauss(theta))**2/yerr**2)/2
        elif model=='4g':
            lnl=-np.sum((y-r.model_4gauss(theta))**2/yerr**2)/2
        return lnl
def log_prior(theta):
        if model=='2g':
            a,b,c,d,e,f= theta
            a0,b0,c0,d0,e0,f0=abs(guess)
            if -4*a0<a<4*a0 and -4*b0<b<4*b0 and -4*c0<c<4*c0 and -4*d0<d<4*d0 and -4*e0<e<4*e0 and -4*f0<f<4*f0:
                return 0.0
        elif model=='1g':
            a,b,c= theta
            a0,b0,c0=guess
            if -4*abs(a0)<a<4*abs(a0) and -4*abs(b0)<b<4*abs(b0) and -4*abs(c0)<c<4*abs(c0):
                return 0.0
        elif model=='l':
            a,b,c= theta
            a0,b0,c0=guess
            if 0<a<5 and -500<b<500 and -10000<c<10000:
                return 0.0
        elif model=='3g':
            a,b,c,d,e,f,g,h,i= theta
            a0,b0,c0,d0,e0,f0,g0,h0,i0= guess
            if--4*a0<a<4*a0 and -4*b0<b<4*b0 and -4*c0<c<4*c0 and -4*d0<d<4*d0 and -4*e0<e<4*e0 and -4*f0<f<4*f0 and -4*g0<g<4*g0 and -4*h0<h<4*h0 and -4*i0<i<4*i0:
                return 0.0
        elif model=='4g':
            a,b,c,d,e,f,g,h,i,j,k,l= theta
            a0,b0,c0,d0,e0,f0,g0,h0,i0,j0,k0,l0=abs(guess)
            if -4*a0<a<4*a0 and -4*b0<b<4*b0 and -4*c0<c<4*c0 and -4*d0<d<4*d0 and -4*e0<e<4*e0 and -4*f0<f<4*f0 and -4*g0<g<4*g0 and -4*h0<h<4*h0 and -4*i0<i<4*i0 and -4*j0<j<4*j0 and -4*k0<k<4*k0 and -4*l0<l<4*l0:  
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
data = (x,y,yerr)
nwalkers=200
niter=10000
ndim=len(guess)
p0 = [np.array(guess) + 1e-7 * np.random.randn(ndim) for i in range(nwalkers)]
sampler, pos, prob, state = main(p0,nwalkers,niter,ndim,lnprob,data)
samples=sampler.flatchain
tau = sampler.get_autocorr_time()
for i in range(len(tau)):
    if 40*tau[i]<niter:
        print('The chains converged for parameter {}'.format{i})
    else:
        print('Convergence failed')
theta=samples[np.argmax(sampler.flatlnprobability)]
labels=['a','b','c','d','e','f','g','h','i','j','k','l']
corner.corner(samples,show_titles=True,labels=labels,plot_datapoints=True,quantiles=[0.16, 0.5, 0.84],title_fmt='.3f')
plt.savefig('MCMC.png',format='png')
plt.show()
def plotting(model):
    if model=='2g':
        plt.plot(x,r.model_2gauss(theta),color='blue')
        #print('MCMC reduced chi squared=',1/(len(x)-len(theta))*sum((y-model_2gauss(theta))**2/(yerr**2)))
    elif model=='3g':
        plt.plot(x,r.model_3gauss(theta),color='blue')
        #print('MCMC reduced chi squared=',1/(len(x)-len(theta))*sum((y-model_3gauss(theta))**2/(yerr**2)))
    elif model=='1g':
        plt.plot(x,r.model_1gauss(theta),color='blue')
        #print('MCMC reduced chi squared=',1/(len(x)-len(theta))*sum((y-model_1gauss(theta))**2/(yerr**2)))
    elif model=='4g':
        plt.plot(x,r.model_4gauss(theta),color='blue')
        #print('MCMC reduced chi squared=',1/(len(x)-len(theta))*sum((y-model_4gauss(theta))**2/(yerr**2)))
    elif model=='l':
        plt.plot(x,r.model_lorentizan(theta),color='blue')
        #print('MCMC reduced chi squared=',1/(len(x)-len(theta))*sum((y-model_lorentzian(theta))**2/(yerr**2)))
    plt.plot(x,y,color='black',alpha=0.5)
    #plt.xlim(-10000,10000)
    plt.savefig('Fitted.png')
    plt.show()
if model=='4g':
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    df=pd.DataFrame(columns=['Comp1_Amp','Comp1_Center','Comp1_SD','Comp2_Amp','Comp2_Center','Comp2_SD','Comp3_Amp','Comp3_Center','Comp3_SD','Comp4_Amp','Comp4_Center','Comp4_SD'])
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "{0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        if i==0:
            df['Comp1_Amp']=[txt]
        if i==1:
            df['Comp1_Center']=[txt]
        if i==2:
            df['Comp1_SD']=[txt]
        if i==3:
            df['Comp2_Amp']=[txt]
        if i==4:
            df['Comp2_Center']=[txt]
        if i==5:
            df['Comp2_SD']=[txt]
        if i==6:
            df['Comp3_Amp']=[txt]
        if i==7:
            df['Comp3_Center']=[txt]
        if i==8:
            df['Comp3_SD']=[txt]
        if i==9:
            df['Comp4_Amp']=[txt]
        if i==10:
            df['Comp4_Center']=[txt]
        if i==11:
            df['Comp4_SD']=[txt]
    df.to_csv('Final Results.csv')
if model=='3g':
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    df=pd.DataFrame(columns=['Comp1_Amp','Comp1_Center','Comp1_SD','Comp2_Amp','Comp2_Center','Comp2_SD','Comp3_Amp','Comp3_Center','Comp3_SD','Comp4_Amp','Comp4_Center','Comp4_SD'])
    ndim=len(guess)
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "{0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        if i==0:
            df['Comp1_Amp']=[txt]
        if i==1:
            df['Comp1_Center']=[txt]
        if i==2:
            df['Comp1_SD']=[txt]
        if i==3:
            df['Comp2_Amp']=[txt]
        if i==4:
            df['Comp2_Center']=[txt]
        if i==5:
            df['Comp2_SD']=[txt]
        if i==6:
            df['Comp3_Amp']=[txt]
        if i==7:
            df['Comp3_Center']=[txt]
        if i==8:
            df['Comp3_SD']=[txt]
    df.to_csv('Final Results.csv')
if model=='1g':
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    df=pd.DataFrame(columns=['Comp1_Amp','Comp1_Center','Comp1_SD'])
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "{0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        if i==0:
            df['Comp1_Amp']=[txt]
        if i==1:
            df['Comp1_Center']=[txt]
        if i==2:
            df['Comp1_SD']=[txt]
    df.to_csv('Final Results.csv')
if model=='2g':
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    df=pd.DataFrame(columns=['Comp1_Amp','Comp1_Center','Comp1_SD','Comp2_Amp','Comp2_Center','Comp2_SD'])
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "{0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        if i==0:
            df['Comp1_Amp']=[txt]
        if i==1:
            df['Comp1_Center']=[txt]
        if i==2:
            df['Comp1_SD']=[txt]
        if i==3:
            df['Comp2_Amp']=[txt]
        if i==4:
            df['Comp2_Center']=[txt]
        if i==5:
            df['Comp2_SD']=[txt]
    df.to_csv('Final Results.csv')
if model=='l':
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    df=pd.DataFrame(columns=['Amp','Center','Width'])
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "{0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        if i==0:
            df['Amp']=[txt]
        if i==1:
            df['Center']=[txt]
        if i==2:
            df['Width']=[txt]
    df.to_csv('Final Results.csv')
plotting(model)    