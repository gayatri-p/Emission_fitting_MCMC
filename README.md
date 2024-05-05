# MCMC-spectra
This repo contains Python code that does MCMC-based fitting to emission line profiles in astronomical spectra, largely aimed at supernova spectra but certainly with utility for other kinds of objects.

In general, the code will run in two steps. First, the code will read in the file and convert the wavelength to velocity space, then make a least-squares based guess as to the best-fitting model to the relevant emission line, whether that be a multi-component gaussian or single lorentzian. The code will output the best-fit values, which the user can then use as a "guess" for the next step, which is MCMC fitting to the emission line to determine parameters with full posterior distributions and thus associated errors. The initial step will also output a full plot of the emission line in velocity space so that the user can determine their own guess. In the MCMC step, the priors will be set by the guess, and the MCMC chains will be run with 200 parallelized walkers and 10000 iterations, with a 1000 step burn-in. The final results will be plotted in a corner plot with 16th and 84th-percentile errors, and the full values are output to a csv .
The code largely runs through these two separate scripts which perform the two relevant tasks using associated command-line arguments.
In the first step, you can run something like this to get a feel for the data/potential best-fitting model

```
python3 Reading_in.py Demo/2020ywx_20220202.txt -z 0.0217 --correction 0.023 --pm 500 --wavelength 6563
```
This would read in the data and find the emission profile at $H \alpha$, plotting the continuum-subtracted resulting profile in velocity space given a redshift and extinction correction as well as a buffer of 500 Å around the central wavelength for the fit region. It also generates a guess for the best-fitting model using basic $\chi^2$ analysis.
The command line arguments available for modification in the first step are the following:

```filename```-The name of the file input. The file must be in one of a couple possble formats(ensure it is in an accessible path from wherever you are running the code):
* ascii file with the first column wavelength(in Å) and second column flux(and ideally third column error)
* similarly-formatted csv file
* fits file with the first extension containing a fits file with associated wavelength, flux and error array

 ```-z```-The redshift of the object.
 
 ```--correction```-The dust correction E(B-V) of the object(often inferred from the Sodium Doublet)
 
 ```--pm```- The plus or minus wavelength range over which to define the fitting region
 
 ```--wavelength``` The central wavelength of the line you want to analyze/fit profiles to
 
 ```--continuum_sub``` Whether you want the continuum subtracted-default True

  ```--plot_limits``` Allows you to change the plot limits for a nicer view as the plot outputs

If the generated guess seems incorrect, you can re-run the Reading_in.py script with your own guess(in the form of an ascii file) to check whether that gives a decent fit(by eye) with least squares analysis. In general, ensure you have some good estimate for the error in your spectra as this code does not involve generating errors on your spectra(it assumes 10 $\%$ error).

In the next step, you run the following command to do the fitting with MCMC with whatever best guess you can generate or the first step generates(with otherwise similar command-line arguments to the first step). The guess file should just be an ascii file with all the components of your guess. For the example given in the Demo directory, the guess we choose is the one output by the Reading in step. We then use this guess to run the following:
```
python3 Fitting_functions.py Demo/2020ywx_20220202.txt --guess Demo/guess_0202.txt -z 0.0217 --correction 0.023 --pm 500 --wavelength 6563
```
This script will run through the first script as well as it uses functions from this, so you can check that your guess makes sense and quit before the MCMC run if something looks off. This second script takes the same command line arguments with one addition: 

 ```--niter``` The number of MCMC iterations you want to run. Defaults to 10000. Could be modified given some strange behavior of the chains.

The code will check for autocorrelation by ensuring the number of iterations is 40x the autocorrelation time for each parameter. The code should take ~ <2 minutes to run. The MCMC corner plot will show up in the python window if you have the matplotlib widget, but it should be easier to look at this through the output .png file.
This will output a corner plot that will save as a .png file(MCMC.png) as well as an output csv file with the final posterior distributions and upper and lower errorbars($\pm 1 \sigma$) saved as Final_results.csv.  The reduced $\\chi^2$ will also be output, but this may be biased by poor error estimation.
