# MCMC-spectra
This repo contains code that will does MCMC-based fitting to emission line profiles in astronomical spectra, largely aimed at supernova spectra but certainly with utility for other kinds of objects.

In general, the code will run in two steps. First, the code will read in the file and convert the wavelength to velocity space, then make a least-squares based guess as to the best-fitting model to the relevant emission line, whether that be a multi-component gaussian or single loretnzian. The code will output the best-fit values, which the user can then use as a "guess" for the next step, which is MCMC fitting to the emission line to determine parameters with full posterior distributions and thus associated errors. The code will also output the plot of the emission line so that the user can determine their own guess. In the MCMC steps, the priors will be set by the guess, and the MCMC chains will be run with 200 parallelized walkers and 10000 iterations, with a 1000 step burn-in. The final results will be plotted in a corner plot with 16th and 84th-percentile errors, and the full values are output to a csv .
The code largely runs through two separate scripts which perform the two relevant tasks with associated command-line arguments.
In the first step, after cloning this repo one can run something like this to get a feel for the data/potential best-fitting model

```
python3 Reading_in.py 20220311.rtf -z 0.01 --correction 0.023 --pm 500 --wavelength 6563
```
The file must be in one of a couple possble formats:
* ascii file with the first column wavelength(in /AA) and second column flux(and ideally third column error)
* similarly-formatted csv file
* fits file with the first extension containing a fits file with one data extensioon and one flux extension

This would read in the data and find the emission profile at 5000 \AA, plotting the continuum-subtracted resulting profile in velocity space given a redshift and extinction correction as well as a buffer of 500 \AA around the central wavelength
Adter running this, the code will output a potential guess. If that guess seems incorrect, you can re-run the Reading_in.py script with your own guess to check whether that gives a decent fit with least squares analysis. The code will output a $\chi^2$, assuming you give it error in the third column of your data. It otherwise assumes 10 % error on the flux.

The guess file should just be an ascii file with the relevant number of values(i.e. 6 for a 2-gaussian fit). In the next step, you run the following command to do the fitting with MCMC(with similar command-line arguments):
```
python3 Fitting_functions.py 20220311.rtf --guess guess.txt -z 0.0217 --correction 0.023 --pm 500 --wavelength 6563
```
This will output a corner plot that will save as a .png file as well as an output csv file with the final posterior distributions and errorbars($1 \sigma$).
