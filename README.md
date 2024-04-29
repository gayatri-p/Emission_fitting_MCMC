# MCMC-spectra
This repo contains code that will does MCMC-based fitting to emission line profiles in astronomical spectra, largely aimed at supernova spectra but certainly with utility for other kinds of objects.

In general, the code will run in two steps. First, the code will read in the file and convert the wavelength to velocity space, then make a least-squares based guess as to the best-fitting model to the relevant emission line, whether that be a multi-component gaussian or single loretnzian. The code will output the best-fit values, which the user can then use as a "guess" for the next step, which is MCMC fitting to the emission line to determine parameters with full posterior distributions and thus associated errors. The code will also output the plot of the emission line so that the user can determine their own guess. In the MCMC steps, the priors will be set by the guess, and the MCMC chains will be run with 200 parallelized walkers and 10000 iterations, with a 1000 step burn-in. The final results will be plotted in a corner plot with 16th and 84th-percentile errors, and the full values are output to a csv .
The code largely runs through two separate scripts which perform the two relevant tasks with associated command-line arguments.
In the first step, after cloning this repo one can run something like this to get a feel for the data/potential best-fitting model

```
python3 Reading_in.py SN2020ywx.txt -z 0.01 --correction 0.023 --pm 500 --wavelength 6563
```
The file must be in one of a couple possble formats:
* ascii file with the first column wavelenghth(in /AA) and second column flux(and ideally third column error)
* similarly-formatted csv file
* fits file with the first extension containing a fits file with one data extensioon and one flux extension
This would read in the data and find the emission profile at 5000 \AA, plotting the continuum-subtracted resulting profile in velocity space given a redshift and extinction correction as well as a buffer of 500 \AA around the central wavelength
