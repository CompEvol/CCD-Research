# Estimating the ESS of phylogenetic tree chains
This repository contains extra information, scripts and data for the research paper:

**On estimating the effective sample size of phylogenetic trees in an autocorrelated chain**<br>
*Jonathan Klawitter*, *Lars Berling*, *Jordan Douglas*, *Dong Xie*, and *Alexei J. Drummond*<br>
[arXiv: doi.org/10.48550/arXiv.2603.03521v1](https://doi.org/10.48550/arXiv.2603.03521)

## Scripts

We provide scripts to generate the different types of plots used in the paper.
Explanations on how to use each one are documented at the top of each file.
You can test them on the provided example data.

* `autocorrelation_plots` - Python script to plot autocorrelation signature plots 
* `accuracy_plots` - Python script to plot accuracy of estimators, absolute and relative and summary plots
* `stability_and_robustness_plots` - Python script to plot the stability of estimators over thinning intervals (as in the stability experiment) 
or their robustness under chain fragmentation (as in the robustness experiment)
as well as to generate the summary tables
* MDS and trace plots (with modality information) were generated with a mix of tools; see for example [this paper](https://doi.org/10.1093/gbe/evw171) for information on trace plots.

If you (re)produce multiple such plots, we recommend using a file containing mappings
between each estimator method (as you named in your experiment code) and a display name and colour.

## Examples

We provide example data of autocorrelation traces, estimates for the accuracy experiment for RNNI and ACT 5, and of the stability and robustness experiments.

## Data

- The full MCMC runs and XML files for DS1-11 are available [here](https://doi.org/10.0.68.200/k6.auckland.c.8351665).
- The full MCMC runs and XML files for Yule50 are available [here](https://doi.org/10.17608/k6.auckland.c.7102354).
- Oversampled: The oversampled MCMC runs for DS1-11 are available here under `data/dsOversampled/`
- Simulated chains: The simulated simple and noisy chains are not provided as they simply repeat sampled trees; the RNNI chains are available with DS1-11 [here](https://doi.org/10.17608/k6.auckland.c.7102354).



