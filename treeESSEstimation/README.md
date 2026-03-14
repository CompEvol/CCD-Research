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
* TBA: scripts for MDS and trace plots (with modality information)

If you (re)produce multiple such plots, we recommend using a file containing mappings
between each estimator method (as you named in your experiment code) and a display name and colour.

## Examples

We provide example data of autocorrelation traces, estimates for the accuracy experiment for RNNI and ACT 5, and of the stability and robustness experiments.

## Data

TBA links to MCMC chains


