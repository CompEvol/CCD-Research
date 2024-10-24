# Skeletons in the Forest: Using Entropy-Based Rogue Detection on Bayesian Phylogenetic Tree Distributions
This repository contains extra information, scripts and data for the research paper:

**Skeletons in the Forest: Using Entropy-Based Rogue Detection on Bayesian Phylogenetic Tree Distributions**
*Jonathan Klawitter*, *Remco R. Bouckaert*, and *Alexei J. Drummond*
[bioRxiv: 10.1101/2024.09.25.615070v1](https://www.biorxiv.org/content/10.1101/2024.09.25.615070)

## Scripts

- Python script to compute WCSS coverage values and generate plots
- Python script to create rank plots of rogue scores for clades of different size
<!-- The trees are drawn with FigTree and the cloudograms are made with [DensiTree](https://www.cs.auckland.ac.nz/~remco/DensiTree/). -->

## Examples

There is a 5k long MCMC chain of the RSV2 dataset.
It can be used to test the RogueAnalysis and SkeletonAnalysis tools in the BEAST 2 [CCD package](https://github.com/CompEvol/CCD).
Also given are example outcomes of these analyses (as used in the paper):
- csv file of rogue rank data
- CCD MAP tree annotated with rogue scores
- Skeleton CCD MAP trees with placement of removed rogues

## Reproducing experiments

### Rogue, skeleton, and placement analyses
These analyses can be conducted with the tools mentioned above;
the RSV2 and DS7 trees used in the experiment, can be obtained [HERE](https://doi.org/10.17608/k6.auckland.27041803). 
Alternatively, you may use the smaller RSV2 exampled provided in this repository or running this [taming the beast (time-stamped data) tutorial](https://taming-the-beast.org/tutorials/MEP-tutorial/) to obtain a chain of desired length.

Th SCC trees can be obtained by rerunning the L86 [phylonco analysis](https://github.com/bioDS/beast-phylonco-paper)
or contacting the respective authors.

### Well-calibrated simulation study
 
The Yule100 trees are also available [HERE](https://doi.org/10.17608/k6.auckland.27041803).
Alternatively, the LPhy model is given in the data directory, so you can create and run your own xml using LPhyBeast and BEAST 2.
You can then apply the tools above to each tree sample and finally use the python coverage script above.

