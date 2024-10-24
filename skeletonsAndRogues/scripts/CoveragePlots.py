import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Bio import Phylo
from io import StringIO

## Script to compute coverage plots ##
# Works in 4 steps
# 1. You specify the path, filenames, and parameters you want
# 1.b Also, you need might need to adapt the paths and filenames in the extract_.. methods
# 2. Extracts true values from log and xml file
# - You might have to adapt the code and paths for your parameters
# 3. Compute HPD intervals
# 4. Construct and save plot

# 1. Base directory and parameters
basePath = "path/to/your/trees/folder/"
simulationName = "Yule100"
plotFileName = "Yule100.pdf"
plotRows = 3
plotColumns = 3
parametersPrint = ["tree height", "tree length", "kappa", "shape", "birth rate", "freq A", "freq C", "freq G", "freq T"]
parametersBEAST = ["phi.height", "phi.treeLength", "kappa", "shape", "birthRate", "frequencies.A", "frequencies.C", "frequencies.G", "frequencies.T"]

# since BEAST and LPhy use different column headers for frequencies, we need a mapping
frequenciesMapping = {"frequencies.A": "frequencies_0",
                      "frequencies.C": "frequencies_1", 
                      "frequencies.G": "frequencies_2", 
                      "frequencies.T": "frequencies_3"}


# 2. Extract true values
def extract_true_values():
    print("Extract true values")
    true_values = {param: [] for param in parametersBEAST}
    extract_true_values_from_xml(true_values)
    extract_true_values_from_log(true_values)
    return true_values    

def extract_true_values_from_xml(true_values):
    for i in range(1, 101):
        # open xml
        xmlFile = os.path.join(basePath, f"reps/rep{i}/yule-n100-{i}.xml")
        xmlTree = ET.parse(xmlFile)
        root = xmlTree.getroot()

        # extract and parse tree, then compute values you want
        newick_str = root.find(".//stateNode[@id='phi']").attrib['newick']
        tree = Phylo.read(StringIO(newick_str), "newick")
        true_values["phi.height"].append(max(tree.depths().values()))
        true_values["phi.treeLength"].append(tree.total_branch_length())
        
        # example code to extract other parameters from xml (e.g. when log file of generation not available)
        #if param in ["kappa", "shape"]:
        #    true_values[param].append(float(root.find(f".//parameter[@id='{param}']").text))

    return true_values

def extract_true_values_from_log(true_values):
    for i in range(1, 101):
        log_file = os.path.join(basePath, f"reps/rep{i}/yule-n100-{i}.log")
        df = pd.read_csv(log_file, comment='#', sep='\t')
        
        for param in parametersBEAST:
            if param in ["phi.height", "phi.treeLength"]:
                # these are not stored in the log file,
                # so have to be computed from the start tree
                continue
            if param.startswith("freq"):
                # LPhy and BEAST use different header names
                # so we need to use the mapping
                true_values[param].append(df[frequenciesMapping[param]].tolist()[0])
            else:
                true_values[param].append(df[param].tolist()[0])
            
    return true_values

# 3. Compute HPD intervals
def compute_hpd_intervals():
    print("Parse log file and compute HPD intervals")
    hpd_intervals = {param: {'lower': [], 'upper': []} for param in parametersBEAST}
    
    for i in range(1, 101):
        combined_data = {param: [] for param in parametersBEAST}
        
        for r in [1, 2]:
            log_file = os.path.join(basePath, f"reps/rep{i}/run{r}/yule-n100-{i}.log")
            df = pd.read_csv(log_file, comment='#', sep='\t')
            for param in parametersBEAST:
                combined_data[param].extend(df[param].tolist())
        
        for param in parametersBEAST:
            data = combined_data[param]
            lower, upper = compute_hpd(data)
            hpd_intervals[param]['lower'].append(lower)
            hpd_intervals[param]['upper'].append(upper)
    
    return hpd_intervals

def compute_hpd(data, level=0.95):
    sorted_data = np.sort(data)
    ci_index = int(np.floor(level * len(sorted_data)))
    n_intervals = len(sorted_data) - ci_index
    interval_width = sorted_data[ci_index:] - sorted_data[:n_intervals]
    min_index = np.argmin(interval_width)
    hpd_min = sorted_data[min_index]
    hpd_max = sorted_data[min_index + ci_index]
    return hpd_min, hpd_max

# 4. Construct and save plot
def plot_coverage(true_values, hpd_intervals):
    print("Construct plot")
    ig, axes = plt.subplots(plotRows, plotColumns, figsize=(15, 15))
    axes = axes.flatten()

    for ax, param, name in zip(axes, parametersBEAST, parametersPrint):
        true_vals = np.array(true_values[param])
        lower_bounds = np.array(hpd_intervals[param]['lower'])
        upper_bounds = np.array(hpd_intervals[param]['upper'])

        covered = [(lower <= true <= upper) for true, lower, upper in zip(true_vals, lower_bounds, upper_bounds)]
        coverage_percentage = sum(covered) / len(covered) * 100
        max_val = max(max(true_vals), max(upper_bounds))
        
        color_list = {True: 'g', False: 'r'}
        color_values = [color_list[x] for x in covered]      
        
        # Plotting
        ax.vlines(true_vals, ymin=lower_bounds, ymax=upper_bounds, colors=color_values, alpha=0.3, lw=3)
        ax.plot([0, max_val], [0, max_val], 'k--')
        ax.set_aspect('equal')
        ax.set_title(f'{name} (Coverage: {coverage_percentage:.2f}%)')
        ax.set_xlabel(f'True {name}')
        ax.set_ylabel(f'Estimated {name}')

    plt.suptitle(f'WCSS Coverage Plots for {simulationName}', fontsize=16)
    plt.tight_layout()
    plt.savefig(plotFileName)

print("# Create coverage plot #")
true_values = extract_true_values()
hpd_intervals = compute_hpd_intervals()
plot_coverage(true_values, hpd_intervals)
