import matplotlib.pyplot as plt
import numpy as np
import argparse

'''
Script to plot the clades based on their size and rogue rank.
Assumed data format is no header and that each row starts with the clade size and
then lists the rogue scores, sorted, of all clades of that size; so:
cladeSize 1st 2nd 3rd 4th ...
1	1.0986122886681096	1.0986122886681096	0.0	0.0 
2	0.5493061443340548	0.5493061443340548
3	0.36620409622270317	0.36620409622270317
...
'''

def main():
    # Hardcoded defaults
    file_name = "../examples/RSV2-rogueRanks.csv"
    save_file = "../examples/RSV2-rogueRanksPlot.pdf"
    max_clade_size = 5

    plot_width = 5
    plot_height = 4
    max_tick_threshold = 16

    parser = argparse.ArgumentParser(description="Rogue Rank Plotter")

    # Add arguments
    parser.add_argument('--input_file', type=str, help="Path to the input file.")
    parser.add_argument('--save_file', type=str, help="Path to save the output file.")
    parser.add_argument('--max_clade_size', type=int, help="Maximum clade size.", default=max_clade_size)
    parser.add_argument('--plot_width', type=int, help="Plot width.", default=plot_width)
    parser.add_argument('--plot_height', type=int, help="Plot height.", default=plot_height)
    parser.add_argument('--max_tick_threshold', type=int, help="Maximum x ticks.", default=max_tick_threshold)

    # Parse arguments
    args = parser.parse_args()

    # Check and assign the provided arguments, or fall back to defaults
    input_file = args.input_file if args.input_file else file_name
    save_file = args.save_file if args.save_file else save_file
    max_clade_size = args.max_clade_size if args.max_clade_size else max_clade_size

    # Output the values to verify
    print(f"Input file: {input_file}")
    print(f"Save file: {save_file}")
    print(f"Max clade size: {max_clade_size}")

    plot_ranks(file_name, save_file, max_clade_size, plot_width, plot_height, max_tick_threshold)

def plot_ranks(input_file: str, save_file: str, max_clade_size=5, plot_width=5, plot_height=4, max_tick_threshold=16):
    plt.figure(figsize=(plot_width, plot_height))

    max_length = 1  # longest list of ranks
    num_rows = 0
    max_value = 0
    min_value = 10000000
    lines = []

    with open(input_file, 'r') as file:
        for i, line in enumerate(file):
            row = line.strip().split('\t')
            label = row[0]  # First element as the label
            try:
                clade_size = int(row[0])
            except ValueError:
                # if not an integer, might be header, so skip this line
                continue
            if clade_size > max_clade_size:
                # don't care about clades larger than specified max size
                break

            values = list(map(float, row[1:]))  # convert rest of line to list of floats
            max_length = max(max_length, len(values))
            min_value = min(min_value, min(values))
            max_value = max(max_value, max(values))

            lines.append((label, values))

    # store lines, colors, and labels so that we can specify order and color mapping in the legend
    # so while we plot the lines first...
    plot_lines = []
    colors = []
    labels = []
    for label, values in reversed(lines):
        x = np.arange(len(values))
        y = values
        line, = plt.plot(x, y, marker='o', markersize=3)
        plot_lines.append(line)
        colors.append(line.get_color())
        labels.append(label)
        num_rows += 1

    # ... we then remove the existing lines again ...
    current_axes = plt.gca()
    for line in current_axes.lines:
        line.remove()

    # ... and re-plot in reverse order with stored colors
    for (label, values), color in zip(reversed(lines), reversed(colors)):
        x = np.arange(len(values))
        y = values
        plt.plot(x, y, color=color, marker='o', markersize=3, label=label)

    # x and y axes
    plt.xlabel('Rank')
    ticks_and_lims(max_length, min_value, max_value, max_tick_threshold)
    plt.ylabel('Clade Rogue Score [log unit]')

    # legend - create custom legend handles with the correct colors
    handles = [plt.Line2D([0], [0], color=color, marker='o', markersize=3) for color in colors]
    plt.legend(handles, reversed(labels), ncol=max(1, num_rows // 15), title="Clade Size")

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_file, bbox_inches='tight', pad_inches=0)
    plt.show()

def ticks_and_lims(max_length: int, min_value: int, max_value: int, max_tick_threshold: int):
    # method tries to guess an appropriate number of x ticks
    max_tick = max_length
    if max_tick < max_tick_threshold:
        # when the longest rank line is shorter than the threshold, plot all
        plt.xticks(ticks=np.arange(max_tick), labels=np.arange(1, max_tick + 1))
        step = 1
    else:
        if max_tick % max_tick_threshold != 0:
            max_tick = max_tick + (max_tick_threshold - max_tick % max_tick_threshold)
        step = max_tick // max_tick_threshold
        if step % 5 != 0:
            step = step + (5 - step % 5)

        ticks = np.concatenate(([0], np.arange(step - 1, max_tick, step)))
        labels = np.concatenate(([1], np.arange(step, max_tick + 1, step)))
        plt.xticks(ticks=ticks, labels=labels)

    plt.xlim(- step / 3, max_length + step / 3)
    plt.ylim(np.min(min_value, 0) - 0.1, max_value + 0.1)

if __name__ == "__main__":
    main()
