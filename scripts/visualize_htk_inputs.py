"""

visualizes data with the x axis as the row and the y-axis as the selected
column values

"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

def get_visualization_plot(data, columns):
    """
    takes in file name for the data and columns to be visualized and returns fig and axs with the visualized data
    """
    fig, axs = plt.subplots(len(columns), 1)

    plot_axs_columns(fig, axs, data, columns)

    return fig, axs

def plot_axs_columns(fig, axs, data, columns):

    if len(columns) == 1:
        # axs will be a single element if only one column visualized
        curr_data = data[:, columns[0]]
        axs.plot(range(len(data)), curr_data)
    else:
        for i, column in enumerate(columns):
            curr_data = data[:, column]

            # axs is 1D if only one dimension is used (even if the second dimension exists but is just 1,
            # which is the case here)
            axs[i].plot(range(len(data)), curr_data)
    

if __name__ == "__main__":

    # choose the index of the columns that we want to visualize
    VISUALIZED_COLUMNS = [0] #list(range(21))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--htk_input",
        required=True,
        help="Relative or absolute path to htk input file for a picklist",
    )
    args = parser.parse_args()
    FILE_NAME = args.htk_input
    print(VISUALIZED_COLUMNS)
    fig, axs = get_visualization_plot(np.genfromtxt(FILE_NAME, delimiter = " "), VISUALIZED_COLUMNS)
    fig.show()
    fig.waitforbuttonpress()
