"""

visualizes data with the x axis as the row and the y-axis as the selected
column values

"""

import numpy as np
import matplotlib.pyplot as plt
from utils import *

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
        axs.plot(range(len(data)), data[:, columns[0]])
    else:
        for i, column in enumerate(columns):
            curr_data = data[:, column]

            # axs is 1D if only one dimension is used (even if the second dimension exists but is just 1,
            # which is the case here)
            axs[i].plot(range(len(data)), curr_data)
    

if __name__ == "__main__":
    configs = load_yaml_config("configs/zm.yaml")

    # choose the index of the columns that we want to visualize
    VISUALIZED_COLUMNS = [0] #list(range(21))

    htk_input_folder = configs["file_paths"]["htk_input_file_path"]

    FILE_NAME = f"""{htk_input_folder}/picklist_44_forward_filled_30_gaussian_filter_9_3.txt"""

    fig, axs = get_visualization_plot(np.genfromtxt(FILE_NAME, delimiter = " "), VISUALIZED_COLUMNS)

    plt.show()
