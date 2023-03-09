# import pandas as pd

# data = pd.read_csv("""file_path""", delimiter = " ")

import numpy as np
import math
import logging
import matplotlib.pyplot as plt

import sys
 
# setting path
sys.path.append('..')

from utils import *
from visualize_htk_inputs import plot_axs_columns 

def reflect_convolve(data, convolution_filter):

    convolution_size = len(convolution_filter)

    # normalize such that everything sums to 1
    convolution_filter_normalized = np.array(convolution_filter) / np.sum(convolution_filter)

    # adding padding for same convolution
    padded_data = np.pad(data, pad_width=(
        # if we take the convolution cenetered on element as the left of center
        # we'll need this much padding (padding more on right)
        ((convolution_size - 1) // 2, convolution_size // 2),
        (0, 0),
        ), mode="reflect")

    convolved_data = np.zeros(data.shape)

    for i in range(data.shape[1]):
        # number of columns
        convolved_data[:, i] = np.convolve(convolution_filter_normalized, padded_data[:, i], mode="valid")

    return convolved_data

def gaussian_kernel1D(length, sigma):
    ax = np.linspace(-(length - 1) / 2., (length - 1) / 2., length)
    kernel = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    return kernel


def average_kernel1D(length):
    return [1/length] * length

configs = load_yaml_config("../configs/jon.yaml")

htk_input_folder = configs["file_paths"]["htk_input_file_path"]

logging.getLogger().setLevel("INFO")

# choose odd number of elements so there's a center element, otherwise we'll use the left
# element as the center element for the convolutions
# CONVOLUTION_FILTER = [1, 6, 15, 20, 15, 6, 1]
length = 200
sigma = 3
convolution_filter = gaussian_kernel1D(length, sigma)
convolution_filter = average_kernel1D(length)
# choose the index of the columns that we want to visualize
VISUALIZED_COLUMNS = [0, 1, 2,3,4,5]

# columns that we want to apply the filter to
FILTER_COLUMNS = [0, 1, 2, 3, 4, 5]


for index in range(135, 235):

    try:
        open(f"""{htk_input_folder}/picklist_{str(index)}""")

    except:
        print("Skipping picklist" + str(index))
        continue

    file_name = f"""{htk_input_folder}/picklist_{str(index)}"""

    data = np.genfromtxt(file_name, delimiter=" ")

    # make a copy
    convolved_data = np.array(data)

    convolved_data[:, FILTER_COLUMNS] = reflect_convolve(data[:, FILTER_COLUMNS], convolution_filter)

    # make sure the lengths are the same
    assert len(data) == len(convolved_data)

    print(file_name + f"_gaussian_filter_{length}_{sigma}.txt")
    # np.savetxt(file_name + f"_gaussian_filter_{length}_{sigma}.txt", convolved_data,
    #            delimiter=" ")

    # show visualization of data
    fig, axs = plt.subplots(len(VISUALIZED_COLUMNS), 2)

    if len(VISUALIZED_COLUMNS) == 1:
        # axs loses that dimension if only one element in that dimension
        old_data = data[:, VISUALIZED_COLUMNS[0]]
        axs[0].plot(range(len(data)), old_data)
        filtered_data = convolved_data[:, VISUALIZED_COLUMNS[0]]
        axs[1].plot(range(len(data)), filtered_data)

        axs[0].set_title("Original data")
        axs[1].set_title("Filtered data")
        
    else:
        for i, column in enumerate(VISUALIZED_COLUMNS):
            old_data = data[:, column]
            axs[i, 0].plot(range(len(data)), old_data)
            filtered_data = convolved_data[:, column]
            axs[i, 1].plot(range(len(data)), filtered_data)

        axs[0, 0].set_title("Original data")
        axs[0, 1].set_title("Filtered data")

    fig.show()
    fig.waitforbuttonpress()
