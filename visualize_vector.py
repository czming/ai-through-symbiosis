import numpy as np
import matplotlib.pyplot as plt

def read_file(filename):
    with open(filename) as f:
        lines = f.readlines()
    return lines


# lines = read_file("../htk_inputs/picklist_1_binned_aruco_gaussian_filter_9_3.txt")
for file_num in range(1,41):
    filename = "../8data/picklist_" + str(file_num)
    lines = read_file(filename)
    A = np.loadtxt(lines, delimiter=' ')

    columns = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    # columns = [0,1,7,11,14,15,23,24]


    columns = [0,1,2,3,4,5,6,7]

    fig, axs = plt.subplots(len(columns))
    fig.suptitle('Selected features for an entire picklist duration')

    for matplot_indice, column_num in enumerate(columns):
        column = A[:, column_num]
        axs[matplot_indice].plot(np.arange(0, len(column)), column)


    # plt.show()
    plt.savefig(str(file_num) + ".png")
    