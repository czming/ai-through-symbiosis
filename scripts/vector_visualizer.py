

# read from file
from importlib.metadata import distribution
import numpy as np 
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# read all files in folder
def read_folder(folder):
    files = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            files.append(filename)
    return files

# read all lines in file
def read_file(filename):
    with open(filename) as f:
        lines = f.readlines()
    return lines

# show htk boundaries through time (last script)
# show vector representation through time
# (this comparison should indicate if our vectors are discriminative)
# picklist_num = '19'
picklist_nums = ['17','18','19','20', '21', '22', '26']
for picklist_num in picklist_nums:
    vectors = read_file("./data-binned-aruco-2-optical/picklist_" + picklist_num)
    vectors = read_file("../../data-forward/picklist_" + picklist_num)
    tree = ET.parse('./elan_annotated/picklist_'+picklist_num+'.eaf')

    # getting the parent tag of
    # the xml document
    root = tree.getroot()
    # elan_boundaries = defaultdict(list)
    elan_annotations = []
    # for annotation in root[2][:]:
    #     all_descendants = list(annotation.iter())
    #     for desc in all_descendants:
    #         if desc.tag == "ANNOTATION_VALUE":
    #             print(desc.text)
    it = root[1][:]
    event_times = [0]

    for index in it:
        event_times.append(int(index.attrib['TIME_VALUE'])/1000)
    event_times.append(len(vectors)/60) #60 frames per second


    aruco_vectors = []
    optical_flow_vectors_x = []
    optical_flow_vectors_y = []
    for vector in vectors:
        #seperate first and rest dimensions of vector
        vector = vector.split(' ')
        optical_flow_vector_x = vector[1]
        optical_flow_vector_y = vector[2]
        optical_flow_vectors_x.append(float(optical_flow_vector_x))
        optical_flow_vectors_y.append(float(optical_flow_vector_y))
        aruco_vector = vector[0]
        aruco_vectors.append(float(aruco_vector))



        labels = ["./results-aruco/results-" + picklist_num]
        # labels = ["./results-25dims/results-" + picklist_num]
        htk_boundaries = []
        for label in labels:
            with open(label, 'r') as f:
                lines = f.readlines()[1:-1]
                for line in lines[1:]:
                    boundaries = line.split()[0:2]
                    letter = line.split()[2]
                    letter_start = [int(boundary)/117000 for boundary in boundaries][0]
                    letter_end = [int(boundary)/117000 for boundary in boundaries][1]
                    htk_boundaries.append(letter_start)
                    htk_boundaries.append(letter_end)
                    final_end = letter_end
        event_times.append(final_end)




    # maptlot libplot of aruco vectors, optical flow vectors, and event times over time
    # matplot libtplot 3 subplots
    fig, axs = plt.subplots(5, 1, constrained_layout=True)
    axs[2].plot(range(0,len(vectors)), aruco_vectors, 'r', label='aruco')
    axs[1].plot(range(0,len(vectors)), optical_flow_vectors_x, 'b', label='optical flow x')
    axs[0].plot(range(0,len(vectors)), optical_flow_vectors_y, 'g', label='optical flow y')
    event_times = [i for i in event_times]
    event_pair = [1 for i in event_times]
    htk_pair = [1 for i in htk_boundaries]
    #matplotlib scatter plot with 1 for event times
    axs[3].scatter(event_times, event_pair)
    axs[4].scatter(htk_boundaries, htk_pair)
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    # axs[2].set_ylim([min(aruco_vectors),max(aruco_vectors)])
    plt.show()
    # plt.savefig('vector_visualizer-' + picklist_num + '.png')
