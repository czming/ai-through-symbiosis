"""

script with the template to compare the data from a feature with ELAN annotations for some picklist(s). modified from iterative_boundaries.py

4/26/23 - currently being used to plot hand open/closed and hand landmark position features

"""

from utils import *
import argparse
import math
import matplotlib.pyplot as plt
from sklearn import metrics
import copy
import logging
from sklearn.utils.multiclass import unique_labels
import pickle
import os


#parses the fingers open/close vector for each frame 
def get_fingers_open_closed(file_name):
    with open(file_name, "r") as infile:
        data = infile.readlines()
        fingers_open_closed = [line.split(" ")[419:424] for line in data]
        fingers_open_closed = [[int(i) for i in _] for _ in fingers_open_closed]
        return fingers_open_closed
        

if __name__ == '__main__':

    PICKLISTS = [105, 138, 142, 177] #TODO add this to argparse later
    VID_FPS = 29.97 #TODO find out if this should be a global constant 
    ELAN_SCALE = 2
    # np.random.seed(42) #why?

    logging.basicConfig(level=logging.DEBUG)
    plt.set_loglevel (level = 'warning') #so our console isn't flooded with debugs


    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="/home/calix/Desktop/aits/ai-through-symbiosis/scripts/configs/calix.yaml",
                        help="Path to experiment config (scripts/configs)")
    args = parser.parse_args()

    configs = load_yaml_config(args.config)

    elan_label_folder = configs["file_paths"]["elan_annotated_file_path"]
    htk_input_folder = configs["file_paths"]["htk_input_file_path"]
    video_folder = configs["file_paths"]["video_file_path"]
    pick_label_folder = configs["file_paths"]["label_file_path"]


    
    # initialization (randomly assign colors)
    for picklist_no in PICKLISTS:
        logging.debug(f"Picklist number {picklist_no}")
        try:
            # load elan boundaries, so we can take average of each picks elan labels
            elan_label_file = os.path.join(f"{elan_label_folder}",f"picklist_{picklist_no}.eaf")
            elan_boundaries = rms_error_utils.get_elan_boundaries(elan_label_file)

            # total_time += elan_boundaries["carry_empty"][-1]
        except Exception as e:
            logging.critical(str(e))
            # no labels yet
            print("Skipping picklist: No elan boundaries")
            continue

        # get the htk_input to load the hsv bins from the relevant lines
        # with open(f"{htk_input_folder}/picklist_{picklist_no}") as infile:
        #     htk_inputs = [i.split() for i in infile.readlines()]

        fingers_open_closed = get_fingers_open_closed(os.path.join(str(htk_input_folder), f"picklist_{picklist_no}"))
        
        #temp: if we can see fewer than 3 fingers, the hand is closed, otherwise open
        open_fingers_feature = [[1 if sum(vec) > n else 0 for vec in fingers_open_closed] for n in range(5)] 


        # for ELAN labels
        # pick_labels = ["carry_red",
        #                "carry_blue",
        #                "carry_green",
        #                "carry_darkblue",
        #                "carry_darkgreen",
        #                "carry_clear",
        #                "carry_alligatorclip",
        #                "carry_yellow",
        #                "carry_orange",
        #                "carry_candle",
        #                ]

        print(f'elan boundaries: {elan_boundaries}')

        
        action_labels = ["carry_empty", "pick", "carry", "place"]

        # using elan boundaries
        all_action_frames = [] #[[carry empty frames], [pick frames], [carry frames], [place frames]]
        for action_label in action_labels:
            # look through each color
            action_frames = []

            for i in range(0, len(elan_boundaries[action_label]), 2):
                # collect the frames
                start_frame = math.ceil(float(elan_boundaries[action_label][i]) * VID_FPS * ELAN_SCALE)
                end_frame = math.ceil(float(elan_boundaries[action_label][i + 1]) * VID_FPS * ELAN_SCALE)
                action_frames.append([start_frame, end_frame])

            # sort based on start
            action_frames = sorted(action_frames, key=lambda x: x[0])
            # print(action_frames)
            # logging.debug(len(action_frames))
            
            #check for frame time overlap
            for i in range(len(action_frames) - 1):
                if action_frames[i + 1][0] <= action_frames[i][1]:
                    # start frame of subsequent pick is at or before the end of the current pick (there's an issue)
                    raise Exception("action timings are overlapping, check data")

            all_action_frames.append(action_frames)

        print(f'all_action_frames: {all_action_frames}')

        #consolidate ground truth action transition times
        elan_boundary_frame_times = []
        for boundary_list in elan_boundaries.values():
            for boundary_value in boundary_list:
                elan_boundary_frame_times.append(math.ceil(float(boundary_value) * VID_FPS * ELAN_SCALE))
        elan_boundary_frame_times = set(elan_boundary_frame_times)

        print(f'elan_boundary_frame_times: {elan_boundary_frame_times}')
            

        action_frame_times = sorted([v for sublist in all_action_frames for subsublist in sublist for v in subsublist])


        # Create a figure and axis object
        fig, axes = plt.subplots(nrows=5, ncols=1)



        for v in elan_boundary_frame_times:
            for ax in axes:
                ax.plot([v]*2, [0, 1], color='blue', linewidth=1)

        colors = ['r', 'y', 'g', 'b']
        for j in range(len(all_action_frames)):
            action_frames = all_action_frames[j]
            # print(action_frames)
            for i in range(len(action_frames)):
                frame_range = action_frames[i]
                for ax in axes:
                    ax.plot(frame_range, [0, 0], color = colors[j], linewidth = 3)
            

        for j in range(len(open_fingers_feature)):
            for i in range(len(open_fingers_feature[j])):
                hand_open = open_fingers_feature[j][i]
                if hand_open:
                    axes[j].plot(i, 1, color= 'r', marker ='o')
                else:
                    axes[j].plot(i, 1, color = 'b', marker = 'o')
        
        # Set axis labels and legend
        for ax in axes:
            ax.set_xlabel('Frame number')
            ax.set_ylabel('')
            ax.set_title(f'Picklist {picklist_no} fingers & ELAN comparison')




        # Show the plot
        plt.show()
      