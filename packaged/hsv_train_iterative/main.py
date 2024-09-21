from dotenv import load_dotenv
load_dotenv()

import os
import time
import logging
import traceback

import matplotlib.pyplot as plt
# import cv2
# import numpy as np
# import skimage.color
from google.protobuf.json_format import MessageToDict
# import mediapipe as mp
from models import CarryHSVHistogramModel,IterativeClusteringModel
import ast

from service import Service

import json
import numpy as np

model = CarryHSVHistogramModel()

htk_input_folder = "/shared/htk_inputs/"
htk_output_folder = "/shared/htk_outputs/"
pick_label_folder = "/shared/raw_labels/"

output_hist_file = "/viz/hist-iter.png"

def get_hsv_hist_fig(input_dict):
    letter_to_name = {
        'r': 'red',
        'g': 'green',
        'b': 'blue',
        'p': 'darkblue',
        'q': 'darkgreen',
        'o': 'orange',
        's': 'alligatorclip',
        'a': 'yellow',
        't': 'clear',
        'u': 'candle'
    }

    colors = {
        'g': 'green',
        'a': 'yellow',
        'u': 'tan',
        'b': 'blue',
        'q': 'darkgreen',
        'r': 'red',
        'o': 'orange',
        'p': 'darkblue',
        't': 'grey',
        's': 'black'
    }
    
    plt_display_index = 0
    fig, axs = plt.subplots(2, len(input_dict) // 2)

    for object, predicted_objects in input_dict.items():

        if plt_display_index < len(input_dict) // 2:
            axs[0, plt_display_index].bar(range(len(predicted_objects)), predicted_objects, color=colors[object])
            axs[0, plt_display_index].set_title(letter_to_name[object])
            axs[0, plt_display_index].set_ylim([-0.15, 0.15])
            axs[0, plt_display_index].set_yticks([])
            axs[0, plt_display_index].set_xticks([])

        else:
            axs[1, plt_display_index - len(input_dict) // 2].bar(range(len(predicted_objects)), predicted_objects, color=colors[object])
            axs[1, plt_display_index - len(input_dict) // 2].set_title(letter_to_name[object])
            axs[1, plt_display_index - len(input_dict) // 2].set_ylim([-0.15, 0.15])
            axs[1, plt_display_index - len(input_dict) // 2].set_yticks([])
            axs[1, plt_display_index - len(input_dict) // 2].set_xticks([])
        plt_display_index += 1

        axs[0, 0].set_yticks([-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15])
        axs[1, 0].set_yticks([-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15])

    return fig

def main(id, picklist_no, hsv_avg_mean, hsv_avg_std, predicted_labels):
    print (f"predicted labels: {predicted_labels}")
    predicted_labels = json.loads(predicted_labels)[picklist_no] # predicted_labels is {picklist_no: labels}
    hsv_avg_mean = json.loads(hsv_avg_mean)
    hsv_avg_std = json.loads(hsv_avg_std)
    

    # instantiate iterative clustering model and do fit iterative on HSVCarry model with that
    iterative_clustering_model = IterativeClusteringModel()

    iterative_clustering_model.class_hsv_bins_mean = {key: np.array(value) for key, value in hsv_avg_mean.items()}
    iterative_clustering_model.class_hsv_bins_std = {key: np.array(value) for key, value in hsv_avg_std.items()}

    model.iterative_clustering_model = iterative_clustering_model

    print (f"predicted labels: {predicted_labels}")

    hsv_avg_mean, hsv_avg_std = model.fit_iterative(int(picklist_no), htk_input_folder, htk_output_folder, predicted_labels, beta=0.9)[:2]

    # convert from np array to lists
    hsv_avg_mean = {key:list(value) for key, value in hsv_avg_mean.items()}
    hsv_avg_std = {key:list(value) for key, value in hsv_avg_std.items()}

    hsv_hist_fig = get_hsv_hist_fig(hsv_avg_mean)

    plt.savefig(output_hist_file)

    output = str((id, hsv_avg_mean, hsv_avg_std))
	
    print (output)

    # only want hsv avg mean and std, don't want the labels from the fit iterative output
    return output

service = Service(
		"hsv_train_iterative",
		lambda id, picklist_no, hsv_avg_mean, hsv_avg_std, predicted_labels: main(id, picklist_no, hsv_avg_mean, hsv_avg_std, predicted_labels),
		{	# take in a specific picklist_no
      		'form': ['id', 'picklist_no', 'hsv_avg_mean', 'hsv_avg_std', 'predicted_labels'], # form is a dictionary, then passing in a list of keys
		}
	).create_service(init_cors=True)
	
if __name__ == '__main__':
	service.run(
		host=os.environ.get("HOST", "0.0.0.0"),
		port=os.environ.get("PORT", 5000)
	)
