from dotenv import load_dotenv
load_dotenv()

import os
import time
import logging
import traceback
import ast
import matplotlib.pyplot as plt

# import cv2
# import numpy as np
# import skimage.color
from google.protobuf.json_format import MessageToDict
# import mediapipe as mp
from models import CarryHSVHistogramModel

from service import Service

model = CarryHSVHistogramModel()

htk_input_folder = "/shared/htk_inputs/"
htk_output_folder = "/shared/htk_outputs/"
pick_label_folder = "/shared/raw_labels/"


output_hist_file = "./hist.png"

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

def main(id, picklist_nos):

	picklist_nos = ast.literal_eval(picklist_nos)

	hsv_avg_mean, hsv_avg_std = model.fit(picklist_nos, htk_input_folder, htk_output_folder, pick_label_folder)

	hsv_hist_fig = get_hsv_hist_fig(hsv_avg_mean)

	plt.savefig(output_hist_file)
	
	return str((id, hsv_avg_mean, hsv_avg_std))


service = Service(
		"hsv_train",
		lambda id, picklist_nos: main(id, picklist_nos),
		{
      		'form': ['id', 'picklist_nos'], # form is a dictionary, then passing in a list of keys
		}
	).create_service(init_cors=True)
if __name__ == '__main__':
	service.run(
		host=os.environ.get("HOST", "0.0.0.0"),
		port=os.environ.get("PORT", 5000)
	)
