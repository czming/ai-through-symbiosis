import os
import sys
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
from hand_utils import get_hand_boundary, get_hand_segmentation
sys.path.append(os.path.join(os.path.dirname('../../')))

from scripts.utils.rms_error_utils import get_elan_boundaries, get_htk_boundaries

import sys

sys.path.append("..")

from utils import *

configs = load_yaml_config("../configs/shivang.yaml")

data_folder = configs["file_paths"]["data_path"]

# import torch
# import torch.hub

# # Create the model
# model = torch.hub.load(
#     repo_or_dir='guglielmocamporese/hands-segmentation-pytorch', 
#     model='hand_segmentor', 
#     pretrained=True
# )

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(
    #     prog='extract_frames.py',
    #     description='DExtract hand frames from videos')
    # parser.add_argument('-d', '--data_folder', default='../../data') 
    # args = parser.parse_args()
    elan_files = []
    videos = sorted(os.listdir(data_folder + '/videos'))
    print(videos)
    for fil in videos:
        # print(fil)
        fil_name = fil.split('.')[0]
        pick_id = fil_name.split('_')[1]
        # print(pick_id)
        # exit()
        htk_file = data_folder + '/avgFold/train_avg_fold/results-' + pick_id
        if not os.path.exists(htk_file):
            continue
        if not os.path.exists(data_folder + '/extracted_frames/' + fil_name):
            os.makedirs(data_folder + '/extracted_frames/' + fil_name)
        else:
            continue
        if not os.path.exists(data_folder + '/extracted_hands/' + fil_name):
            os.makedirs(data_folder + '/extracted_hands/' + fil_name)
        else:
            continue
        # elan_fil = 'data/elan_annotated/' + fil_name + '.eaf'
        # boundaries = get_elan_boundaries(elan_fil)
        print(pick_id)
        boundaries = get_htk_boundaries(htk_file)
        print(boundaries)
        # exit()
        # carry_times = boundaries['carry']
        carry_times = boundaries['e']
        print(carry_times)
        vid_fil = data_folder + '/videos/' + fil
        vidcap = cv2.VideoCapture(vid_fil)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        print(fps)
        success,image = vidcap.read()
        count = 0
        success = True
        time_idx = 0
        while success:
            success,frame = vidcap.read()
            frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5) 
            count+=1
            # print(frame.shape)
            # print(data_folder + '/extracted_frames/'+fil_name+'/'+str(count)+'.png')
            # plt.imsave('data/extracted_frames/'+fil_name+'/'+str(count)+'.png', hand)
            # if count % 50 == 0:
            cur_time = count/fps
            if time_idx >= len(carry_times)-1:
                break
            # print(count)
            
            # break
            if cur_time >= carry_times[time_idx] and cur_time <= carry_times[time_idx+1]:
                try:
                    os.mkdir(data_folder + '/extracted_hands/'+fil_name)
                    os.mkdir(data_folder + '/extracted_seg/'+fil_name)
                except:
                    pass
                hand = get_hand_boundary(frame)
                if hand is not None:
                    print(hand.shape)
                    if hand.shape[0]>0 and hand.shape[1]>0:
                        # hand_seg = get_hand_segmentation(frame)
                        print(data_folder + '/extracted_hands/'+fil_name+'/'+str(count)+'.png')
                        cv2.imwrite(data_folder + '/extracted_hands/'+fil_name+'/'+str(count)+'.png', hand)
                        # cv2.imwrite(data_folder + '/extracted_seg/'+fil_name+'/'+str(count)+'.png', hand_seg)
                        cv2.imwrite(data_folder + '/extracted_frames/'+fil_name+'/'+str(count)+'.png', frame)
                        # plt.imsave('data/extracted_frames/'+fil_name+'/'+str(count)+'.png', hand)
                        print("time stamp current frame:",count/fps)
            elif cur_time > carry_times[time_idx+1]:
                time_idx += 2
                if time_idx > len(carry_times):
                    break