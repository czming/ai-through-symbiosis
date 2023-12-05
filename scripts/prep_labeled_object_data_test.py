import xml.etree.ElementTree as ET
from collections import defaultdict
import os
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# def get_list_colors():
#     df = pd.read_csv('../scripts/data/video_items.csv')
#     # print(df.values)
#
#     picklist_label_dict = {}
#
#     lists = df.values[:,1]
#     ids = df.values[:,0]
#
#     fin_list = []
#     for pid,pick in zip(ids, lists):
#         # print(pick)
#         pick_len = len(pick)
#         # print(pick_len)
#         cur_order = []
#         for i in range(0,len(pick),2):
#             # print(pick[i])
#             # try:
#             cur_order.append(pick[i])
#             # except:
#             #     cur_order.append('')
#         # print(cur_order)
#         fin_list.append(cur_order)
#         picklist_label_dict[pid] = cur_order
#
#     ids = df.values[:,0]
#     fin_list = np.array(fin_list, dtype=object)
#     return picklist_label_dict

def get_list_colors_new():
    picklist_label_dict = {}
    # use ground truth pick labels for testing instead of predicted pick labels from iterative clustering
    pick_label_path = 'deep_learning/data/gt_pick_labels/'
    for fil in os.listdir(pick_label_path):
        with open(pick_label_path+fil, 'r') as f:
            data = f.read()
            data = data.split(',')
            pick_no = int(data[0])
            pick_label = list(data[1].strip())
            print(pick_no, pick_label)
            picklist_label_dict[pick_no] = pick_label
    return picklist_label_dict

def get_elan_boundaries(file_name):
    # Passing the path of the xml document to enable the parsing process
    tree = ET.parse(file_name)

    # getting the parent tag of the xml document
    root = tree.getroot()
    elan_boundaries = defaultdict(list)
    elan_annotations = []
    for annotation in root[2][:]:
        all_descendants = list(annotation.iter())
        for desc in all_descendants:
            if desc.tag == "ANNOTATION_VALUE":
                if 'empty' not in desc.text:
                    elan_annotations.append(desc.text[0:desc.text.index("_")])
                else:
                    elan_annotations.append(desc.text)
    prev_time_value = int(root[1][1].attrib['TIME_VALUE'])/1000
    it = root[1][:]
    for index in range(0, len(it), 2):
        letter = elan_annotations[int(index/2)]
        letter_start = int(it[index].attrib['TIME_VALUE'])/1000
        letter_end = int(it[index + 1].attrib['TIME_VALUE'])/1000
        elan_boundaries[letter].append(letter_start)
        elan_boundaries[letter].append(letter_end)

    # remove the first two points from carry_empty because seems to have extra 2 elements for the period before the first pick starts (if first element is 0 then this
    # seems to be the case)
    elan_boundaries["carry_empty"] = elan_boundaries["carry_empty"][2:] if elan_boundaries["carry_empty"][0] == 0 else elan_boundaries["carry_empty"]

    return elan_boundaries

def get_htk_boundaries(file_name):

    htk_boundaries = defaultdict(list)
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines[2:]:
            if line.strip() == ".":
                # terminating char
                break
            boundaries = line.split()[0:2]
            letter = line.split()[2]
            letter_start = [int(boundary)/60000 for boundary in boundaries][0]
            letter_end = [int(boundary)/60000 for boundary in boundaries][1]
            htk_boundaries[letter].append(letter_start)
            htk_boundaries[letter].append(letter_end)

    return htk_boundaries

def get_hand_boundary(img):
  results = hands.process(img)
  # print(results.multi_hand_landmarks)
  try:
    landmarks = results.multi_hand_landmarks[0]
    # print(type(landmarks))
    h, w, c = img.shape
    x_max = 0
    y_max = 0
    x_min = w
    y_min = h
    # print(len(landmarks.landmark))
    if len(landmarks.landmark)==21:
      for lm in landmarks.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        if x > x_max:
            x_max = x
        if x < x_min:
            x_min = x
        if y > y_max:
            y_max = y
        if y < y_min:
            y_min = y
      # print(x_min, y_min, x_max, y_max)
      # cv2.rectangle(img_rgb, (x_min-10, y_min-10), (x_max+10, y_max+10), (0, 255, 0), 2)
      # print(img.shape)
      hand = img[y_min-150:y_max+150,x_min-150:x_max+150, :]
    #   hand_rgb = cv2.cvtColor(hand,cv2.COLOR_BGR2RGB)
    #   cv2.imwrite("/content/videoHands/hand%d.jpg" % count, hand_rgb)
      # count += 1
      # mpDraw.draw_landmarks(img_rgb, landmarks, mpHands.HAND_CONNECTIONS)
      # plt.imshow(hand)
      # plt.show()
      return hand
  except:
    pass

if __name__ == '__main__':
    picklist_label_dict = get_list_colors_new()
    print(picklist_label_dict)
    picklists = []
    frame_ids = []
    labels = []
    elan_files = []
    videos = sorted(os.listdir('deep_learning/data/Videos'))
    test_picklists = [
        'picklist_138',
        'picklist_201',
        'picklist_185',
        'picklist_153',
        'picklist_181',
        'picklist_199',
        'picklist_214',
        'picklist_188',
        'picklist_227',
        'picklist_215',
    ]
    for fil in test_picklists:
        fil = fil + '.mp4'
        # print(fil)
        fil_name = fil.split('.')[0]
        
        fil_id = int(fil_name[-3:])
        pick_id = fil_name.split('_')[1]
        # print(fil_id)
        # elan_fil = 'data/elan_annotated/' + fil_name + '.eaf'
        # boundaries = get_elan_boundaries(elan_fil)
        # carry_times = boundaries['carry']
        htk_file = '../htk_outputs/icassp_test_folds/avgFold/results-' + pick_id
        if not os.path.exists(htk_file):
            os.makedirs(htk_file)
        print(picklist_label_dict.keys())
        if int(pick_id) not in picklist_label_dict.keys():
            print('Not found:', pick_id)
            # print(type(pick_id))
            continue
        if not os.path.exists('deep_learning/data/extracted_frames_test/'+fil_name):
            os.makedirs('deep_learning/data/extracted_frames_test/'+fil_name)
        boundaries = get_htk_boundaries(htk_file)
        carry_times = boundaries['e']
        # print(carry_times)
        vid_fil = 'deep_learning/data/Videos/' + fil
        vidcap = cv2.VideoCapture(vid_fil)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        # print(fps)
        success,image = vidcap.read()
        count = 0
        success = True
        time_idx = 0
        while success:
            success,frame = vidcap.read()
            count+=1
            cur_time = count/fps
            if time_idx >= len(carry_times)-1:
                break
            if cur_time >= carry_times[time_idx] and cur_time <= carry_times[time_idx+1]:
                hand = get_hand_boundary(frame)
                if hand is not None:
                    print(hand.shape)
                    if hand.shape[0]>0 and hand.shape[1]>0:
                        cur_label = picklist_label_dict[fil_id][int(time_idx/2)]
                        print(fil_name, count, cur_label)
                        picklists.append(fil_name)
                        frame_ids.append(count)
                        labels.append(cur_label)
                        cv2.imwrite('deep_learning/data/extracted_frames_test/'+fil_name+'/'+str(count)+'.png', hand)
                        # print ('deep_learning/data/extracted_frames_test/'+fil_name+'/'+str(count)+'.png')
                        # plt.imsave('data/extracted_frames/'+fil_name+'/'+str(count)+'.png', hand)
                        print("time stamp current frame:",count/fps)
            elif cur_time > carry_times[time_idx+1]:
                time_idx += 2
                if time_idx > len(carry_times):
                    break
        # while success:
        #     success,frame = vidcap.read()
        #     count+=1
        #     cur_time = count/fps
        #     if time_idx >= len(carry_times)-1:
        #         break
        #     if cur_time >= carry_times[time_idx] and cur_time <= carry_times[time_idx+1]:
        #         hand = get_hand_boundary(frame)
        #         if hand is not None:
        #             print(hand.shape)
        #             if hand.shape[0]>0 and hand.shape[1]>0:
        #                 cv2.imwrite('data/extracted_frames_new/'+fil_name+'/'+str(count)+'.png', hand)
        #                 # plt.imsave('data/extracted_frames/'+fil_name+'/'+str(count)+'.png', hand)
        #                 print("time stamp current frame:",count/fps)
        #     elif cur_time > carry_times[time_idx+1]:
        #         time_idx += 2
        #         if time_idx > len(carry_times):
        #             break

    df = pd.DataFrame()
    df['picklist'] = picklists
    df['frame'] = frame_ids
    df['label'] = labels
    df.to_csv('deep_learning/data/labeled_objects_test.csv')

        

