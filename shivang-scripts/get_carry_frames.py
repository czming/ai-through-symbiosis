import xml.etree.ElementTree as ET
from collections import defaultdict
import os
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

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

if __name__=='__main__':
    elan_files = []
    videos = sorted(os.listdir('../scripts/data/Videos'))
    for fil in videos:
        # print(fil)
        fil_name = fil.split('.')[0]
        if not os.path.exists('data/extracted_frames/'+fil_name):
            os.makedirs('data/extracted_frames/'+fil_name)
        else:
            continue
        elan_fil = 'data/elan_annotated/' + fil_name + '.eaf'
        boundaries = get_elan_boundaries(elan_fil)
        carry_times = boundaries['carry']
        print(carry_times)
        vid_fil = 'data/Videos/' + fil
        vidcap = cv2.VideoCapture(vid_fil)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        print(fps)
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
                        cv2.imwrite('data/extracted_frames/'+fil_name+'/'+str(count)+'.png', hand)
                        # plt.imsave('data/extracted_frames/'+fil_name+'/'+str(count)+'.png', hand)
                        print("time stamp current frame:",count/fps)
            elif cur_time > carry_times[time_idx+1]:
                time_idx += 2
                if time_idx > len(carry_times):
                    break


