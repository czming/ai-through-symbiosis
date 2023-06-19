import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm

import sys

sys.path.append("..")

from utils import *

configs = load_yaml_config("../configs/shivang.yaml")

data_folder = configs["file_paths"]["data_path"]

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def save_negative_picks():
    df = pd.read_csv(data_folder + '/video_items.csv')
    print(df.values)

    lists = df.values[:,1]

    fin_list = []
    for pick in lists:
        pick_len = len(pick)
        cur_order = []
        for i in range(0,len(pick),2):
            cur_order.append(pick[i])
        fin_list.append(cur_order)

    ids = df.values[:,0]
    fin_list = np.array(fin_list, dtype=object)
    all_negatives = []
    for pid, pick in zip(ids, fin_list):
        cur_negatives = []
        for idx in range(len(ids)):
            inter = intersection(pick, fin_list[idx])
            if len(inter) == 0:
                cur_negatives.append(ids[idx])
        all_negatives.append(cur_negatives)

    df['colors'] = fin_list
    df['negatives'] = all_negatives

    df.to_csv(data_folder+'/picklist_negatives.csv', index=False)

def prep_negative_pick_dataset():
    anchors = []
    positives = []
    negatives = []
    negs = pd.read_csv(data_folder+'/picklist_negatives.csv')
    print(negs.head())
    neg_values = negs.values
    for idx in tqdm(range(neg_values.shape[0])):
        cur_id = neg_values[idx][0]
        cur_negatives = json.loads(neg_values[idx][3])
        if not os.path.exists(data_folder+'/extracted_frames/picklist_'+str(cur_id)):
            continue
        anchor_images = os.listdir(data_folder+'/extracted_frames/picklist_'+str(cur_id))[:20]
        for anchor_img in anchor_images:
            anchor_img_name = anchor_img.split('.')[0]
            next_anchor_img = str(int(anchor_img_name)+1)+'.png'
            if os.path.isfile(data_folder+'/extracted_frames/picklist_'+str(cur_id)+'/'+next_anchor_img):
                for neg_id in cur_negatives:
                    if not os.path.exists(data_folder+'/extracted_frames/picklist_'+str(neg_id)):
                        continue
                    neg_images = sorted(os.listdir(data_folder+'/extracted_frames/picklist_'+str(neg_id)))
                    for neg_img in neg_images[:20]:
                        anchors.append(data_folder+'/extracted_frames/picklist_'+str(cur_id)+'/'+anchor_img)
                        positives.append(data_folder+'/extracted_frames/picklist_'+str(cur_id)+'/'+next_anchor_img)
                        negatives.append(data_folder+'/extracted_frames/picklist_'+str(neg_id)+'/'+neg_img)

    df = pd.DataFrame()
    df['anchor'] = anchors
    df['positive'] = positives
    df['negative'] = negatives
    df.to_csv(data_folder+'/triplet_dataset.csv', index=False)

if __name__ == '__main__':
    save_negative_picks()
    prep_negative_pick_dataset()

        

