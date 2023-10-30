import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def save_negative_picks():
    df = pd.read_csv('../scripts/data/video_items.csv')
    print(df.values)

    lists = df.values[:,1]

    fin_list = []
    for pick in lists:
        # print(pick)
        pick_len = len(pick)
        # print(pick_len)
        cur_order = []
        for i in range(0,len(pick),2):
            # print(pick[i])
            # try:
            cur_order.append(pick[i])
            # except:
            #     cur_order.append('')
        # print(cur_order)
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
        # print(pid, pick)
    # print(all_negatives)

    df['colors'] = fin_list
    df['negatives'] = all_negatives

    df.to_csv('data/picklist_negatives.csv', index=False)

def prep_negative_pick_dataset():
    anchors = []
    positives = []
    negatives = []
    negs = pd.read_csv('../scripts/data/picklist_negatives.csv')
    print(negs.head())
    neg_values = negs.values
    for idx in tqdm(range(neg_values.shape[0])):
        # print(neg_values[idx])
        cur_id = neg_values[idx][0]
        cur_negatives = json.loads(neg_values[idx][3])
        # print(cur_negatives)
        if not os.path.exists('data/extracted_frames/picklist_'+str(cur_id)):
            continue
        anchor_images = os.listdir('data/extracted_frames/picklist_'+str(cur_id))[:20]
        # print(anchor_images)
        for anchor_img in anchor_images:
            anchor_img_name = anchor_img.split('.')[0]
            # print(anchor_img_name)
            next_anchor_img = str(int(anchor_img_name)+1)+'.png'
            if os.path.isfile('data/extracted_frames/picklist_'+str(cur_id)+'/'+next_anchor_img):
                # print(anchor_img, next_anchor_img)
                for neg_id in cur_negatives:
                    if not os.path.exists('data/extracted_frames/picklist_'+str(neg_id)):
                        continue
                    neg_images = sorted(os.listdir('data/extracted_frames/picklist_'+str(neg_id)))
                    # print(len(neg_images))
                    for neg_img in neg_images[:20]:
                        anchors.append('data/extracted_frames/picklist_'+str(cur_id)+'/'+anchor_img)
                        positives.append('data/extracted_frames/picklist_'+str(cur_id)+'/'+next_anchor_img)
                        negatives.append('data/extracted_frames/picklist_'+str(neg_id)+'/'+neg_img)

    # print(len(anchors), len(positives), len(negatives))
    df = pd.DataFrame()
    df['anchor'] = anchors
    df['positive'] = positives
    df['negative'] = negatives
    df.to_csv('data/final_dataset.csv', index=False)


if __name__ == "__main__":
    # save_negative_picks()
    prep_negative_pick_dataset()
    # data = pd.read_csv('data/final_dataset.csv')
    # print(data.shape)
