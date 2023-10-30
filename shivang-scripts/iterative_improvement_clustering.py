import numpy as np
import argparse
import cv2
import math
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import metrics
import copy
import logging
import pickle
import collections
from models import Resnet18Triplet
import pandas as pd
import os
import torch
from tqdm import tqdm
from torchvision.io import read_image
from torchvision import transforms


np.random.seed(42)

PICKLISTS = list(range(136, 235))
htk_output_folder = 'data/testFolds'
htk_input_folder = 'data/htk_inputs'
pick_label_folder = 'data/Labels'
ckpt_path = 'model_training_checkpoints/model_resnet18_triplet_epoch_3.pt'
img_base_pth = 'data/extracted_frames_test/'

# {picklist_no: [index in objects_avg_hsv_bins for objects that are in this picklist]}
picklist_objects = collections.defaultdict(lambda: set())

# {object type: [index in objects_avg_hsv_bins for objects that are predicted to be of that type]
pred_objects = collections.defaultdict(lambda: set())

objects_pred = {}

combined_pick_labels = []

def get_model(pretrained=False, embedding_dimension=512):
    model = Resnet18Triplet(
        embedding_dimension=embedding_dimension,
        pretrained=pretrained
    )
    return model

def preprocess_img(img):
    image_transforms = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomApply([transforms.RandomResizedCrop((256,256))], p = 0.2),
        transforms.Resize((64, 64)),
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomVerticalFlip(0.3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.6071, 0.4609, 0.3944],
            std=[0.2457, 0.2175, 0.2129]
        )
    ])
    img = image_transforms(img)
    return img

def get_embedding(img, model):
    pre_img = preprocess_img(img)
    pre_img = torch.unsqueeze(pre_img, dim=0)
    pre_img = pre_img.cuda()
    emb = model(pre_img)
    return emb[0]

def gaussian_kernel1D(length, sigma):
    ax = np.linspace(-(length - 1) / 2., (length - 1) / 2., length)
    kernel = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    return kernel

def collapse_hue_bins(hue_bins, centers, sigma):
    # [0, 60, 120] for the centers, need to control for wraparound
    gaussian_kernel = gaussian_kernel1D(180, sigma)
    gaussian_kernel_one_side = gaussian_kernel[len(gaussian_kernel) // 2:]
    bin_sums = [0 for _ in range(len(centers))]
    for center_index, center in enumerate(centers):
        # this one needs to be treated specially otherwise would be summed twice
        bin_sums[center_index] += gaussian_kernel_one_side[0] * hue_bins[center]
        for i in range(1, 60):
            bin_sums[center_index] += gaussian_kernel_one_side[i] * hue_bins[(center + i) % len(hue_bins)]
            bin_sums[center_index] += gaussian_kernel_one_side[i] * hue_bins[center - i]
    return bin_sums

def get_all_picklist_unique_objects():
    df = pd.read_csv('../scripts/data/labeled_objects_test.csv')
    # df_new = pd.DataFrame()
    new_picks = []
    new_ids = []
    new_labels = []
    all_picklists = sorted(os.listdir('../scripts/data/Videos'))
    for picklist in tqdm(all_picklists):
        pick_name = picklist.split('.')[0]
        picklist_images = sorted(os.listdir('data/extracted_frames_test/'+pick_name))
        cur_dict = {}
        # print(picklist_images)
        for pick_img in picklist_images:
            img_id = int(pick_img.split('.')[0])
            # print(pick_name, img_id)
            obj_row = df.loc[(df['frame'] == img_id) & (df['picklist'] == pick_name )]
            label = obj_row['label'].values[0]
            # print(label)
            if label not in cur_dict:
                cur_dict[label] = img_id
                new_picks.append(pick_name)
                new_ids.append(img_id)
                new_labels.append(label)
    new_df = pd.DataFrame()
    new_df['picklist'] = new_picks
    new_df['frame'] = new_ids
    new_df['label'] = new_labels

    new_df.to_csv('data/labeled_objects_test.csv')

def get_all_object_embeddings():

    model = get_model()
    model = model.cuda()
    model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
    model.eval()

    obj_data = pd.read_csv('../scripts/data/labeled_objects_test.csv')
    print(obj_data.head())
    object_embeddings = []
    
    picklist_nos = []

    for idx in range(obj_data.shape[0]):
        print(idx)
        cur_obj = obj_data.iloc[idx]
        print(cur_obj)
        object_id = idx
        img_path = img_base_pth + cur_obj['picklist'] + '/' + str(cur_obj['frame']) + '.png'
        img = read_image(img_path)
        img_embedding = get_embedding(img, model)
        object_embeddings.append(img_embedding.cpu().detach().numpy())
        picklist_no = int(cur_obj['picklist'].split('_')[-1])
        cur_label = cur_obj['label']
        picklist_nos.append(picklist_no)
        picklist_objects[picklist_no].add(object_id)
        pred_objects[cur_label].add(object_id)
        objects_pred[object_id] = cur_label

    combined_pick_labels = list(obj_data['label'])

    object_embeddings = np.array(object_embeddings)
    objects_avg_hsv_bins = object_embeddings
    
    # print(picklist_objects)
    print(len(combined_pick_labels))
    # print(objects_pred)
    # print(object_embeddings.shape)
    return objects_avg_hsv_bins, combined_pick_labels
    


def iterative_clustering():
    # get_all_picklist_unique_objects()

    objects_avg_hsv_bins, combined_pick_labels = get_all_object_embeddings()

    print(objects_avg_hsv_bins.shape)
    # exit()

    color_mapping = {
        'r': '#ff0000',
        'g': '#009900',
        'b': '#000099',
        'p': '#0000ff',
        'q': '#00ff00',
        'o': '#FFA500',
        's': '#000000',
        'a': '#FFEA00',
        't': '#777777',
        'u': '#E1C16E'
    }

    epochs = 0
    num_epochs = 100

    cluster_fig = plt.figure()

    cluster_ax = cluster_fig.add_subplot(projection="3d")

    plt.show(block=False)

    while epochs < num_epochs:
        print (f"Starting epoch {epochs}...")
        # for each epoch, iterate through all the picklists and offer one swap
        for picklist_no, objects_in_picklist in picklist_objects.items():
            # check whether swapping the current element with any of the other

            # randomly pick an object
            object1_id = np.random.choice(list(picklist_objects[picklist_no]))

            # stores the reductions for the different object2s
            object2_distance_reduction = {}

            ## TODO: add controls on the number of objects looked at (if limiting the number of objects in the
            # picklist to compare to, just pick a bunch randomly to be looked at)
            for object2_id in objects_in_picklist:
                object1_pred = objects_pred[object1_id]
                object2_pred = objects_pred[object2_id]

                if objects_pred[object2_id] == objects_pred[object1_id]:
                    # they objects have the same prediction currently, no point swapping
                    continue

                # simple approach, look at the relative distance between the two points and the center of their
                # clusters, don't count themselves
                object1_pred_mean = np.array([objects_avg_hsv_bins[i] for i in pred_objects[object1_pred] if i != object1_id]).mean(axis=0)
                object2_pred_mean = np.array([objects_avg_hsv_bins[i] for i in pred_objects[object2_pred] if i != object2_id]).mean(axis=0)

                object1_pred_std = np.array([objects_avg_hsv_bins[i] for i in pred_objects[object1_pred] if i != object1_id]).std(axis=0, ddof=1)
                object2_pred_std = np.array([objects_avg_hsv_bins[i] for i in pred_objects[object2_pred] if i != object2_id]).std(axis=0, ddof=1)

                # should be proportional to log likelihood (assume normal, then take exponential of these / 2 to get pdf

                # current distance between the objects and their respective means, 0.00001 added for numerical stability
                curr_distance = (((object1_pred_mean - objects_avg_hsv_bins[object1_id]) ** 2) / (object1_pred_std + 0.0000001)).sum(axis=0) + \
                                (((object2_pred_mean - objects_avg_hsv_bins[object2_id]) ** 2) / (object2_pred_std + 0.0000001)).sum(axis=0)

                # distance if they were to swap
                new_distance = (((object2_pred_mean - objects_avg_hsv_bins[object1_id]) ** 2) / (object2_pred_std + 0.0000001)).sum(axis=0) + \
                                (((object1_pred_mean - objects_avg_hsv_bins[object2_id]) ** 2) / (object1_pred_std + 0.0000001)).sum(axis=0)

                distance_reduction = curr_distance - new_distance

                object2_distance_reduction[object2_id] = distance_reduction

            if len(object2_distance_reduction) == 0:
                # no object of different type
                continue

            # keep best distance reduction
            best_pos_object2 = list(object2_distance_reduction.keys())[0]

            for object2, distance_reduction in object2_distance_reduction.items():
                if distance_reduction > object2_distance_reduction[best_pos_object2]:
                    # found a better one
                    best_pos_object2 = object2



            if object2_distance_reduction[best_pos_object2] < 0:
                object2_distance_reduction_sum = sum(object2_distance_reduction.values())
                # all negative, need to pick one at random with decreasing probability based on the numbers,
                # take exponential of the distance reduction which should all be negative
                swap_object_id = np.random.choice(list(object2_distance_reduction.keys()), p=np.array([math.e ** (i / object2_distance_reduction_sum) for i in
                                            object2_distance_reduction.values()]) / sum([math.e ** (i / object2_distance_reduction_sum) for i in
                                            object2_distance_reduction.values()]))

                # give reducing odds of getting a random swap
                to_swap = np.random.choice([0, 1], p = [1 - math.e ** (-epochs/50), math.e ** (-epochs/50)])

                if not to_swap:
                    # to_swap is False so skip the rest of the assignment
                    continue


            else:
                # have the most positive distance reduction, just swap
                swap_object_id = best_pos_object2


                # remove randomness towards the end

                # only want to consider those with positive distance reduction
                object2_distance_reduction_positive = {key: value for key, value in object2_distance_reduction.items() if value > 0}

                # use to normalize the distances otherwise can get very small
                object2_distance_reduction_sum_positive = sum([i for i in object2_distance_reduction_positive.values()])

                # definitely swap, but some uncertainty about which element it is swapped with (decreasing temperature by adding epochs
                # as a factor)
                swap_object_id = np.random.choice(list(object2_distance_reduction_positive.keys()), p=np.array([math.e ** (i) for i in
                                            object2_distance_reduction_positive.values()]) / sum([math.e ** (i) for i in
                                            object2_distance_reduction_positive.values()]))


            # swap the objects, update objects_pred and pred_objects
            object1_prev_pred = objects_pred[object1_id]
            swap_object_prev_pred = objects_pred[swap_object_id]

            pred_objects[object1_prev_pred].remove(object1_id)
            pred_objects[swap_object_prev_pred].remove(swap_object_id)

            pred_objects[object1_prev_pred].add(swap_object_id)
            pred_objects[swap_object_prev_pred].add(object1_id)

            objects_pred[object1_id] = swap_object_prev_pred
            objects_pred[swap_object_id] = object1_prev_pred


        epochs += 1

        predicted_picklists = [objects_pred[i] for i in range(len(combined_pick_labels))]

        cluster_ax.clear()

        cluster_ax.scatter([i[0] for i in objects_avg_hsv_bins], [i[1] for i in objects_avg_hsv_bins], [i[2] for i in objects_avg_hsv_bins], \
                c = [color_mapping[i] for i in predicted_picklists])


        cluster_fig.canvas.draw()




    plt_display_index = 0
    fig, axs = plt.subplots(2, len(pred_objects) // 2)

    for object, predicted_objects in pred_objects.items():

        hsv_bins = np.array([objects_avg_hsv_bins[i] for i in predicted_objects]).mean(axis=0)

        if plt_display_index < len(pred_objects) // 2:
            axs[0, plt_display_index].bar(range(len(hsv_bins)), hsv_bins)
            axs[0, plt_display_index].set_title(object)
        else:
            axs[1, plt_display_index - len(pred_objects) // 2].bar(range(len(hsv_bins)), hsv_bins)
            axs[1, plt_display_index - len(pred_objects) // 2].set_title(object)
        plt_display_index += 1


    plt.show()





    # print the results on a per picklist level
    for picklist_no, objects_in_picklist in picklist_objects.items():
        print (f"Picklist no. {picklist_no}")
        picklist_pred = [objects_pred[i] for i in objects_in_picklist]
        print (f"Predicted labels: {picklist_pred}")
        # use the same mapping for combined_pick_labels as object ids
        picklist_gt = [combined_pick_labels[i] for i in objects_in_picklist]
        print (f"Actual labels:    {picklist_gt}")


    # flatten arrays
    actual_picklists = combined_pick_labels
    predicted_picklists = [objects_pred[i] for i in range(len(combined_pick_labels))]

    ax = plt.figure().add_subplot(projection='3d')

    ax.scatter([i[0] for i in objects_avg_hsv_bins], [i[1] for i in objects_avg_hsv_bins], [i[2] for i in objects_avg_hsv_bins], \
            c = [color_mapping[i] for i in predicted_picklists])

    plt.show()

    confusions = collections.defaultdict(int)
    label_counts = collections.defaultdict(int)
    for pred, label in zip(predicted_picklists, actual_picklists):
        if pred != label:
            confusions[pred + label] += 1

    print(confusions)


    confusion_matrix = metrics.confusion_matrix(actual_picklists, predicted_picklists)
    print(actual_picklists)
    print(predicted_picklists)
    from sklearn.utils.multiclass import unique_labels

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
    names = ['red', 'green', 'blue', 'darkblue', 'darkgreen', 'orange', 'alligatorclip', 'yellow', 'clear', 'candle']

    actual_picklists_names = []
    predicted_picklists_names = []

    letter_counts = collections.defaultdict(lambda: 0)
    for letter in actual_picklists:
        name = letter_to_name[letter]
        letter_counts[name] += 1
        actual_picklists_names.append(name)

    for index, letter in enumerate(predicted_picklists):
        # print(index)
        predicted_picklists_names.append(letter_to_name[letter])
    confusion_matrix = metrics.confusion_matrix(actual_picklists_names, predicted_picklists_names)
    conf_mat_norm = (confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis])

    unique_names = unique_labels(actual_picklists_names, predicted_picklists_names)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_mat_norm, display_labels = unique_names)

    cm_display.plot(cmap=plt.cm.Blues)

    plt.xticks(rotation=90)

    plt.savefig('results_triplet_test.png')

    plt.show()

    

    # epochs = 0
    # num_epochs = 500
    

if __name__ == '__main__':
    iterative_clustering()