from torch.utils.data import Dataset
import pandas as pd
# import cv2
from PIL import Image
from torchvision.io import read_image
from sklearn.preprocessing import OneHotEncoder
import torch

class EgoObjectDataset(Dataset):
    def __init__(self, dataset_path, transform):
        self.data = pd.read_csv(dataset_path)
        self.anchor_images = self.data['anchor'].values
        self.positive_images = self.data['positive'].values
        self.negative_images = self.data['negative'].values
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        anchor_image = read_image(self.anchor_images[idx])
        positive_image = read_image(self.positive_images[idx])
        negative_image = read_image(self.negative_images[idx])

        anchor_image = self.transform(anchor_image)
        positive_image = self.transform(positive_image)
        negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image


class EgoObjectClassificationDataset(Dataset):
    def __init__(self, dataset_path, data_folder, transform, test=False, numbers=False, picklists=None):
        self.data = pd.read_csv(dataset_path)
        self.numbers = numbers
        self.data_folder = data_folder

        def letter_to_name(val):
            dic = {
                'r': 0,
                'g': 1,
                'b': 2,
                'p': 3,
                'q': 4,
                'o': 5,
                's': 6,
                'a': 7,
                't': 8,
                'u': 9
            }
            return dic[val]

        # self.onehot_encoder = OneHotEncoder()
        if picklists is None:
            self.picklist = self.data['picklist'].values
            self.frames = self.data['frame'].values
            self.labels = self.data['label'].apply(letter_to_name).values
            self.test = test
            # print(self.labels)
            # self.onehot_encoded_labels = self.onehot_encoder.fit_transform(self.labels)
            self.transform = transform
        else:
            # unique_picklists = sorted(list(set(self.data['picklist'].values)))
            # print("LENGTH")
            # print(len(unique_picklists))
            # kept_picklists = [unique_picklists[i] for i in inds]
            picklists = [f'picklist_{no}' for no in picklists]
            self.data = self.data[self.data['picklist'].isin(picklists)]

            print(self.data.groupby('label')['label'].count())
            print(self.data.groupby('picklist')['picklist'].count())
            self.picklist = self.data['picklist'].values
            self.frames = self.data['frame'].values
            self.labels = self.data['label'].apply(letter_to_name).values
            self.test = test
            # print(self.labels)
            # self.onehot_encoded_labels = self.onehot_encoder.fit_transform(self.labels)
            self.transform = transform
            
        # print("PICKLIST: ")
        # from collections import Counter
        # print(Counter(self.picklist))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.test:
            image = read_image(
                f'{self.data_folder}/extracted_frames_test/' + self.picklist[idx] + '/' + str(self.frames[idx]) + '.png')
        else:
            image = read_image(f'{self.data_folder}/extracted_frames_new/' + self.picklist[idx] + '/' + str(self.frames[idx]) + '.png')
        label = self.labels[idx]
        image = self.transform(image)
        if not self.numbers:
            return image, label
        else:
            return (image, self.frames[idx], self.picklist[idx]), label
