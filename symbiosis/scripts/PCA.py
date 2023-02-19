#!/usr/bin/python
import numpy as np
from string import ascii_uppercase
from sklearn import preprocessing
from sklearn.decomposition import PCA
import os
import pickle as pk

pca_components = 6
pca = PCA(pca_components)
mmscaler = preprocessing.MinMaxScaler()
file_directory1 = "./pca-training-data/"
file_list1 = os.listdir(file_directory1)

print(len(file_list1))
for i in range(0, len(file_list1)):
    file_list1[i] = file_directory1 + file_list1[i]

file_list = file_list1
vector_dim = 25

all_files = np.zeros((vector_dim,0))
total_length = 0
file_length = []
file_names = []
for filename in file_list:
    # print(filename) 
    curr_file = open(filename, 'r') 
    Lines = curr_file.readlines()
    file_length.append(len(Lines))
    file_names.append(filename)
    # print(len(Lines))
    all_frames = np.zeros((vector_dim,0))
    for line in range(0,len(Lines)):
        total_length += 1
        curr_line = np.asarray([float(x) for x in Lines[line].split()]) #line as float values in nparray
        curr_line = np.expand_dims(curr_line,1)
        all_frames = np.concatenate((all_frames, curr_line),1)
    all_files = np.concatenate((all_files[:, 0:all_files.shape[1]], all_frames), 1)
# print(total_length)    
    
all_files = all_files[:, 0:all_files.shape[1]]
all_files = np.transpose(all_files)
print(all_files.shape)    
pcaed_files = pca.fit_transform(all_files)
print(pcaed_files.shape)
new_files = mmscaler.fit_transform(pcaed_files)
print(new_files.shape)
pk.dump(pca, open("pca.pkl","wb"))
pk.dump(mmscaler, open("mmscaler.pkl","wb"))

