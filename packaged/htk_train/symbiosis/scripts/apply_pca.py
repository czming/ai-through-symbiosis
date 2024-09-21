#!/usr/bin/python
from sklearn.decomposition import PCA
from sklearn import preprocessing
import pickle as pk
import numpy as np
import glob
import os


data_dir = ""
output_dir = "./pca-data/"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
mmscaler = preprocessing.MinMaxScaler()
pca_reload = pk.load(open("./pca.pkl",'rb'))
mmscaler_reload = pk.load(open("./mmscaler.pkl", 'rb'))
np.set_printoptions(suppress=True)


data_files = glob.glob("./all-data/*")

print(len(data_files))
for num, data_file in enumerate(data_files):
    print(num)
    print(data_file)
    with open(data_file) as fp:
        with open(os.path.join(output_dir, os.path.basename(data_file)), 'a') as f:
            for line in fp:
                frame = np.asarray(line.split(), dtype=float)
                if len(frame) > 0:
                    pca_frame = pca_reload.transform(np.expand_dims(frame, 0))
                    mm_frame = mmscaler_reload.transform(pca_frame)
                    new_frame = str(mm_frame).replace("[[","").replace("]]","").replace("\n","")
                    
                    f.write(new_frame + "\n")



#To run on multiple files

#for f in ./*
#do
#cat $f | python3 apply_pca.py >> ../phrase_data_pca/${f:2}
#done

#for f in ./*
#do
#cat $f | python3 ../../../apply_pca.py >> ../../../test_on_thad/training/al#ex_${f:2} 
#done
