import sys

sys.path.append("..")

from utils import *

import base64
import urllib.request
import pandas as pd
import os
import argparse

configs = load_yaml_config("../configs/shivang.yaml")

data_folder = configs["file_paths"]["data_path"]
video_folder = configs["file_paths"]["video_file_path"]

def create_onedrive_directdownload (onedrive_link):
    data_bytes64 = base64.b64encode(bytes(onedrive_link, 'utf-8'))
    data_bytes64_String = data_bytes64.decode('utf-8').replace('/','_').replace('+','-').rstrip("=")
    resultUrl = f"https://api.onedrive.com/v1.0/shares/u!{data_bytes64_String}/root/content"
    return resultUrl

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(
    #     prog='download_videos.py',
    #     description='Download Egocentric Video Dataset from OneDrive')
    # parser.add_argument('-d', '--data_folder', default='../../data') 
    # args = parser.parse_args()
    df = pd.read_csv(data_folder + '/video_links.csv')
    df.dropna(inplace=True)
    data = df.values
    print(df.head(n=50))
    for idx in range(data.shape[0]):
        print(data[idx][0])
        vid = data[idx][0]
        vlink = data[idx][1]
        download_link = create_onedrive_directdownload(vlink)
        print("wget -O Videos/picklist_" + str(vid) + ".mp4 '"+ str(download_link) +"'")
        os.system("wget -O " + data_folder + "/videos/picklist_" + str(vid) + ".mp4 '"+ str(download_link) +"'")