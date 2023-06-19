#!/bin/bash
mkdir ../../data/videos
python dl_utils/download_videos.py 
python dl_utils/extract_frames.py
python prep_triplet_data.py
python train_triplet.py