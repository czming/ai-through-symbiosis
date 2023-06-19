#!/bin/bash
mkdir ../../data/videos
python utils/download_videos.py 
python utils/extract_frames.py