import glob
import numpy as np
import cv2
import os

intrinsic = np.load("intrinsic_gopro.npy")
distortion = np.load("distortion_gopro.npy")

image_files = glob.glob("G:/My Drive/Georgia Tech/AI Through Symbiosis/GoPro/images/distorted/*.jpg")

output_folder = "G:/My Drive/Georgia Tech/AI Through Symbiosis/GoPro/images/undistorted"

if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

for image_file in image_files:
    image = cv2.imread(image_file)
    image = cv2.undistort(image, intrinsic, distortion, None)

    print (output_folder + "/" + os.path.basename(image_file))

    cv2.imwrite(output_folder + "/" + os.path.basename(image_file), image)