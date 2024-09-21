import cv2
import glob, os
import numpy as np
from PIL import Image
from pkg_resources import resource_stream
import skimage.color

import matplotlib.pyplot as plt

dir = "./dataset/"
os.chdir(dir)
np.set_printoptions(suppress=True)
videos = ["picklist_17","picklist_18","picklist_19","picklist_20","picklist_21","picklist_22"]
for video in videos:
    files = sorted(glob.glob(video + '/*.jpg'), key=os.path.getmtime)
    with open("../" + video, 'w') as fi:
        for file in files:
            print(file)
            # if "blue1" in file:
            pil_image = Image.open(resource_stream(__name__, dir+file))
            color_image = np.array(Image.open(resource_stream(__name__, dir+file)))
            # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            number_of_pixels = color_image.shape[0] * color_image.shape[1]
            hsv_image = skimage.color.rgb2hsv(color_image)
            dims = hsv_image.shape
            hues = []
            saturations = []
            for i in range(0, dims[0]):
                for j in range(0, dims[1]):
                    # subsample
                    if i % 1 == 0:
                        # BGR
                        hsv_value = np.array([[hsv_image[i, j, 0],
                                               hsv_image[i, j, 1],
                                               hsv_image[i, j, 2]]])
                        # rgb_value = np.array([[color_image[i, j, 0],
                        #                        color_image[i, j, 1],
                        #                        color_image[i, j, 2]]]) / 255.0
                        hues.append(hsv_value[0][0])
                        saturations.append(hsv_value[0][1])
            f, axarr = plt.subplots(2, 2)

            axarr[0,0].imshow(color_image)
            h = sum(hues)/len(hues)
            s = sum(saturations)/len(saturations)
            # print(max(set(hues), key=hues.count)) #mode
            # print(max(set(saturations), key=saturations.count)) #mode
            V = np.array([[h, s]])
            origin = np.array([[0, 0, 0], [0, 0, 0]])  # origin point
            # axarr[1].set_xlim([0, 10])
            # axarr[1].set_ylim([0, 10])
            axarr[0,1].quiver(*origin, V[:, 0], V[:, 1], color=['r'], scale=10)
            circle1 = plt.Circle((0, 0), 1 / 21, fill=False)
            axarr[0,1].add_patch(circle1)
            hist_n,hist_bins, _ = axarr[1,0].hist(hues, bins=10, range=(0,1))
            axarr[1,0].set_xlim([0, 1])
            # axarr[1,0].set_title("Hue")
            sat_n,sat_bins, _ = axarr[1,1].hist(saturations, bins=10, range=(0,1))
            # print(hist_n, hist_bins)
            # print(sat_n, sat_bins)

            histsat = np.concatenate((hist_n, sat_n))
            mystring = str(histsat).replace("[","").replace("]","").replace(".","").replace("\n","")
            newstring = ' '.join(mystring.split())
            print(newstring)
            fi.write(newstring)
            fi.write("\n")

            axarr[1,1].set_xlim([0, 1])
            # axarr[1,1].set_title("Saturation")
            # plt.show()
            # plt.close()



