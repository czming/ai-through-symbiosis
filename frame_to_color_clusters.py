from curses import color_content
from gzip import READ
from turtle import color
import cv2
import glob, os
import numpy as np
from PIL import Image
from pkg_resources import resource_stream
import skimage.color
from pysnic.algorithms.snic import snic
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans
import matplotlib.image as img
import matplotlib.pyplot as plt
from scipy.cluster.vq import whiten
from scipy.cluster.vq import kmeans
import pandas as pd
from mpl_toolkits import mplot3d
  


def write_hsv_bins_to_file():
    dir = "../2022Spring/dataset/"
    os.chdir(dir)
    np.set_printoptions(suppress=True)
    videos = ["picklist_3","picklist_4","picklist_6","picklist_8","picklist_9"]
    videos = ["picklist_4","picklist_6","picklist_8","picklist_9"]
    videos = ["picklist_17", "picklist_18", "picklist_19", "picklist_20", "picklist_21", "picklist_22"]
    videos = ["picklist_19"]
    for video in videos:
        files = sorted(glob.glob(video + '/*.jpg'), key=os.path.getmtime)
        with open(video + "-sat.txt", 'a') as fi:
        # with open(video + ".txt", 'a') as fi:
            for file in files:
                print(file[0:-4].replace(video,"").replace("/",""))
                pil_image = Image.open(resource_stream(__name__, dir+file))
                color_image = np.array(Image.open(resource_stream(__name__, dir+file)))
            
                # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)




                batman_image = color_image
                
                r = []
                g = []
                b = []
                for row in batman_image:
                    for temp_r, temp_g, temp_b in row:
                        r.append(temp_r)
                        g.append(temp_g)
                        b.append(temp_b)
                
                batman_df = pd.DataFrame({'red' : r,
                                        'green' : g,
                                        'blue' : b})
                
                batman_df['scaled_color_red'] = whiten(batman_df['red'])
                batman_df['scaled_color_blue'] = whiten(batman_df['blue'])
                batman_df['scaled_color_green'] = whiten(batman_df['green'])
                
                cluster_centers, _ = kmeans(batman_df[['scaled_color_red',
                                                    'scaled_color_blue',
                                                    'scaled_color_green']], 5)
                
                dominant_colors = []
                
                red_std, green_std, blue_std = batman_df[['red',
                                                        'green',
                                                        'blue']].std()
                
                for cluster_center in cluster_centers:
                    red_scaled, green_scaled, blue_scaled = cluster_center
                    dominant_colors.append((
                        red_scaled * red_std / 255,
                        green_scaled * green_std / 255,
                        blue_scaled * blue_std / 255
                    ))

                print("/Users/jonwomack/Documents/projects/ai-through-symbiosis/2022Spring/dataset" + file)
                im = Image.open("/Users/jonwomack/Documents/projects/ai-through-symbiosis/2022Spring/dataset/" + file)
                px = im.load()
                            # f, axarr = plt.subplots(2,1) 
                x = []
                y = []
                z = []
                c = []
                subsample = 4
                for row in range(0,im.height, subsample):
                    for col in range(0, im.width, subsample):
                        pix = px[col,row]
                        newCol = (pix[0] / 255, pix[1] / 255, pix[2] / 255)
                        # if(not newCol in c):
                        x.append(pix[0])
                        y.append(pix[1])
                        z.append(pix[2])
                        c.append(newCol)
                im = im.convert('HSV')
                px = im.load()
                h = []
                s = []
                v = []
                for row in range(0,im.height, subsample):
                    for col in range(0, im.width, subsample):
                        pix = px[col,row]
                        hue, saturation, value = (pix[0] / 255, pix[1] / 255, pix[2] / 255)
                        h.append(hue)
                        s.append(saturation)
                        v.append(value)
                # Set up a figure twice as tall as it is wide
                fig = plt.figure(figsize=plt.figaspect(2.))
                fig.suptitle('A tale of 2 subplots')

                # First subplot
                ax = fig.add_subplot(7, 1, 1)
                # fgbg = cv2.createBackgroundSubtractorMOG2()
                fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=False)
                fgmask = fgbg.apply(color_image)
                ax.imshow(color_image)

                # Second subplot
                ax = fig.add_subplot(7, 1, 2, projection='3d')
                ax.scatter(x, y, z, c=c)
                ax.set_xlim(0, 255)
                ax.set_ylim(0, 255)
                ax.set_zlim(0, 255)

                ax = fig.add_subplot(7, 1, 3)
                ax.scatter(h, s, c=c)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

                ax = fig.add_subplot(7, 1, 4)
                hist_n,hist_bins, _ = ax.hist(h, bins=10, range=(0,1))
                print(hist_n)
                ax.set_xlim([0, 1])
                ax.set_title("Hue")

                ax = fig.add_subplot(7, 1, 5)
                sat_n,sat_bins, _ = ax.hist(s, bins=10, range=(0,1))
                ax.set_xlim([0, 1])
                ax.set_ylim(0, 50)

                ax.set_title("Saturation")

                ax = fig.add_subplot(7, 1, 6)
                val_n,val_bins, _ = ax.hist(v, bins=10, range=(0,1))
                ax.set_xlim([0, 1])
                ax.set_title("Value")

                ax = fig.add_subplot(7, 1, 7)
                ax.scatter(s, v, c=c)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)


                # fi.write(str(hist_n) + "\n")
                # fi.write(str(sat_n) + "\n")  
                plt.show(block=False)
                plt.pause(.1)
                plt.close()         


def read_hue_bins():
    dir = "../2022Spring/dataset/"
    dir = "./Sequential/data-full"
    files = glob.glob(dir + '*.txt-new')
    files = [dir + "picklist_19.txt"]
    for file in files:
        print(file)
        fig = plt.figure(figsize=plt.figaspect(2.))
        with open(file, 'r') as fi:
            lines = fi.readlines()
            #initialize empty numpy array
            hist_n = []
            # convert line into 2d numpy array split by spaces
            for line in lines:
                # read string array into numpy array
                line = line.replace(".","").replace("[","").replace("]","")
                hist_n.append(np.array(line.split()).astype(np.float))
            np_hist = np.asarray(hist_n)
            print(np_hist.shape)
            # print first column of numpy array
            print(np_hist[:,0].shape)
            for i in range(np_hist.shape[1]):
                col_values = np_hist[:,i]
                # matplotlib line graph of col_values
                ax = fig.add_subplot(10, 1, i+1)
                ax.plot(col_values)
                ax.set_ylim(0,250)
            plt.draw()
def read_sat_bins():
    dir = "../2022Spring/dataset/"
    files = glob.glob(dir + '*-sat.txt')
    files = [dir + "picklist_19.txt"]
    for file in files:
        print(file)
        fig = plt.figure(figsize=plt.figaspect(2.))
        with open(file, 'r') as fi:
            lines = fi.readlines()
            #initialize empty numpy array
            hist_n = []
            # convert line into 2d numpy array split by spaces
            for line in lines:
                # read string array into numpy array
                line = line.replace(".","").replace("[","").replace("]","")
                hist_n.append(np.array(line.split()).astype(np.float))
            np_hist = np.asarray(hist_n)
            print(np_hist.shape)
            # print first column of numpy array
            print(np_hist[:,0].shape)
            for i in range(np_hist.shape[1]):
                col_values = np_hist[:,i]
                # matplotlib line graph of col_values
                ax = fig.add_subplot(10, 1, i+1)
                ax.plot(col_values)
                # ax.set_ylim(0,250)
            plt.draw()

read_hue_bins()
read_sat_bins()
# write_hsv_bins_to_file()


