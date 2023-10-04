"""
Provides functions for drawing mock place bin setup
"""
import utils 
import cv2
import numpy as np
import math 
import os


'''create a black background'''
def create_background(width = 720, height = 640):
    return np.zeros([height, width, 3], dtype=np.uint8)

'''
draw bins on an image. written for 3 bins for now. consider re-writing for a generic n bins
'''
def draw_bins(img):
    (h, w) = img.shape[:2]

    bins = [[int(w / 30), int(h / 10), int(3 * w / 10), int(9 * h / 10)], [int(11 * w / 30), int(h / 10), int(19 * w / 30), int(9 * h / 10)], [int(7 * w / 10), int(h / 10), int(29 * w / 30), int(9 * h / 10)]]

    for i in range(len(bins)):
       bin = bins[i]
       img[bin[1] : bin[3], bin[0] : bin[2]].fill(83)
       img = cv2.putText(img, "Bin " + str(i + 1), (bin[0], int(0.98 * bin[3])), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    return img, bins

'''
draw images in a bin in the image. only draws 3 for now.
'''
def draw_images_in_bin(img, images, bin):
    PADDING = 10
    bin_h, bin_w = bin[3] - bin[1], bin[2] - bin[0]
    num_images = len(images)

    if num_images > 0:
        item_h, item_w = int(bin_h * 0.8 / num_images), int(bin_w * 0.8)


        

        for i in range(num_images):
            item_img = images[i]
            item_img_h, item_img_w = item_img.shape[:2]
            h_ratio, w_ratio = item_img_h / item_h, item_img_w / item_w 
            r_h, r_w = int(item_img_h / max(h_ratio, w_ratio)), int(item_img_w / max(h_ratio, w_ratio)) 
            
            resized_item_img = cv2.resize(item_img, (r_w, r_h), cv2.INTER_AREA)
            
            img[bin[1] + PADDING * (i + 1) + i * r_h : bin[1] + PADDING * (i + 1) + (i + 1) * r_h, bin[0] + PADDING : bin[0] + PADDING + r_w] = resized_item_img   

    return img

'''
color_image_mapping: dict of char representing color to 3d array (single image)
TODO: turn this into something that works with both 3d or 4d arrays
bin_color_mapping: dict of bin number to list of items that go in there

Input:
- color_image_mapping: dict from char of color to an image representing that color
- bin_image_mapping: dict from bin number to list of colors in that bin
- width: the width to use for image
- height: the height to use for image
- show_img: whether or not to display the image before returning it

Returns:
- an image of 3 bins with items in those bins
'''
def create_drawing(color_image_mapping, bin_color_mapping, width = 720, height = 640, show_img = False):
    img = create_background(width, height)
    img, bins = draw_bins(img)
    
    for bin, colors in bin_color_mapping.items():
        #bin in [1, 2, 3]
        bin_bounds = bins[bin - 1] #[min x, min y, max x, max y]
        color_imgs = [color_image_mapping[c] for c in colors]
        img = draw_images_in_bin(img, color_imgs, bin_bounds)

    if show_img:
        utils.show_frame(img, [width, height])

    return img 

if __name__ == '__main__':
    r, g, b = [cv2.imread(os.path.join(os.getcwd(), "pick_items", "new_photos", "normal_lens", x + ".JPG")) for x in ["red", "lightgreen", "darkblue"]]
    
    r = utils.extract_hand_region_from_frame(r)
    g = utils.extract_hand_region_from_frame(g)
    b = utils.extract_hand_region_from_frame(b)

    # utils.show_frame(r)
    
    color_image_mapping = {'r' : r, 'g': g, 'b': b}
    # fake_bin_color_mapping = {1: ['r', 'r', 'g'], 2: ['g', 'b'], 3:['r', 'b', 'g', 'b', 'r']}

    picklist_41_mapping = utils.label_to_dict(os.path.join(os.getcwd(), "labels", "picklist_41_raw.txt"))
    img = create_drawing(color_image_mapping, picklist_41_mapping, show_img = True)
    cv2.imwrite("picklist_41_gui.png", img)

    picklist_41_mapping = utils.label_to_dict(os.path.join(os.getcwd(), "labels", "picklist_41_perturbed.txt"))
    img = create_drawing(color_image_mapping, picklist_41_mapping, show_img = True)
    cv2.imwrite("picklist_41_perturbed_gui.png", img)

    picklist_42_mapping = utils.label_to_dict(os.path.join(os.getcwd(), "labels", "picklist_42_raw.txt"))
    img = create_drawing(color_image_mapping, picklist_42_mapping, show_img = True)
    cv2.imwrite("picklist_42_gui.png", img)

    picklist_42_mapping = utils.label_to_dict(os.path.join(os.getcwd(), "labels", "picklist_42_perturbed.txt"))
    img = create_drawing(color_image_mapping, picklist_42_mapping, show_img = True)
    cv2.imwrite("picklist_42_perturbed_gui.png", img)
