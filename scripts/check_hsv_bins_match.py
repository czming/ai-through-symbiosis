# compare whether the hsv bins from Jon's show_hands script matches main.py generated after converting BGR to RGB

from utils import vectors_rms_error

with open("C:/Users/chngz/Downloads/jon-aits/ai-through-symbiosis/experiments/show-hand-color/data/picklist_3.txt") as infile:
    hsv_bins_1 = [[float(j) for j in i.split()] for i in infile.readlines()]

with open("../picklist_3.txt") as infile:
    hsv_bins_2 = [[float(j) for j in i.split()[72:92]] for i in infile.readlines()]

print (vectors_rms_error(hsv_bins_1[:len(hsv_bins_2)], hsv_bins_2[:-1]))