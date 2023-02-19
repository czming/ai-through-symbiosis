"""

processes raw label inputs into htk labels based on the color of pick and bin place


"""
import glob
import json
import sys

# append the path of the parent directory
sys.path.append("..")

from utils import *

# use path from the current working directory (not the one from the utils module)
configs = load_yaml_config("../configs/zm.yaml")

label_folder = configs["file_paths"]["label_file_path"]

files = glob.glob(f"{label_folder}/*_raw.txt")

for file in files:
    with open(file, "r") as infile:
        phrase = infile.read().strip()
    with open(file.replace("_raw.txt", ".lab"), "w") as outfile:
        #prepend sil for HTK formatting
        outfile.write("sil\n")
        for token_index in range(0, len(phrase), 2):
            token = phrase[token_index: token_index + 2]
            if token[0] == "r":
                #red object pick and carry
                outfile.write("b\nf\n")
            elif token[0] == "b":
                #blue object pick and carry
                outfile.write("c\ng\n")
            elif token[0] == "g":
                #green object pick and carry
                outfile.write("d\nh\n")
            else:
                raise Exception(f"No such letter found for token[0]: {token}")

            if token[1] == "1":
                #place in object bin 1
                outfile.write("j\n")
            elif token[1] == "2":
                #place in object bin 2
                outfile.write("k\n")
            elif token[1] == "3":
                #place in object bin 3
                outfile.write("l\n")
            else:
                raise Exception(f"No such integer found for token[1]: {token}")

            # carry empty
            outfile.write("m\n")
        #append sil to end the file
        outfile.write("sil")

    print (file)
