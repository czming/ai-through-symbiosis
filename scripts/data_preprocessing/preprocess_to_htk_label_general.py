"""

processes raw label inputs into htk labels based on the color of pick and bin place


"""
import glob

import json

files = glob.glob("../../Labels/*_raw.txt")

for file in files:
    with open(file, "r") as infile:
        phrase = infile.read().strip()
    print(file.replace("_raw.txt", ".lab"))
    print(file.replace("_raw.txt", ".lab"))
    with open(file.replace("_raw.txt", ".lab"), "w") as outfile:
        #prepend sil for HTK formatting
        outfile.write("sil\n")
        for token_index in range(0, len(phrase), 2):
            token = phrase[token_index: token_index + 2]
            print(token)
            outfile.write("a\ne\n")

            if token[1] == "1":
                #place in object bin 1
                outfile.write("i\n")
            elif token[1] == "2":
                #place in object bin 2
                outfile.write("i\n")
            elif token[1] == "3":
                #place in object bin 3
                outfile.write("i\n")
            else:
                raise Exception(f"No such integer found for token[1]: {token}")

            # carry empty
            outfile.write("m\n")
        #append sil to end the file
        outfile.write("sil")
