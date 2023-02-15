"""

processes raw label inputs into pick specific, general action grammars


"""
import glob

import json

files = glob.glob("../../../Labels/*60_raw.txt")

for file in files:
    with open(file, "r") as infile:
        phrase = infile.read().strip()
        print(file)
        
        pick_num = file[file.index("_")+1: file.rindex("_")]
        with open("grammar_letter_isolated_ai_general-" + pick_num, "w") as outfile:
            outfile.write("$char = a e i m;\n")
            pick_specific_grammar = '(sil '
            pick_specific_grammar += '$char ' * int(len(phrase)/2)
            pick_specific_grammar += 'sil)'
            outfile.write(pick_specific_grammar)


# $char = a e i m;
# (sil $char $char $char $char $char $char $char sil)