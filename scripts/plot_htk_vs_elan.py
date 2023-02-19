import argparse
import glob as glob
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np


from collections import defaultdict


def produce_plots(result_dir, plots_dir):
    viz_offset = 0.01
    print(result_dir)
    picklists = ['1','2','3','17','18','19','20','21','22','26']
    picklists = ['20', '21', '22', '26']
    # picklists = ['1', '2', '3', '17', '18', '19']
    picklists = list(range(0,135))

    for picklist_num in picklists:
        # Getting HTK boundaries (predicted)
        results_filename = "results-" + str(picklist_num)
        htk_label_path = os.path.join(result_dir, results_filename)
        if not os.path.exists(htk_label_path):
            continue
        print("Picklist " + str(picklist_num))
        htk_boundaries = defaultdict(list)
        with open(htk_label_path, 'r') as f:
            # lines = f.readlines()[3:-2] # Output files may differ in heading
            lines = f.readlines()[1:-1] # Output files may differ in heading
            for line in lines[1:]:
                if line[0] == ".": #hacky solution for erroneous labels (predictions written twice)
                    break
                boundaries = line.split()[0:2]
                letter = line.split()[2]
                letter_start = [int(boundary)/120000 for boundary in boundaries][0] #2000 (HTK multiplier) * 60FPS
                letter_end = [int(boundary)/120000 for boundary in boundaries][1] - viz_offset
                htk_boundaries[letter].append(letter_start)
                htk_boundaries[letter].append(letter_end)
        # Getting Elan boundaries (ground truth)
        tree = ET.parse('./elan_annotated/picklist_'+str(picklist_num)+'.eaf')
        root = tree.getroot()
        elan_boundaries = defaultdict(list)
        elan_annotations = []
        # parsing annotation for letter / token
        for annotation in root[2][:]:
            all_descendants = list(annotation.iter())
            for desc in all_descendants:
                if desc.tag == "ANNOTATION_VALUE":
                    if 'empty' not in desc.text:
                        elan_annotations.append(desc.text[0:desc.text.index("_")])
                    else:
                        elan_annotations.append(desc.text)
        prev_time_value = int(root[1][1].attrib['TIME_VALUE'])/1000
        it = root[1][:]
        for index in range(0, len(it), 2):
            letter = elan_annotations[int(index/2)]
            letter_start = int(it[index].attrib['TIME_VALUE'])/1000
            letter_end = int(it[index + 1].attrib['TIME_VALUE'])/1000 - viz_offset
            elan_boundaries[letter].append(letter_start)
            elan_boundaries[letter].append(letter_end)


        legends = {}
        colors = {'sil':'r',
                'a':'c',
                'e':'y',
                'i':'m',
                'm':'r'}
        key_swap = {'pick':'a',
                    'carry':'e',
                    'place':'i',
                    'carry_empty':'m'} # elan to htk

        swap_key = {'a':'pick',
                    'e':'carry',
                    'i':'place',
                    'm':'carry_empty',
                    'sil':'sil'}
        color_to_action = {
            'r':'carry_empty',
            'c':'pick',
            'y':'carry',
            'm':'place'
        }
        plt.figure(figsize=(24, 2), dpi=80)
        dot_size = 20*4
        max_boundary = 0
        previous_value = 0
        all_value = []
        value_color = []
        for key, value in elan_boundaries.items():
            if max(value) > max_boundary:
                max_boundary = int(max(value))
            y_key = [1.5] * len(value)
            new_key = key_swap[key]
            for i in range(0,len(value),2):
                item1 = value[i] #start
                item2 = value[i+1] #end
                value_color.append((item1, colors[new_key]))
                value_color.append((item2, colors[new_key]))

        value_color = sorted(value_color, key=lambda x: x[0])
        all_value = [i[0] for i in value_color]
        all_y_key = [1.5] * len(all_value)
        all_colors = [i[1] for i in value_color]
        zipped =  zip(all_value, all_y_key, all_colors)
        items = []
        for x, y, colory in zipped:
            items.append((x,y,colory))
            action = color_to_action[colory]

            legends[action] = plt.scatter(x, y, color=colory, marker="|", s=dot_size)

        # Plot Lines
        for current, previous in zip(items, [None, *items]):
            try:
                x = [previous[0], current[0]]
                y = [previous[1], current[1]]
                plt.plot(x, y, color=current[2])
            except:
                continue

        value_color = []
        all_value = [] 
        for key, value in htk_boundaries.items():
            if max(value) > max_boundary:
                max_boundary = int(max(value))
            y_key = [2] * len(value)
            new_key = key
            for i in range(0,len(value),2):
                item1 = value[i] #start
                item2 = value[i+1] #end
                value_color.append((item1, colors[new_key]))
                value_color.append((item2, colors[new_key]))

        value_color = sorted(value_color, key=lambda x: x[0])
        all_value = [i[0] for i in value_color]
        all_y_key = [2] * len(all_value)
        all_colors = [i[1] for i in value_color]
        zipped =  zip(all_value, all_y_key, all_colors)
        items = []
        for x, y, colory in zipped:
            items.append((x,y,colory))
            action = color_to_action[colory]
            legends[action] = plt.scatter(x, y, color=colory, marker="|", s=dot_size)

        # Plot Lines
        for current, previous in zip(items, [None, *items]):
            try:
                x = [previous[0], current[0]]
                y = [previous[1], current[1]]
                plt.plot(x, y, color=current[2])
            except:
                continue

        plt.legend(legends.values(),
                legends.keys(),
                scatterpoints=1,
                loc='best',
                ncol=1,
                fontsize=8)
        plt.title("HTK (top) vs. ELAN (bottom) - " + str(picklist_num))
        plt.xlabel("Time (seconds)")

        result_dir_no_forward_slash = result_dir.replace("/","-")
        # plt.savefig(os.path.join(plots_dir, result_dir_no_forward_slash + "-" + str(picklist_num)+'.png') )
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        required=True,
        help="Relative or absolute path to a directory containing a single fold's results",
    )
    parser.add_argument(
        "--plots_dir",
        required=True,
        help="Relative or absolute path to a directory where plots are stored",
    )
    args = parser.parse_args()
    produce_plots(args.results_dir, args.plots_dir)