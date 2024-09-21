import pickle
import matplotlib.pyplot as plt
import numpy as np
import pickle

def get_hsv_hist_fig(input_dict):
    letter_to_name = {
        'r': 'red',
        'g': 'green',
        'b': 'blue',
        'p': 'darkblue',
        'q': 'darkgreen',
        'o': 'orange',
        's': 'alligatorclip',
        'a': 'yellow',
        't': 'clear',
        'u': 'candle'
    }

    colors = {
        'g': 'green',
        'a': 'yellow',
        'u': 'tan',
        'b': 'blue',
        'q': 'darkgreen',
        'r': 'red',
        'o': 'orange',
        'p': 'darkblue',
        't': 'grey',
        's': 'black'
    }
    
    plt_display_index = 0
    fig, axs = plt.subplots(2, len(input_dict) // 2)

    for object, predicted_objects in input_dict.items():

        print (predicted_objects)
        if plt_display_index < len(input_dict) // 2:
            axs[0, plt_display_index].bar(range(len(predicted_objects)), predicted_objects, color=colors[object])
            axs[0, plt_display_index].set_title(letter_to_name[object])
            axs[0, plt_display_index].set_ylim([-0.15, 0.15])
            axs[0, plt_display_index].set_yticks([])
            axs[0, plt_display_index].set_xticks([])

        else:
            axs[1, plt_display_index - len(input_dict) // 2].bar(range(len(predicted_objects)), predicted_objects, color=colors[object])
            axs[1, plt_display_index - len(input_dict) // 2].set_title(letter_to_name[object])
            axs[1, plt_display_index - len(input_dict) // 2].set_ylim([-0.15, 0.15])
            axs[1, plt_display_index - len(input_dict) // 2].set_yticks([])
            axs[1, plt_display_index - len(input_dict) // 2].set_xticks([])
        plt_display_index += 1

        axs[0, 0].set_yticks([-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15])
        axs[1, 0].set_yticks([-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15])

    return fig

if __name__ == "__main__":
    # Comment this out later, function call for testing purposes
    with open('object_type_hsv_bins_copy.pkl', 'rb') as f:
        input = pickle.load(f)
    fig = get_hsv_hist_fig(input)

    fig.show()
    plt.savefig("object_type_hsv_bins_visualization.png")
    plt.show()