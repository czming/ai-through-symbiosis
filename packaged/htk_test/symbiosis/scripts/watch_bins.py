

import os
import matplotlib.pyplot as plt
import numpy as np
import subprocess


# with open("/Users/jonathanwomack/projects/ai-through-symbiosis/github-repo/experiments/raw/picklist_1-new") as infile:
with open("/Users/jonathanwomack/projects/ai-through-symbiosis/github-repo/scripts/test.txt") as infile:
    htk_inputs = [i.split() for i in infile.readlines()]
    for frame_num, htk_line in enumerate(htk_inputs[::5]):
        hsv_bin_sum = [0 for i in range(20)]
        for k in range(0,20):
            # sum up the current values
            hsv_bin_sum[k] += float(htk_line[k])

        print(frame_num)
        fig, axs = plt.subplots(1, 1)
        axs.bar(range(20), np.asarray(hsv_bin_sum))
        axs.set_ylim([0, 1])
        plt.savefig("/Users/jonathanwomack/projects/ai-through-symbiosis/github-repo/scripts/imgs" + "/file%02d.png" % frame_num)
        plt.close()

    os.chdir("/Users/jonathanwomack/projects/ai-through-symbiosis/github-repo/scripts/imgs")
    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        'video_name.mp4'
    ])
    # for file_name in glob.glob("*.png"):
    #     os.remove(file_name)