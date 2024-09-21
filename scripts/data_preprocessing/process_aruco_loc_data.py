# runs forward filling and gets location of marker 211 (index 14 and 15)
# as data for htk_input

# from aruco binning script
# pick bins
pick_bins = [i for i in range(24 * 2)]
place_bins = [i for i in range(24 * 2, 36 * 2)]

# number of frames that the aruco marker is assumed to be detected for
# 0 indicates no persistence (i.e. only store for one frame and that's it
PERSISTANCE = 30

data_folder = "../../htk_inputs"

for i in range(1,71):
    with open(f"{data_folder}/picklist_{i}.txt") as infile:

        # overwrite output file first

        # track the last frame when the aruco marker was detected (delete the marker if more than the desired number of
        # frames has passed)
        last_frame_detected = {}

        with open(f"{data_folder}/picklist_{i}_forward_filled_{PERSISTANCE}_aruco_211.txt", "w") as outfile:
            
            infile = infile.readlines()
            for index, line in enumerate(infile):
                line = line.split()
                aruco_markers = line[:72]

                # taking into account the forward fill
                aruco_pick_place_counter = 0

                for x_index in range(0, 72, 2):
                    x = float(aruco_markers[x_index])
                    if x != 0:
                        # the x value for the aruco marker is non-zero (used 0 as the placeholder for undetected marker)
                        # update the last frame where this aruco marker was detected
                        last_frame_detected[x_index] = index

                # go through last_frame_detected and then count the bins again

                for x_index, last_seen in last_frame_detected.items():
                    # check if the current x_index has been visited in the last PERSISTENCE number of frames
                    if index - PERSISTANCE <= last_seen:
                        if x_index in pick_bins:
                            # +1 for pick bin
                            aruco_pick_place_counter += 1
                        if x_index in place_bins:
                            # -1 for place bin
                            aruco_pick_place_counter -= 1

                # get the location of marker 211
                new_line = [str(aruco_pick_place_counter)] + line[14:16]


                outfile.write(" ".join(new_line))
                outfile.write("\n")
                
        print (f"picklist_{i}.txt")

