import xml.etree.ElementTree as ET

tree = ET.parse('../elan_annotated/picklist_17.eaf')

# frames per second of the camera to correspond between the event timings and the frames
FPS = 60

root = tree.getroot()
elan_annotations = []
it = root[1][:]
event_frames = [0]

for index in it:
    event_frames.append(int(int(index.attrib['TIME_VALUE']) * 60/1000))

# getting the ending time based on the htk_inputs
#event_times.append(len(vectors)/60) #60 frames per second
# event_times contains the time of each of the events
print (event_frames)