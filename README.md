# AI Through Symbiosis

----------------------

Code repo for AI Through Symbiosis project, accepted for publication by Ambient AI: Multimodal Wearable Sensor Understanding workshop at ICASSP 2023.

Keep track of this repo as we plan to release our picklist video dataset soon!

## <ins>Quickstart</ins>

For a quickstart, you can download the intermediate outputs generated from our dataset and change the config paths to match

## <ins>Playing around</ins>

### Extracting features from video

Running extract_features.py script to extract features from the videos
``` 
# run from ai-through-symbiosis root directory 
python3 extract_features.py -v <path to video file>
```

To visualize features that are being detected (e.g. ArUco markers, hand locations), set `DISPLAY_TRUE` in 
extract_features.py script to `True`

Next, preprocess the data by running the ```scripts/data_preprocessing/forward_fill.py``` and ```scripts/data_preprocessing/gaussian_convolution.py```
scripts. Create a config file by copying ```scripts/configs/base.yaml``` and filling in the appropriate paths to folders in your local environment.

To set the configs for the above and subsequent python script calls, use the ```-c, --config_file``` argument in the command line call.

### Training HMMs

(In development)

### Extracting pick labels for pick sequences

Set the ```PICKLISTS``` variable in ```iterative_improvement_clustering.py``` to specify the range of picklists numbers to iterate over (script will skip over picklists that are not in the folder)

```
# run from ai-through-symbiosis root directory
python3 scripts/iterative_improvement_clustering.py
```

which should give you the predicted labels for the different picklists using the iterative clustering algorithm constrained by the object count set for each picklist. It will also output the hsv bins to ```object_type_hsv_bins.pkl``` representing the different object types which serves as our object representation, along with a confusion matrix similar to the results below

### Testing HSV object representation

After obtaining ```object_type_hsv_bins.pkl``` from the previous step, run the object classification script on the relevant picklists by setting the ```PICKLISTS``` variable in ```object_classification.py```

```
# run from ai-through-symbiosis root directory
python3 scripts/object_classification.py
```


NOTE: If you are using a 30fps video instead of 60fps video, change the 



## <ins> Results </ins>

Train set results from clustering (i.e. accuracy of matching labels to pick frame sequences in the train set), which is used to
label the weakly supervised action sets and obtain object representations
![clustering train set results](images/clustering_train_set.png)

Test set results on three object test set (restricting output predictions to three objects):
![three object test set results](images/3_objects_type_constrained_picklist.png)

Test set results on ten object test set (constrained to set of objects that appeared in the picklist):
![ten object test set constrained](images/10_objects_constrained_avg_boundaries.png)

Test set results on ten object test set (unconstrained):
![ten object test set](images/10_objects_no_constraints_avg_boundaries.png)


[//]: # (## <ins> Contribute! </ins>)

[//]: # ()
[//]: # (If you find our work helpful, consider contributing )

