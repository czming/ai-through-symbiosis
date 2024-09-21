import pickle

with open("saved_models/carry_histogram_hsv_model.pkl", "rb") as f:
    carry_histogram_hsv_model = pickle.load(f)

print (carry_histogram_hsv_model.iterative_clustering_model.class_hsv_bins_mean['s'])

carry_histogram_hsv_model.fit_iterative([[0. for i in range(12)]], ['s'], 0.9)

# check to see if the value got reduced
print (carry_histogram_hsv_model.iterative_clustering_model.class_hsv_bins_mean['s'])

print (carry_histogram_hsv_model.iterative_clustering_model.class_hsv_bins_mean['t'])

carry_histogram_hsv_model.fit_iterative([[0. for i in range(12)] for j in range(2)], ['s', 't'], 0.5)

# check to see if the value got reduced by half for both
print (carry_histogram_hsv_model.iterative_clustering_model.class_hsv_bins_mean['s'])
print (carry_histogram_hsv_model.iterative_clustering_model.class_hsv_bins_mean['t'])
