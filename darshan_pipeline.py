from scripts.data_preprocessing.preprocess_pipeline import preprocess
from symbiosis.run_n_fold_pipeline import
import cv2

video_file = "video.mp4"
label_file = "label.txt"
feature_vector_file = "feature_vector.txt"
preprocessed_vector_file = "data/preprocessed_vector.txt"
cap = cv2.VideoCapture(video_file)
feature_vector = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # call feature extraction service on frame
    # feature_vector.append(features for frame)

# write feature_vector to feature_vector_file

# call preprocessing on feature_vector
preprocessed_features = preprocess(feature_vector, preprocessed_vector_file)
# call HTK on preprocessed_features
htk_features = run_nfold()
# call image model on htk features and feature vector
image_model = foo(htk_features, feature_vector)
cap.release()
cv2.destroyAllWindows()
