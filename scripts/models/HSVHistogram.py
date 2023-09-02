from .base_model import Model

class HSVHistogramModel(Model):
    # takes in some embedding vector and

    def __init__(self):
        pass

    def preprocess(self, action_boundaries):
        # get the avg hsv vectors from the video and subtract away hand, then ready to train/fit on the vector obtained
        pass

    def fit(self):
        pass

    def predict(self, hsv_vector, action_boundaries):
        # hsv_vector: int[280], 180 bins for hue, 100 bins for saturation
        # action_boundaries: dict[str, int] --> contains the timestamps of the different action start and end times
        # e.g. {'a': [start1, end1]}

        return predicted_class