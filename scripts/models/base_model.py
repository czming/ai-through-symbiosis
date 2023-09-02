from abc import ABC, abstractmethod

class Model(ABC):

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self, input_vector):
        pass