from abc import ABC, abstractmethod
from typing import List

class PredictorInterface(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        """
        Constructor method for initializing the predictor.
        """
        pass

    @abstractmethod
    def predict(self, input_data):
        """
        Predict method for making predictions on a single input.

        Parameters:
            - input_data: Input data for prediction.

        Returns:
            - Prediction result.
        """
        pass
