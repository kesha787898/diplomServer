from MLclasses.predictors.predictor import Predictor
import torch


class TorchPredictor(Predictor):
    threshold = None

    def __init__(self, path, threshold=0.5):
        self.predictor = torch.load(path)
        self.threshold = threshold

    def predict(self, x):
        return self.predict_probability(x) > self.threshold

    def predict_probability(self, x):
        return self.predictor(x)

    def __call__(self, x):
        return self.predict_probability(x)

    def __str__(self):
        return self.predictor.__str__()
