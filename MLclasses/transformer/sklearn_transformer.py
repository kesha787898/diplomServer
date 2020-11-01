from MLclasses.transformer.transformer import Transformer
import pickle


class SklearnTransformer(Transformer):

    def __init__(self, path):
        super().__init__()
        with open(path, 'wb') as f:
            self.transformer = pickle.load(f)
        self.num_in = self.transformer.n_features_
        self.num_out = self.transformer.n_components_

    def transform(self, x):
        return self.transformer.transform(x)

    def __call__(self, x):
        return self.transform(x)

    def __str__(self):
        return self.transformer.__str__()