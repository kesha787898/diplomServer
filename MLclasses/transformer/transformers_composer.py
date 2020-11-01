from MLclasses.transformer.transformer import Transformer


class TransformerComposer(Transformer):
    transformers = []

    def __init__(self, transformers):
        super().__init__()
        for i, _ in range(len(transformers) - 1):
            if transformers[i].num_out != transformers[i + 1].num_in:
                raise Exception(f"Num out!= num_in. {transformers[i].num_out} != {transformers[i + 1].num_in}")
        self.transformers = transformers
        self.num_in = transformers[0].num_in
        self.num_out = transformers[-1].num_out

    def transform(self, x):
        for transformer in self.transformers:
            x = transformer(x)
        return x

    def __str__(self):
        rez = ""
        for transformer in self.transformers:
            rez += transformer.__str__() + "\n"
        return rez
