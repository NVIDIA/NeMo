from comet import download_model, load_from_checkpoint


class CometScore:
    def __init__(self, model_name: str = "Unbabel/wmt22-comet-da"):
        self.default_model = model_name
        model_path = download_model(self.default_model)
        self.model = load_from_checkpoint(model_path)


    def __call__(self, hypotheses, references, sources):

        data = {
            "src": sources,
            "mt": hypotheses,
            "ref": references,
        }

        data = [dict(zip(data, t)) for t in zip(*data.values())]

        model_output = self.model.predict(data)

        return model_output.scores