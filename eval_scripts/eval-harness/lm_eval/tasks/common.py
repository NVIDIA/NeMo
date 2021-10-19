import datasets
from ..base import Task


class HFTask(Task):
    DATASET_PATH = None
    DATASET_NAME = None

    def __init__(self):
        self.data = None
        super().__init__()

    def download(self):
        self.data = datasets.load_dataset(path=self.DATASET_PATH, name=self.DATASET_NAME)

    def has_training_docs(self):
        """Whether the task has a training set"""
        return True if "train" in self.data.keys() else False

    def has_validation_docs(self):
        """Whether the task has a validation set"""
        return True if "validation" in self.data.keys() else False

    def has_test_docs(self):
        """Whether the task has a test set"""
        return True if "test" in self.data.keys() else False

    def _convert_standard(self, doc):
        return doc

    def training_docs(self):
        # Cache training for faster few-shot.
        # If data is too large to fit in memory, override this method.
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(map(self._convert_standard, self.data["train"]))
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return map(self._convert_standard, self.data["validation"])

    def test_docs(self):
        if self.has_test_docs():
            return map(self._convert_standard, self.data["test"])


def yesno(x):
    if x:
        return 'yes'
    else:
        return 'no'
