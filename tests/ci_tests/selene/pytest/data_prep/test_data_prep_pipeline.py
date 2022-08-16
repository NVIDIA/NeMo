import os

BASE_RESULTS_DIR = os.environ.get("RESULTS_DIR")

class TestDataPrepPipeline:
    job_name = BASE_RESULTS_DIR.rsplit("/",1)[1]
    model = job_name.split("_")[2] #data_prep_gpt3_pile
    if model == "t5":
        base_file="data/my-t5_00_text_document"
    elif model == "gpt3":
        base_file="data/my-gpt3_00_text_document"
    else:
        base_file="data/mc4/preprocessed/mt_000-001_text_document"

    def test_data_prep(self):
        idx_path = os.path.join(BASE_RESULTS_DIR, self.base_file + ".idx")
        bin_path = os.path.join(BASE_RESULTS_DIR, self.base_file + ".bin")
        assert os.path.exists(idx_path), f"File not found: {idx_path}"
        assert os.path.exists(bin_path), f"File not found: {bin_path}"
