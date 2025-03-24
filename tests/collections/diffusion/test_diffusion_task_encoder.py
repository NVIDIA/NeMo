import pytest
from nemo.collections.diffusion.data.diffusion_taskencoder import cook_raw_images


class TestTaskEncoder:
    @pytest.mark.unit
    def test_cook_raw_images(self):
        sample = {"jpg": "original_image_data", "png": "control_image_data", "txt": "raw_text_data"}

        processed_sample = cook_raw_images(sample)

        assert "images" in processed_sample
        assert "hint" in processed_sample
        assert "txt" in processed_sample

        assert processed_sample["images"] == sample["jpg"]
        assert processed_sample["hint"] == sample["png"]
        assert processed_sample["txt"] == sample["txt"]
