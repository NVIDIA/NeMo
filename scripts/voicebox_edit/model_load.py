import torch
from nemo.collections.tts.models.voicebox import VoiceboxModel, fix_alignment

class MainExc:
    def __init__(self, vb_ckpt_path=None, dp_ckpt_path=None, gen_data_dir="data/gen_dataset", sample_std=0.9):
        self.vb_ckpt_path = vb_ckpt_path
        self.dp_ckpt_path = dp_ckpt_path

        self.gen_data_dir = gen_data_dir
        self.sample_std = sample_std

    def load_model(self,):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model = VoiceboxModel.load_from_checkpoint(self.vb_ckpt_path, map_location=device)
        model = VoiceboxModel.load_from_checkpoint(self.vb_ckpt_path, map_location=device, strict=False)

        # dp_model = VoiceboxModel.load_from_checkpoint(self.dp_ckpt_path, map_location=device)
        dp_model = VoiceboxModel.load_from_checkpoint(self.dp_ckpt_path, map_location=device, strict=False)

        del model.duration_predictor, model.cfm_wrapper.duration_predictor
        model.duration_predictor = dp_model.duration_predictor
        model.cfm_wrapper.duration_predictor = dp_model.duration_predictor
        del dp_model

        torch.cuda.empty_cache()

        model.cap_vocode = True
        return model

    @property
    def model(self):
        if not hasattr(self, "_model"):
            self._model = self.load_model()
        return self._model

    def prepare_val_dl(self, ds_name="libriheavy", corpus_dir="/datasets/LibriLight/", manifest_filepath="data/parsed/LibriHeavy/libriheavy_cuts_dev.jsonl.gz",
                       old_prefix="download/librilight", min_duration=-1, max_duration=float("inf"), load_audio=True, filter_ids=None, shuffle=False, batch_duration=100):
        # load from val set
        self.model.cfg.ds_name = ds_name
        self.model.cfg.corpus_dir = corpus_dir
        self.model.cfg.validation_ds.manifest_filepath = manifest_filepath
        self.model.cfg.validation_ds.lhotse.cuts_path = self.model.cfg.validation_ds.manifest_filepath
        with open_dict(self.model.cfg.validation_ds):
            self.model.cfg.validation_ds.min_duration = min_duration
            self.model.cfg.validation_ds.max_duration = max_duration
            self.model.cfg.validation_ds.ds_kwargs.load_audio = load_audio
            self.model.cfg.validation_ds.filter_ids = filter_ids
            self.model.cfg.validation_ds.num_workers = 8
            self.model.cfg.validation_ds.shuffle = shuffle
            self.model.cfg.validation_ds.batch_duration = batch_duration
        with open_dict(self.model.cfg):
            self.model.cfg["old_prefix"] = old_prefix
        self.model.setup_validation_data(self.model.cfg.validation_ds)