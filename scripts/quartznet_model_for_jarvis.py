# Import NeMo and ASR collection
import nemo
import nemo.collections.asr as nemo_asr
nf = nemo.core.NeuralModuleFactory(placement=nemo.core.DeviceType.CPU)

pre_trained_qn_model = nemo_asr.models.QuartzNet.from_pretrained(model_info="QuartzNet15x5-En-BASE")
pre_trained_qn_model.export('quartznet.nemo', optimize_for_deployment=True)