def get_llava_data_module(model_id: str, data_path: str, mbs: int, gbs: int):
    from transformers import AutoProcessor

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
    from nemo.collections.multimodal.data.energon import SimpleMultiModalDataModule
    from nemo.collections.multimodal.data.energon.config import MultiModalSampleConfig
    from nemo.collections.vlm import LlavaNextTaskEncoder

    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = AutoTokenizer(model_id)

    multimodal_sample_config = MultiModalSampleConfig()
    # Setting system prompt to empty string
    multimodal_sample_config.conversation_template_config.system = ''

    task_encoder = LlavaNextTaskEncoder(
        tokenizer=tokenizer.tokenizer,
        image_processor=processor.image_processor,
        multimodal_sample_config=multimodal_sample_config,
    )
    return SimpleMultiModalDataModule(
        path=data_path,
        tokenizer=tokenizer,
        image_processor=processor.image_processor,
        num_workers=32,
        micro_batch_size=mbs,
        global_batch_size=gbs,
        multimodal_sample_config=multimodal_sample_config,
        task_encoder=task_encoder,
    )