python translate_ddp.py \
--model=/raid/nemo_models/teacher_24_6_de_en/AAYNBase.nemo \
--text2translate=/raid/wmt21_de_en_yttm_tokens_8000/parallel.batches.tokens.8000._OP_0..3517_CL_.tar \
--src_language de \
--tgt_language en \
--metadata_path /raid/wmt21_de_en_yttm_tokens_8000/metadata.tokens.8000.json \
--twoside \
--result_dir /raid/results/teacher_predictions_dataset