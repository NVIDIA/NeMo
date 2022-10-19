## You can use any other G2P model instead of this, just see the expected output format. 

NEMO_PATH=/home/aleksandraa/nemo

PRETRAINED_MODEL=../g2p/nemo_experiments/training.nemo

export TOKENIZERS_PARALLELISM=false

## here we reuse inference script from normalization_as_tagging, because this is the same model
python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/normalization_as_tagging_infer.py \
  pretrained_model=${PRETRAINED_MODEL} \
  inference.from_file=yago.vocab.txt \
  inference.out_file=yago.vocab.to_cmu.output \
  model.max_sequence_len=1024 \
  inference.batch_size=256 \
  lang=en

python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/dataset_preparation/prepare_input_for_tts.py --yago_input_name yago.uniq3 --phonematic_name yago.vocab.to_cmu.output --output_name tts_input.txt

## tts_input.txt should have the following format (space is also a phoneme)
##aadityana       AA0,AA2,D,AH0,T,Y,AE1,N,AH0
##aadivaram aadavallaku selavu    AA2,D,IH1,V,ER0,AE2,M, ,AA2,AA0,D,AH0,V,AA1,L,AA1,K,UW2, ,S,EH1,L,AH0,V,UW0
##aa divasam      EY1,EY1, ,D,IH0,V,AH0,S,AA1,M
##aadi velli      AA1,D,IY0, ,V,EH1,L,IY0

