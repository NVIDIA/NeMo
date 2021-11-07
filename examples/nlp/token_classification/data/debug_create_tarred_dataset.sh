python create_punctuation_capitalization_tarred_dataset.py \
  --text /media/apeganov/DATA/wiki_unsplit_48_65_filtered_characters/very_small_wiki/train/input.txt \
  --labels /media/apeganov/DATA/wiki_unsplit_48_65_filtered_characters/very_small_wiki/train/bert_labels.txt \
  --output_dir /media/apeganov/DATA/wiki_unsplit_48_65_filtered_characters/very_small_wiki_debug/train_tarred \
  --lines_per_dataset_fragment 10000 \
  --tokens_in_batch 8000 \
  --num_batches_per_tarfile 5 \
  --tokenizer_name char \
  --vocab_file /media/apeganov/DATA/wiki_unsplit_48_65_filtered_characters/very_small_wiki/char_vocabulary.txt
