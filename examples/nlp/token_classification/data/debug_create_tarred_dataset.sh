python create_punctuation_capitalization_tarred_dataset.py \
  --text /media/apeganov/DATA/wiki_unsplit_48_65_filtered_characters/very_small_wiki/train/input.txt \
  --labels /media/apeganov/DATA/wiki_unsplit_48_65_filtered_characters/very_small_wiki/train/bert_labels.txt \
  --output_dir /media/apeganov/DATA/wiki_unsplit_48_65_filtered_characters/very_small_wiki/train_tarred \
  --lines_per_dataset_fragment 100000 \
  --num_batches_per_tarfile 50
