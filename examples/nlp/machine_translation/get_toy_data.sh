mkdir -p toy_data
sacrebleu -t wmt14 -l de-en --echo src > toy_data/wmt14-de-en.src
sacrebleu -t wmt14 -l de-en --echo ref > toy_data/wmt14-de-en.ref
cat toy_data/wmt14* > toy_data/all.txt
yttm bpe --data toy_data/all.txt --model toy_data/tokenizer.BPE.1024.model --vocab_size 1024