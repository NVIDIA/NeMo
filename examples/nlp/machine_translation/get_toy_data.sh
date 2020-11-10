mkdir -p toy_data

echo "Generating copy example data"
sacrebleu -t wmt14 -l de-en --echo ref > toy_data/wmt14-de-en.ref
sacrebleu -t wmt12 -l de-en --echo ref > toy_data/wmt13-de-en.ref
yttm bpe --data toy_data/wmt14-de-en.ref --model toy_data/copy_tokenizer.BPE.1024.model --vocab_size 1024
echo "Run this example (a single 1080Ti is enough): python nmt_transformer.py --config-name=copy_example"

echo "Generating train-on-test data"
echo "This example aims to memorize few translations"
sacrebleu -t wmt14 -l de-en --echo src > toy_data/wmt14-de-en.src
cat toy_data/wmt14-de-en.* > toy_data/all.txt
yttm bpe --data toy_data/all.txt --model toy_data/tt_tokenizer.BPE.4096.model --vocab_size 4096
echo "Run this example (uses 4 GPUs): python nmt_transformer.py --config-name=train_on_test_example"
