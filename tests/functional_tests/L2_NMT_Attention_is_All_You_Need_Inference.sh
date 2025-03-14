cd examples/nlp/machine_translation &&
    coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo nmt_transformer_infer.py \
        --model=/home/TestData/nlp/nmt/toy_data/enes_v16k_s100k_6x6.nemo \
        --srctext=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.test.src \
        --tgtout=/home/TestData/nlp/nmt/toy_data/out.txt \
        --target_lang en \
        --source_lang de
