pip install -r requirements/requirements.txt \
  && pip install -r requirements/requirements_nlp.txt \
  && export nemo_path=$(pwd) \
  && export HYDRA_FULL_ERROR=1 \
  && echo "NeMo path: ${nemo_path}" \
  && export PYTHONPATH="${nemo_path}" \
  && mkdir -p /data \
  && cd /data \
  && wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.de -O valid.de \
  && wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.en -O valid.en \
  && wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de -O test.de \
  && wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en -O test.en \
  && wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de \
  && wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en \
  && cat train.en train.de > yttm_train.ende \
  && echo "current path when creating yttm model: $(pwd)" \
  && yttm bpe --data yttm_train.ende --model bpe_32k_en_de_yttm.model --vocab_size 32000 \
  && mkdir -p ../wmt14_en_de2 \
  && cd ../wmt14_en_de2 \
  && cp ../wmt14_en_de/bpe_32k_en_de_yttm.model ./ \
  && cp ../wmt14_en_de/test* ./ \
  && cp ../wmt14_en_de/valid* ./ \
  && cp valid.en train.en \
  && cp valid.de train.de \
  && cd  "${nemo_path}/examples/nlp/machine_translation" \
  && python transformer_mt.py -cn debug_on_ngc

