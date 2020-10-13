pip install -r requirements/requirements.txt \
  && pip install -r requirements/requirements_nlp.txt \
  && cd .. \
  && mkdir -p data \
  && cd data \
  && wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.de -O valid.de \
  && wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.en -O valid.en \
  && cp valid.de test.de \
  && cp valid.en test.en \
  && wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de \
  && wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en \
  && cat train.en train.de > yttm_train.ende \
  && yttm bpe --data yttm_train.ende --model bpe_32k_en_de_yttm.model --vocab_size 32000 \
  && export PYTHONPATH=/NeMo \
  && cd  ../NeMo/examples/nlp/machine_translation \
  && python transformer_mt.py -cn ngc_8gpu

