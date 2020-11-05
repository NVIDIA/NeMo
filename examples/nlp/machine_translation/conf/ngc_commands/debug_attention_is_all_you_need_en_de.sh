pip install -r requirements/requirements.txt \
  && pip install -r requirements/requirements_nlp.txt \
  && export nemo_path=$(pwd) \
  && export HYDRA_FULL_ERROR=1 \
  && echo "NeMo path: ${nemo_path}" \
  && export PYTHONPATH="${nemo_path}" \
  && mkdir -p /data/wmt14_en_de \
  && cd /data/wmt14_en_de \
  && wget -q -O valid.de https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.de \
  && wget -q -O valid.en https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.en \
  && wget -q -O test.de https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de \
  && wget -q -O test.en https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en \
  && wget -q https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de \
  && wget -q https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en \
  && cat train.en train.de > yttm_train.ende \
  && echo "current path when creating yttm model: $(pwd)" \
  && yttm bpe --data yttm_train.ende --model bpe_37k_en_de_yttm.model --vocab_size 37000 \
  && mkdir -p ../wmt14_en_de2 \
  && cd ../wmt14_en_de2 \
  && cp ../wmt14_en_de/bpe_37k_en_de_yttm.model ./ \
  && cp ../wmt14_en_de/test* ./ \
  && cp ../wmt14_en_de/valid* ./ \
  && cp valid.en train.en \
  && cp valid.de train.de \
  && cd  "${nemo_path}/examples/nlp/machine_translation" \
  && python train.py -cn debug_on_ngc \
  && export best_ckpt_path=$(cat best_checkpoint_path.txt) \
  && echo "best ckpt path:" ${best_ckpt_path} \
  && ln -s ${best_ckpt_path} best.ckpt \
  && python test.py model.test_checkpoint_path=best.ckpt -cn debug_on_ngc

