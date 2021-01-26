**STEPS 2 RUN**
===============
0. (Make sure you have Flask installed - ``pip install flask``)
1. Train NMT model derived from ``nemo.collections.nlp.models.machine_translation.mt_enc_dec_model.MTEncDecModel``
2. Download resulting .nemo file and store it locally at location PATH2NEMO_FILE
3. In ``nmt_service.py`` file set PATH2NEMO_FILE to correct location
4. To run: ``python nmt_service.py``
5. To translate: ``http://127.0.0.1:5000/translate?text=Frohe%20Weihnachten`` (here %20 means space)