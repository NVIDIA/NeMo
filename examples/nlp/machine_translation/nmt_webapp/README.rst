**STEPS 2 RUN**
===============
0. (Make sure you have Flask installed - ``pip install -r requirements.txt``)
1. Edit "config.json" file to only contain models you need. If model's location starts with "NGC/" - it will load this model from NVIDIA's NGC. Otherwise, specify full path to .nemo file.
4. To run: ``python nmt_service.py``
5. To translate: ``http://127.0.0.1:5000/translate?text=Frohe%20Weihnachten`` (here %20 means space)
6. Run web UI: ``python -m http.server``