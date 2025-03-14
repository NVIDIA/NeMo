coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo scripts/speech_recognition/oomptimizer.py \
  -c /home/TestData/oomptimizer/fast-conformer_aed.yaml \
  -m nemo.collections.asr.models.EncDecMultiTaskModel \
  -b "[[5.0,30],[5.0,45],[10.0,57],[10.0,71]]"
