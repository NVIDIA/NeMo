set -e

echo 'Uninstalling stuff'
pip uninstall -y nemo_toolkit
pip uninstall -y nemo_asr
pip uninstall -y nemo_nlp
pip uninstall -y nemo_tts
pip uninstall -y nemo_simple_gan

echo 'Installing stuff'
cd nemo
pip install -e .
cd ../collections/nemo_asr
pip install -e .
cd ../nemo_nlp
pip install -e .
cd ../nemo_simple_gan
pip install -e .
cd ../nemo_tts
pip install -e .

cd ..
echo 'All done!'
