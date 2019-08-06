set -e

echo 'Uninstalling stuff'
pip uninstall -y nemo
pip uninstall -y nemo_asr
pip uninstall -y nemo_nlp
pip uninstall -y nemo_lpr

echo 'Installing stuff'
cd nemo
python setup.py develop
cd ../collections/nemo_asr
python setup.py develop
cd ../nemo_nlp
python setup.py develop
cd ../nemo_lpr
python setup.py develop

echo 'All done!'
