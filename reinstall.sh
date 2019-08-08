set -e

echo 'Uninstalling stuff'
pip uninstall -y nemo
pip uninstall -y nemo_asr

echo 'Installing stuff'
cd nemo
python setup.py develop
cd ../collections/nemo_asr
python setup.py develop

echo 'All done!'
