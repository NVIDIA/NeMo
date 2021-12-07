if [ ! -d "$1" ]; then
    echo "Error: First argument must be a valid directory. Recommended path is scripts/tts_dataset_files"
    exit 1
fi

echo "Downloading cmudict-0.7b from official site to $1"
wget http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b -P $1