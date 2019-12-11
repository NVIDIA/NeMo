rm -rf ../html 
make clean
mkdir build/html
make html
cp -r build/html ../html
make clean

# generate mandarin doc
echo "begin to generate mandarin html"
cd ../docs_zh/sources
./update_docs.sh
cp -r ../zh ../../html/
echo "Finish"
