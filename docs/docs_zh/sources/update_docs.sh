rm ../html -rf
make clean
mkdir build/html
make html
cp -r build/html ../html
make clean
rm -r ../zh
mv ../html ../zh
