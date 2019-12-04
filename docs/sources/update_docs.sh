rm -rf ../html 
make clean
mkdir build/html
make html
cp -r build/html ../html
make clean
