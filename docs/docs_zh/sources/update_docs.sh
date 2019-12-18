rm ../html -rf
make clean
mkdir build/html
make html
cp -r build/html ../html
make clean

if [ -d "../zh" ]; then
    rm -r ../zh
fi

mv ../html ../zh
