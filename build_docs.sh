#/bin/bash

docker pull squidfunk/mkdocs-material && docker run --rm -it -v ${PWD}:/docs squidfunk/mkdocs-material build 
