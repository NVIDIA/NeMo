#/bin/bash

PORT=${1:-8000}
echo "Serving docs on : 0.0.0.0:${PORT}"

docker pull squidfunk/mkdocs-material && docker run --rm -it -p ${PORT}:${PORT} -v ${PWD}:/docs squidfunk/mkdocs-material serve -a 0.0.0.0:${PORT}
