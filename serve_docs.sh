#/bin/bash

PORT=${1:-8000}
echo "Serving docs on : 0.0.0.0:${PORT}"

docker pull squidfunk/mkdocs-material && \
docker run --rm -it -p ${PORT}:${PORT} -v ${PWD}:/docs --entrypoint "/bin/sh" squidfunk/mkdocs-material -c \
  "cd /docs && \
  pip install -r /docs/requirements/requirements.txt && \
  mkdocs serve -a 0.0.0.0:${PORT}"
