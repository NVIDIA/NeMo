#/bin/bash

docker pull squidfunk/mkdocs-material && \
docker run --rm -it -v ${PWD}:/docs --entrypoint "/bin/sh" squidfunk/mkdocs-material -c \
  "cd /docs && \
  pip install -r /docs/requirements/requirements.txt && \
  mkdocs build"
