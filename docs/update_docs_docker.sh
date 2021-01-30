cd ../
docker run --rm -v $PWD:/workspace python:3.7 /bin/bash -c "cd /workspace && pip install .[all] && cd docs/ && bash update_docs.sh"
