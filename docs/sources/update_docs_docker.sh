cd ../../
docker run --rm -v $PWD:/workspace python:3.7 /bin/bash -c "cd /workspace && pip install -r requirements/requirements_docs.txt && cd docs/sources/ && bash update_docs.sh"
