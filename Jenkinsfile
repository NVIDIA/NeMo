pipeline {
  agent {
        docker {
            image 'nvcr.io/nvidia/pytorch:20.01-py3'
            args '--device=/dev/nvidia0 --gpus all --user 0:128 -v /home/TestData:/home/TestData -v $HOME/.cache/torch:/root/.cache/torch --shm-size=8g'
        }
  }
  options {
    timeout(time: 1, unit: 'HOURS')
    disableConcurrentBuilds()
  }
  stages {

    stage('PyTorch version') {
      steps {
        sh 'python -c "import torch; print(torch.__version__)"'
      }
    }
    stage('Install test requirements') {
      steps {
        sh 'apt-get update && apt-get install -y bc && pip install -r requirements/requirements_test.txt'
      }
    }
    stage('Code formatting checks') {
      steps {
        sh 'python setup.py style'
      }
    }

  post {
    always {
        sh "chmod -R 777 ."
        cleanWs()
    }
  }
}