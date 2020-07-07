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

    stage('L2: Parallel NLP Examples 1') {
      failFast true
      parallel {
        stage ('Text Classification with BERT Test') {
          steps {
            sh 'cd examples/nlp/text_classification && CUDA_VISIBLE_DEVICES=0 python text_classification_with_bert.py --pretrained_model_name bert-base-uncased --max_epochs=1 --max_seq_length=50 --data_dir=/home/TestData/nlp/retail/ --eval_file_prefix=dev --batch_size=10 --num_train_samples=-1 --do_lower_case --work_dir=outputs'
            sh 'rm -rf examples/nlp/text_classification/outputs'
          }
        }
      }
    }
  }
  post {
    always {
        sh "chmod -R 777 ."
        cleanWs()
    }
  }
}