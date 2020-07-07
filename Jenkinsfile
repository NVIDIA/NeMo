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

    stage('Installation') {
      steps {
        sh './reinstall.sh'
      }
    }

    stage('L2: NLP-BERT pretraining BERT on the fly preprocessing') {
      when {
        anyOf{
          branch 'candidate'
          changeRequest()
        }
      }
      failFast true
        steps {
          sh 'cd examples/nlp && CUDA_VISIBLE_DEVICES=0 python bert_pretraining_from_text.py --precision 16 --amp_level=O1 --data_dir /home/TestData/nlp/wikitext-2/  --batch_size 64 --config_file /home/TestData/nlp/bert_configs/bert_3200.json --lr 0.01 --warmup_ratio 0.05 --max_steps=300 --tokenizer_name=sentencepiece --sample_size 10000000 --mask_probability 0.15 --short_seq_prob 0.1'
          sh 'rm -rf examples/nlp/lightning_logs'
        }
    }

    stage('L2: NLP-BERT pretraining BERT offline preprocessing') {
      when {
        anyOf{
          branch 'candidate'
          changeRequest()
        }
      }
      failFast true
        steps {
          sh 'cd examples/nlp && CUDA_VISIBLE_DEVICES=0 python bert_pretraining_from_preprocessed.py --precision 16 --amp_level=O1 --data_dir /home/TestData/nlp/wiki_book_mini/training --batch_size 8 --config_file /home/TestData/nlp/bert_configs/uncased_L-12_H-768_A-12.json  --gpus 1 --warmup_ratio 0.01 --optimizer adamw  --opt_args weight_decay=0.01  --lr 0.875e-4 --max_steps 300'
          sh 'rm -rf examples/nlp/lightning_logs'
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