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

    stage('L0: Unit Tests GPU') {
      steps {
        sh 'pytest -m "unit and not skipduringci and not pleasefixme"'
      }
    }

    stage('L0: Unit Tests CPU') {
      when {
        anyOf{
          branch 'candidate'
          changeRequest target: 'candidate'
        }
      }
      steps {
        sh 'pytest -m "unit and not pleasefixme" --cpu'
      }
    }

    // We have no integration tests, please enable this when one is added
    // stage('L0: Integration Tests GPU') {
    //   steps {
    //     sh 'pytest -s -m "integration and not skipduringci and not pleasefixme"'
    //   }
    // }

    // stage('L0: Integration Tests CPU') {
    //   when {
    //     anyOf{
    //       branch 'candidate'
    //       changeRequest target: 'candidate'
    //     }
    //   }
    //   steps {
    //     sh 'pytest -s -m "integration and not pleasefixme" --cpu'
    //   }
    // }

    // We have no system tests, please enable this when one is added
    // stage('L1: System Tests GPU') {
    //   steps {
    //     sh 'pytest -m "system and not skipduringci and not pleasefixme"'
    //   }
    // }

    // stage('L1: System Tests CPU') {
    //   when {
    //     anyOf{
    //       branch 'candidate'
    //       changeRequest target: 'candidate'
    //     }
    //   }
    //   steps {
    //     sh 'pytest -m "system and not pleasefixme" --cpu'
    //   }
    // }

    stage('L2: Speech 2 Text dev run') {
      steps {
        sh 'python examples/asr/speech_to_text.py model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json pl.trainer.gpus=1 +pl.trainer.fast_dev_run=True'
      }
    }

    stage('L2: BERT Squad v1.1') {
      when {
        anyOf{
          branch 'candidate'
          changeRequest target: 'candidate'
        }
      }
      failFast true
        steps {
          sh 'cd examples/nlp && CUDA_VISIBLE_DEVICES=0 python question_answering_squad.py --precision 16 --amp_level=O1 --train_file /home/TestData/nlp/squad_mini/v1.1/train-v1.1.json --eval_file /home/TestData/nlp/squad_mini/v1.1/dev-v1.1.json --batch_size 8 --gpus 1 --do_lower_case --pretrained_model_name bert-base-uncased --optimizer adamw --lr 5e-5 --max_steps 2 --scheduler WarmupAnnealing'
          sh 'rm -rf examples/nlp/lightning_logs && rm -rf /home/TestData/nlp/squad_mini/v1.1/*cache*'
        }
    }

    stage('L2: BERT Squad v2.0') {
      when {
        anyOf{
          branch 'candidate'
          changeRequest target: 'candidate'
        }
      }
      failFast true
        steps {
          sh 'cd examples/nlp && CUDA_VISIBLE_DEVICES=0 python question_answering_squad.py --precision 16 --amp_level=O1 --train_file /home/TestData/nlp/squad_mini/v2.0/train-v2.0.json --eval_file /home/TestData/nlp/squad_mini/v2.0/dev-v2.0.json --batch_size 8 --gpus 1 --do_lower_case --pretrained_model_name bert-base-uncased --optimizer adamw --lr 1e-5 --max_steps 2 --version_2_with_negative --scheduler WarmupAnnealing'
          sh 'rm -rf examples/nlp/lightning_logs && rm -rf /home/TestData/nlp/squad_mini/v2.0/*cache*'
        }
    }

    stage('L2: Roberta Squad v1.1') {
      when {
        anyOf{
          branch 'candidate'
          changeRequest target: 'candidate'
        }
      }
      failFast true
        steps {
          sh 'cd examples/nlp && CUDA_VISIBLE_DEVICES=0 python question_answering_squad.py --precision 16 --amp_level=O1 --train_file /home/TestData/nlp/squad_mini/v1.1/train-v1.1.json --eval_file /home/TestData/nlp/squad_mini/v1.1/dev-v1.1.json --batch_size 5 --gpus 1 --pretrained_model_name roberta-base --optimizer adamw --lr 1e-5 --max_steps 2 --scheduler WarmupAnnealing'
          sh 'rm -rf examples/nlp/lightning_logs && rm -rf /home/TestData/nlp/squad_mini/v1.1/*cache*'
        }
    }
    stage('L2: Parallel NLP Examples 1') {
      failFast true
      parallel {
        stage ('Text Classification with BERT Test') {
          steps {
            sh 'cd examples/nlp/text_classification && python text_classification_with_bert.py pl.trainer.gpus=1 model.language_model.pretrained_model_name=bert-base-uncased pl.trainer.max_epochs=1 model.language_model.max_seq_length=50 model.data_dir=/home/TestData/nlp/retail/ model.validation_ds.prefix=dev model.train_ds.batch_size=10 model.train_ds.num_samples=-1 model.language_model.do_lower_case=true'
            sh 'rm -rf examples/nlp/text_classification/outputs'
          }
        }
      }
    }

    stage('L2: NLP-BERT pretraining BERT on the fly preprocessing') {
      when {
        anyOf{
          branch 'candidate'
          changeRequest target: 'candidate'
        }
      }
      failFast true
        steps {
          sh 'cd examples/nlp && CUDA_VISIBLE_DEVICES=0 python bert_pretraining_from_text.py --precision 16 --amp_level=O1 --data_dir /home/TestData/nlp/wikitext-2/  --batch_size 64 --config_file /home/TestData/nlp/bert_configs/bert_3200.json --lr 0.01 --warmup_ratio 0.05 --max_steps=2 --tokenizer_name=sentencepiece --sample_size 10000000 --mask_probability 0.15 --short_seq_prob 0.1'
          sh 'rm -rf examples/nlp/lightning_logs'
        }
    }
    stage('L2: NLP-BERT pretraining BERT offline preprocessing') {
      when {
        anyOf{
          branch 'candidate'
          changeRequest target: 'candidate'
        }
      }
      failFast true
        steps {
          sh 'cd examples/nlp && CUDA_VISIBLE_DEVICES=0 python bert_pretraining_from_preprocessed.py --precision 16 --amp_level=O1 --data_dir /home/TestData/nlp/wiki_book_mini/training --batch_size 8 --config_file /home/TestData/nlp/bert_configs/uncased_L-12_H-768_A-12.json  --gpus 1 --warmup_ratio 0.01 --optimizer adamw  --opt_args weight_decay=0.01  --lr 0.875e-4 --max_steps 2'
          sh 'rm -rf examples/nlp/lightning_logs'
        }
    }
    stage('L3: NER') {
      when {
        anyOf{
          branch 'candidate'
          changeRequest target: 'candidate'
        }
      }
      failFast true
        steps {
          sh 'cd examples/nlp/token_classification && python ner.py \
          model.data_dir=/home/TestData/nlp/token_classification_punctuation/ +pl.trainer.fast_dev_run=true'
          sh 'rm -rf /home/TestData/nlp/token_classification_punctuation/*cache*'
        }
    }
    stage('L4: Punctuation and capitalization: DistilBert + MultiGPU') {
      when {
        anyOf{
          branch 'candidate'
          changeRequest target: 'candidate'
        }
      }
      failFast true
        steps {
          sh 'cd examples/nlp/token_classification && python punctuation_capitalization.py \
          model.data_dir=/home/TestData/nlp/token_classification_punctuation/ +pl.trainer.fast_dev_run=true \
          pl.trainer.gpus=2 pl.trainer.distributed_backend=ddp \
          model.language_model.pretrained_model_name=distilbert-base-uncased'
          sh 'rm -rf /home/TestData/nlp/token_classification_punctuation/*cache*'
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