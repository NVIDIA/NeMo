pipeline {
  agent {
        docker {
            image 'nvcr.io/nvidia/pytorch:20.06-py3'
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

    stage('L0: Computer Vision Integration') {
      when {
        anyOf{
          branch 'candidate'
          changeRequest target: 'candidate'
        }
      }
      failFast true
      parallel {
        stage ('MNIST image classification with LeNet-5 Integration Test - on CPU') {
          steps {
            sh 'cd examples/cv && \
            python mnist_lenet5_image_classification_pure_lightning.py trainer.gpus=0 \
            trainer.fast_dev_run=true model.dataset.data_folder=/home/TestData \
            && rm -rf outputs'
          }
        }
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

    stage('L2: ASR dev run') {
      when {
        anyOf{
          branch 'candidate'
          changeRequest target: 'candidate'
        }
      }
      failFast true
      parallel {
        stage('Speech to Text') {
          steps {
            sh 'python examples/asr/speech_to_text.py \
            model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
            model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
            trainer.gpus=[0] \
            +trainer.fast_dev_run=True \
            exp_manager.root_dir=examples/asr/speech_to_text_results'
            sh 'rm -rf examples/asr/speech_to_text_results'
          }
        }

        stage('Speech to Label') {
          steps {
            sh 'python examples/asr/speech_to_label.py \
            model.train_ds.manifest_filepath=/home/TestData/speech_commands/train_manifest.json \
            model.validation_ds.manifest_filepath=/home/TestData/speech_commands/test_manifest.json \
            trainer.gpus=[1] \
            +trainer.fast_dev_run=True \
            model.preprocessor.cls=nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor \
            model.preprocessor.params=null \
            exp_manager.root_dir=examples/asr/speech_to_label_results'
            sh 'rm -rf examples/asr/speech_to_label_results'
          }
        }

        stage('Speaker Recognition') {
          steps {
            sh 'python examples/speaker_recognition/speaker_reco.py \
            model.train_ds.batch_size=10 \
            model.validation_ds.batch_size=2 \
            model.train_ds.manifest_filepath=/home/TestData/an4_speaker/train.json \
            model.validation_ds.manifest_filepath=/home/TestData/an4_speaker/dev.json \
            trainer.gpus=[1] \
            +trainer.fast_dev_run=True \
            exp_manager.root_dir=examples/speaker_recognition/speaker_recognition_results'
            sh 'rm -rf examples/speaker_recognition/speaker_recognition_results'
          }
        }
      }
    }
    


    stage('L2: Parallel BERT SQUAD v1.1 / v2.0') {
      when {
        anyOf{
          branch 'candidate'
          changeRequest target: 'candidate'
        }
      }
      failFast true
      parallel {
        stage('BERT SQUAD 1.1') {
          steps {
            sh 'cd examples/nlp/question_answering && \
            python question_answering_squad.py \
            model.train_ds.file=/home/TestData/nlp/squad_mini/v1.1/train-v1.1.json \
            model.train_ds.use_cache=false \
            model.validation_ds.file=/home/TestData/nlp/squad_mini/v1.1/dev-v1.1.json \
            model.validation_ds.use_cache=false \
            model.language_model.pretrained_model_name=bert-base-uncased \
            model.version_2_with_negative=false \
            trainer.precision=16 \
            trainer.amp_level=O1 \
            trainer.gpus=[0] \
            +trainer.fast_dev_run=true \
            exp_manager.root_dir=exp_bert_squad_1.1 \
            '
            sh 'rm -rf examples/nlp/question_answering/exp_bert_squad_1.1'
          }
        }
        stage('BERT SQUAD 2.0') {
          steps {
            sh 'cd examples/nlp/question_answering && \
            python question_answering_squad.py \
            model.train_ds.file=/home/TestData/nlp/squad_mini/v2.0/train-v2.0.json \
            model.train_ds.use_cache=false \
            model.validation_ds.file=/home/TestData/nlp/squad_mini/v2.0/dev-v2.0.json \
            model.language_model.pretrained_model_name=bert-base-uncased \
            model.version_2_with_negative=true \
            trainer.precision=16 \
            trainer.amp_level=O1 \
            trainer.gpus=[1] \
            +trainer.fast_dev_run=true \
            exp_manager.root_dir=exp_bert_squad_2.0 \
            '
            sh 'rm -rf examples/nlp/question_answering/exp_bert_squad_2.0'
          }
        }
      }

    }
    stage('L2: Parallel RoBERTa SQUAD v1.1 / v2.0') {
      when {
        anyOf{
          branch 'candidate'
          changeRequest target: 'candidate'
        }
      }
      failFast true
      parallel {
        stage('RoBERTa SQUAD 1.1') {
          steps {
            sh 'cd examples/nlp/question_answering && \
            python question_answering_squad.py \
            model.train_ds.file=/home/TestData/nlp/squad_mini/v1.1/train-v1.1.json \
            model.train_ds.use_cache=false \
            model.validation_ds.file=/home/TestData/nlp/squad_mini/v1.1/dev-v1.1.json \
            model.language_model.do_lower_case=true \
            model.language_model.pretrained_model_name=roberta-base \
            model.version_2_with_negative=false \
            trainer.precision=16 \
            trainer.amp_level=O1 \
            trainer.gpus=[0] \
            +trainer.fast_dev_run=true \
            exp_manager.root_dir=exp_roberta_squad_1.1 \
            '
            sh 'rm -rf examples/nlp/question_answering/exp_roberta_squad_1.1'
          }
        }
        stage('RoBERTa SQUAD 2.0') {
          steps {
            sh 'cd examples/nlp/question_answering && \
            python question_answering_squad.py \
            model.train_ds.file=/home/TestData/nlp/squad_mini/v2.0/train-v2.0.json \
            model.train_ds.use_cache=false \
            model.validation_ds.file=/home/TestData/nlp/squad_mini/v2.0/dev-v2.0.json \
            model.language_model.do_lower_case=true \
            model.language_model.pretrained_model_name=roberta-base \
            model.version_2_with_negative=true \
            trainer.precision=16 \
            trainer.amp_level=O1 \
            trainer.gpus=[1] \
            +trainer.fast_dev_run=true \
            exp_manager.root_dir=exp_roberta_squad_2.0 \
            '
            sh 'rm -rf examples/nlp/question_answering/exp_roberta_squad_2.0'
          }
        }
      }
    }

    stage('L2: Parallel NLP Examples 1') {
      when {
        anyOf{
          branch 'candidate'
          changeRequest target: 'candidate'
        }
      }
      failFast true
      parallel {
        stage ('Text Classification with BERT Test') {
          steps {
            sh 'cd examples/nlp/text_classification && \
            python text_classification_with_bert.py \
            model.language_model.pretrained_model_name=bert-base-uncased \
            model.language_model.max_seq_length=50 \
            model.data_dir=/home/TestData/nlp/retail/ \
            model.validation_ds.prefix=dev \
            model.train_ds.batch_size=10 \
            model.train_ds.use_cache=false \
            model.language_model.do_lower_case=true \
            trainer.gpus=[0] \
            +trainer.fast_dev_run=true \
            exp_manager.root_dir=exp_bert_base_uncased \
            '
            sh 'rm -rf examples/nlp/text_classification/exp_bert_base_uncased'
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
   stage('L2: NER') {
      when {
        anyOf{
          branch 'candidate'
          changeRequest target: 'candidate'
        }
      }
      failFast true
        steps {
          sh 'cd examples/nlp/token_classification && \
          python ner.py \
          model.data_dir=/home/TestData/nlp/token_classification_punctuation/ \
          trainer.gpus=[0] \
          +trainer.fast_dev_run=true \
          model.use_cache=false \
          '
        }
    }

    stage('L2: Punctuation and capitalization: DistilBert + MultiGPU') {
      when {
        anyOf{
          branch 'candidate'
          changeRequest target: 'candidate'
        }
      }
      failFast true
        steps {
          sh 'cd examples/nlp/token_classification && \
          python punctuation_capitalization.py \
          model.data_dir=/home/TestData/nlp/token_classification_punctuation/ \
          model.language_model.pretrained_model_name=distilbert-base-uncased \
          model.use_cache=false \
          trainer.gpus=[0,1] \
          trainer.distributed_backend=ddp \
          +trainer.fast_dev_run=true \
          exp_manager.root_dir=exp_distilbert_base_uncased \
          '
          sh 'rm -rf examples/nlp/token_classification/exp_distilbert_base_uncased'
        }
    }

     stage('L2: NER with cased Megatron') {
          when {
            anyOf{
              branch 'candidate'
              changeRequest target: 'candidate'
            }
          }          
          failFast true
          steps {
                sh 'cd examples/nlp/token_classification && \
                python ner.py \
                model.data_dir=/home/TestData/nlp/token_classification_punctuation/ \
                trainer.gpus=[0] \
                +trainer.fast_dev_run=true \
                model.use_cache=false \
                model.language_model.pretrained_model_name=megatron-bert-345m-cased trainer.distributed_backend=null \
                exp_manager.root_dir=exp_ner_megatron_bert_base_cased'
                sh 'rm -rf examples/nlp/token_classification/exp_ner_megatron_bert_base_cased'
           }
    }

    stage('L2: NER with uncased Megatron') {
          when {
            anyOf{
              branch 'candidate'
              changeRequest target: 'candidate'
            }
          }          
          failFast true
	  steps {
                sh 'cd examples/nlp/token_classification && \
                python ner.py \
                model.data_dir=/home/TestData/nlp/token_classification_punctuation/ \
                trainer.gpus=[0] \
                +trainer.fast_dev_run=true \
                model.use_cache=false \
                model.language_model.pretrained_model_name=megatron-bert-345m-uncased trainer.distributed_backend=null \
                exp_manager.root_dir=exp_ner_megatron_bert_base_uncased'
                sh 'rm -rf examples/nlp/token_classification/exp_ner_megatron_bert_base_uncased'
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
