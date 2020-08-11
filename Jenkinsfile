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

    stage('Copyright Headers check') {
      steps {
        sh 'python /home/TestData/check_copyright_header.py --dir .'
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
            exp_manager.exp_dir=examples/asr/speech_to_text_results'
            sh 'rm -rf examples/asr/speech_to_text_results'
          }
        }
        stage('Speech to Label') {
          steps {
            sh 'python examples/asr/speech_to_label.py \
            model.train_ds.manifest_filepath=/home/TestData/speech_commands/train_manifest.json \
            model.validation_ds.manifest_filepath=/home/TestData/speech_commands/test_manifest.json \
            model.test_ds.manifest_filepath=/home/TestData/speech_commands/test_manifest.json \
            trainer.gpus=[1] \
            +trainer.fast_dev_run=True \
            model.preprocessor.cls=nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor \
            model.preprocessor.params=null \
            exp_manager.exp_dir=examples/asr/speech_to_label_results'
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
            model.test_ds.manifest_filepath=/home/TestData/an4_speaker/test.json \
            trainer.gpus=[1] \
            +trainer.fast_dev_run=True \
            exp_manager.exp_dir=examples/speaker_recognition/speaker_recognition_results'
            sh 'rm -rf examples/speaker_recognition/speaker_recognition_results'
          }
        }

        stage('L2: Speech to Text WPE') {
          steps {
            sh 'python examples/asr/speech_to_text_bpe.py \
            --config-path="experimental/configs/" --config-name="config_bpe" \
            model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
            model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
            model.tokenizer.dir="/home/TestData/asr_tokenizers/an4_wpe_128/" \
            model.tokenizer.type="wpe" \
            trainer.gpus=[1] \
            +trainer.fast_dev_run=True \
            exp_manager.exp_dir=examples/asr/speech_to_text_wpe_results'
            sh 'rm -rf examples/asr/speech_to_text_wpe_results'
          }
        }
      }
    }



    stage('L2: ASR Multi-dataloader dev run') {
      when {
        anyOf{
          branch 'candidate'
          changeRequest target: 'candidate'
        }
      }
      failFast true
      parallel {
        stage('Speech to Text multi-dataloader') {
          steps {
            sh 'python examples/asr/speech_to_text.py \
            model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
            model.validation_ds.manifest_filepath=[/home/TestData/an4_dataset/an4_val.json,/home/TestData/an4_dataset/an4_val.json] \
            trainer.gpus=[0] \
            trainer.max_epochs=1 \
            +trainer.max_steps=1 \
            +trainer.num_sanity_val_steps=1 \
            exp_manager.exp_dir=examples/asr/speech_to_text_results'
            sh 'rm -rf examples/asr/speech_to_text_results'
          }
        }

        stage('Speech to Label multi-dataloader') {
          steps {
            sh 'python examples/asr/speech_to_label.py \
            model.train_ds.manifest_filepath=/home/TestData/speech_commands/train_manifest.json \
            model.validation_ds.manifest_filepath=[/home/TestData/speech_commands/test_manifest.json,/home/TestData/speech_commands/test_manifest.json] \
            trainer.gpus=[1] \
            trainer.max_epochs=1 \
            +trainer.max_steps=1 \
            +trainer.num_sanity_val_steps=1 \
            model.preprocessor.cls=nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor \
            model.preprocessor.params=null \
            exp_manager.exp_dir=examples/asr/speech_to_label_results'
            sh 'rm -rf examples/asr/speech_to_label_results'
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
          // Cannot do fast_dev_run because squad needs whole dev dataset
          steps {
            sh 'cd examples/nlp/question_answering && \
            python question_answering_squad.py \
            model.train_ds.file=/home/TestData/nlp/squad_mini/v1.1/train-v1.1.json \
            model.dataset.use_cache=false \
            model.validation_ds.file=/home/TestData/nlp/squad_mini/v1.1/dev-v1.1.json \
            model.train_ds.batch_size=8 \
            model.validation_ds.batch_size=8 \
            trainer.max_epochs=1 \
            +trainer.max_steps=1 \
            model.language_model.pretrained_model_name=bert-base-uncased \
            model.dataset.version_2_with_negative=false \
            trainer.precision=16 \
            trainer.amp_level=O1 \
            trainer.gpus=[0] \
            trainer.num_sanity_val_steps=1000 \
            exp_manager.exp_dir=exp_bert_squad_1.1 \
            '
            sh 'rm -rf examples/nlp/question_answering/exp_bert_squad_1.1'
          }
        }
        stage('BERT SQUAD 2.0') {
          // Cannot do fast_dev_run because squad needs whole dev dataset
          steps {
            sh 'cd examples/nlp/question_answering && \
            python question_answering_squad.py \
            model.train_ds.file=/home/TestData/nlp/squad_mini/v2.0/train-v2.0.json \
            model.dataset.use_cache=false \
            model.train_ds.batch_size=8 \
            model.validation_ds.batch_size=8 \
            trainer.max_epochs=1 \
            +trainer.max_steps=1 \
            model.validation_ds.file=/home/TestData/nlp/squad_mini/v2.0/dev-v2.0.json \
            model.language_model.pretrained_model_name=bert-base-uncased \
            model.dataset.version_2_with_negative=true \
            trainer.precision=16 \
            trainer.amp_level=O1 \
            trainer.gpus=[1] \
            trainer.num_sanity_val_steps=1000 \
            exp_manager.exp_dir=exp_bert_squad_2.0 \
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
          // Cannot do fast_dev_run because squad needs whole dev dataset
          steps {
            sh 'cd examples/nlp/question_answering && \
            python question_answering_squad.py \
            model.train_ds.file=/home/TestData/nlp/squad_mini/v1.1/train-v1.1.json \
            model.dataset.use_cache=false \
            model.train_ds.batch_size=8 \
            model.validation_ds.batch_size=8 \
            trainer.max_epochs=1 \
            +trainer.max_steps=1 \
            model.validation_ds.file=/home/TestData/nlp/squad_mini/v1.1/dev-v1.1.json \
            model.dataset.do_lower_case=false \
            model.language_model.pretrained_model_name=roberta-base \
            model.dataset.version_2_with_negative=false \
            trainer.precision=16 \
            trainer.amp_level=O1 \
            trainer.gpus=[0] \
            trainer.num_sanity_val_steps=1000 \
            exp_manager.exp_dir=exp_roberta_squad_1.1 \
            '
            sh 'rm -rf examples/nlp/question_answering/exp_roberta_squad_1.1'
          }
        }
        stage('RoBERTa SQUAD 2.0') {
          // Cannot do fast_dev_run because squad needs whole dev dataset
          steps {
            sh 'cd examples/nlp/question_answering && \
            python question_answering_squad.py \
            model.train_ds.file=/home/TestData/nlp/squad_mini/v2.0/train-v2.0.json \
            model.dataset.use_cache=false \
            model.train_ds.batch_size=8 \
            model.validation_ds.batch_size=8 \
            trainer.max_epochs=1 \
            +trainer.max_steps=1 \
            model.validation_ds.file=/home/TestData/nlp/squad_mini/v2.0/dev-v2.0.json \
            model.dataset.do_lower_case=false \
            model.language_model.pretrained_model_name=roberta-base \
            model.dataset.version_2_with_negative=true \
            trainer.precision=16 \
            trainer.amp_level=O1 \
            trainer.gpus=[1] \
            trainer.num_sanity_val_steps=1000 \
            exp_manager.exp_dir=exp_roberta_squad_2.0 \
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
            model.train_ds.file_name=/home/TestData/nlp/retail/train.tsv \
            model.validation_ds.file_name=/home/TestData/nlp/retail/dev.tsv \
            model.language_model.pretrained_model_name=bert-base-uncased \
            model.train_ds.batch_size=10 \
            model.dataset.max_seq_length=50 \
            model.dataset.use_cache=false \
            model.dataset.do_lower_case=true \
            trainer.gpus=[0] \
            +trainer.fast_dev_run=true \
            exp_manager.exp_dir=exp_bert_base_uncased \
            '
            sh 'rm -rf examples/nlp/text_classification/exp_bert_base_uncased'
          }
        }
        stage ('NER with BERT') {
          steps {
            sh 'cd examples/nlp/token_classification && \
            python token_classification.py \
            model.dataset.data_dir=/home/TestData/nlp/token_classification_punctuation/ \
            trainer.gpus=[1] \
            +trainer.fast_dev_run=true \
            model.dataset.use_cache=false \
            '
          }
        }
      }
    }

    stage('L2: Intent and Slot Classification') {
      when {
        anyOf{
          branch 'candidate'
          changeRequest target: 'candidate'
        }
      }
      failFast true

      steps {
        sh 'cd examples/nlp/intent_slot_classification && \
        python intent_slot_classification.py \
        model.data_dir=/home/TestData/nlp/retail/ \
        model.validation_ds.prefix=dev \
        trainer.gpus=[0] \
        +trainer.fast_dev_run=true'
      }
    }

    stage('L2: Parallel GLUE Examples') {
      when {
        anyOf{
          branch 'candidate'
          changeRequest target: 'candidate'
        }
      }
      failFast true

      parallel {
        stage('MRPC') {
          steps {
            sh 'python examples/nlp/glue_benchmark/glue_benchmark.py \
            model.dataset.use_cache=false \
            model.task_name=mrpc \
            model.dataset.data_dir=/home/TestData/nlp/glue_fake/MRPC \
            trainer.gpus=[0] \
            +trainer.fast_dev_run=True \
            exp_manager.exp_dir=examples/nlp/glue_benchmark/mrpc'
            sh 'rm -rf examples/nlp/glue_benchmark/mrpc'
          }
        }
        stage('STS-b') {
          steps {
            sh 'python examples/nlp/glue_benchmark/glue_benchmark.py \
            model.dataset.use_cache=false \
            model.task_name=sts-b \
            model.dataset.data_dir=/home/TestData/nlp/glue_fake/STS-B \
            trainer.gpus=[1] \
            +trainer.fast_dev_run=True \
            exp_manager.exp_dir=examples/nlp/glue_benchmark/sts-b'
            sh 'rm -rf examples/nlp/glue_benchmark/sts-b'
          }
        }
      }
    }
    
    stage('L2: Parallel Pretraining BERT pretraining from Text/Preprocessed') {
      when {
        anyOf{
          branch 'candidate'
          changeRequest target: 'candidate'
        }
      }
      failFast true
      parallel {
        stage('L2: Pretraining BERT pretraining from Text') {
            steps {
              sh 'cd examples/nlp/language_modeling && \
              python bert_pretraining.py \
              --config-name=bert_pretraining_from_text_config.yaml \
              trainer.gpus=[0] \
              trainer.precision=16 \
              trainer.amp_level=O1 \
              +trainer.fast_dev_run=true \
              model.train_ds.data_file=/home/TestData/nlp/wikitext-2/train.txt  \
              model.train_ds.batch_size=64 \
              model.validation_ds.data_file=/home/TestData/nlp/wikitext-2/valid.txt  \
              model.validation_ds.batch_size=64 \
              model.language_model.bert_config_file=/home/TestData/nlp/bert_configs/bert_3200.json \
              model.optim.lr=0.01 \
              model.optim.sched.warmup_ratio=0.1 \
              model.tokenizer.tokenizer_name=sentencepiece \
              model.tokenizer.data_file=/home/TestData/nlp/wikitext-2/train.txt \
              model.tokenizer.sample_size=10000000 \
              model.mask_prob=0.15 \
              model.short_seq_prob=0.1 \
              exp_manager.exp_dir=PretrainingBERTFromText \
              '
              sh 'rm -rf /home/TestData/nlp/wikitext-2/spt'
              sh 'rm -f /home/TestData/nlp/wikitext-2/*.pkl'
              sh 'rm -rf examples/nlp/language_modeling/PretrainingBERTFromText'
              sh 'ls -lha examples/nlp/language_modeling'
            }
        }
        stage('L2: Pretraining BERT from Preprocessed') {
            steps {
              sh 'cd examples/nlp/language_modeling && \
              python bert_pretraining.py \
              --config-name=bert_pretraining_from_preprocessed_config.yaml \
              trainer.gpus=[1] \
              trainer.precision=16 \
              trainer.amp_level=O1 \
              +trainer.fast_dev_run=true \
              model.train_ds.data_file=/home/TestData/nlp/wiki_book_mini/training \
              model.train_ds.batch_size=8 \
              model.language_model.bert_checkpoint=/home/TestData/nlp/bert_ckpts/nemo1.0/bert_base_uncased_mlm_final_1074591_nemo1.0.pt \
              model.language_model.bert_config_file=/home/TestData/nlp/bert_configs/uncased_L-12_H-768_A-12.json \
              model.optim.lr=0.875e-4 \
              model.optim.weight_decay=0.01 \
              model.optim.sched.warmup_ratio=0.01 \
              exp_manager.exp_dir=PretrainingBERTFromPreprocessed \
              '
              sh 'rm -rf examples/nlp/language_modeling/PretrainingBERTFromPreprocessed'
              sh 'ls -lha examples/nlp/language_modeling'
            }
        }
      }
    }

   stage('L2: Punctuation & Capitalization, 2GPUs with DistilBERT') {
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
        model.dataset.data_dir=/home/TestData/nlp/token_classification_punctuation/ \
        model.language_model.pretrained_model_name=distilbert-base-uncased \
        model.dataset.use_cache=false \
        trainer.gpus=[0,1] \
        trainer.distributed_backend=ddp \
        +trainer.fast_dev_run=true \
        exp_manager.exp_dir=exp_distilbert_base_uncased \
        '
        sh 'rm -rf examples/nlp/token_classification/exp_distilbert_base_uncased'
      }
    }

    stage('L2: NER with cased Megatron') {
      // Megatron uses all GPUs under the hood, cannot parallel
      when {
        anyOf{
          branch 'candidate'
          changeRequest target: 'candidate'
        }
      }
      failFast true
      steps {
        sh 'cd examples/nlp/token_classification && \
        python token_classification.py \
        model.dataset.data_dir=/home/TestData/nlp/token_classification_punctuation/ \
        trainer.gpus=[0] \
        +trainer.fast_dev_run=true \
        model.dataset.use_cache=false \
        model.language_model.pretrained_model_name=megatron-bert-345m-cased trainer.distributed_backend=null \
        exp_manager.exp_dir=exp_ner_megatron_bert_base_cased'
        sh 'rm -rf examples/nlp/token_classification/exp_ner_megatron_bert_base_cased'
      }
    }

    stage('L2: NER with uncased Megatron') {
      // Megatron uses all GPUs under the hood, cannot parallel
      when {
        anyOf{
          branch 'candidate'
          changeRequest target: 'candidate'
        }
      }
      failFast true
      steps {
        sh 'cd examples/nlp/token_classification && \
        python token_classification.py \
        model.dataset.data_dir=/home/TestData/nlp/token_classification_punctuation/ \
        trainer.gpus=[0] \
        +trainer.fast_dev_run=true \
        model.dataset.use_cache=false \
        model.language_model.pretrained_model_name=megatron-bert-uncased \
        model.language_model.bert_checkpoint=/home/TestData/nlp/megatron_345m_uncased/model_optim_rng.pt \
        model.language_model.bert_config=/home/TestData/nlp/megatron_345m_uncased/345m_config.json \
        trainer.distributed_backend=null \
        exp_manager.exp_dir=exp_ner_megatron_bert_base_uncased'
        sh 'rm -rf examples/nlp/token_classification/exp_ner_megatron_bert_base_uncased'
      }
    }

    stage('L2: TTS Fast dev runs 1') {
      when {
        anyOf{
          branch 'candidate'
          changeRequest target: 'candidate'
        }
      }
      parallel {
        stage('Tacotron 2') {
          steps {
            sh 'python examples/tts/tacotron2.py \
            train_dataset=/home/TestData/an4_dataset/an4_train.json \
            validation_datasets=/home/TestData/an4_dataset/an4_val.json \
            trainer.gpus="[0]" \
            +trainer.fast_dev_run=True \
            trainer.distributed_backend=null \
            trainer.max_epochs=-1 \
            model.train_ds.dataloader_params.batch_size=12 \
            model.validation_ds.dataloader_params.batch_size=12'
          }
        }
        stage('WaveGlow') {
          steps {
            sh 'python examples/tts/waveglow.py \
            train_dataset=/home/TestData/an4_dataset/an4_train.json \
            validation_datasets=/home/TestData/an4_dataset/an4_val.json \
            trainer.gpus="[1]" \
            +trainer.fast_dev_run=True \
            trainer.distributed_backend=null \
            trainer.max_epochs=-1 \
            model.train_ds.dataloader_params.batch_size=4 \
            model.validation_ds.dataloader_params.batch_size=4'
          }
        }
      }
    }

    stage('L2: TTS Fast dev runs 2') {
      when {
        anyOf{
          branch 'candidate'
          changeRequest target: 'candidate'
        }
      }

      parallel {
        stage('GlowTTS') {
          steps {
            sh 'python examples/tts/glow_tts.py \
            train_dataset=/home/TestData/an4_dataset/an4_train.json \
            validation_datasets=/home/TestData/an4_dataset/an4_val.json \
            trainer.gpus="[1]" \
            +trainer.fast_dev_run=True \
            trainer.distributed_backend=null \
            trainer.max_epochs=-1 \
            model.train_ds.batch_size=4 \
            model.validation_ds.batch_size=4'
          }
        }
      }
    }

    stage('L??: ASR Checkpoints tests') {
      when {
        anyOf{
          branch 'candidate'
          changeRequest target: 'candidate'
        }
      }
      failFast true
      parallel {
        stage('QuartzNet15x5Base-En') {
          steps {
            sh 'python examples/asr/speech_to_text_infer.py --asr_model QuartzNet15x5Base-En --dataset /home/TestData/librispeech/librivox-dev-other.json --wer_tolerance 0.1011 --batch_size 64'
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
