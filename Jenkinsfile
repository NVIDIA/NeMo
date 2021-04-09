pipeline {
  agent {
        docker {
      image 'nvcr.io/nvidia/pytorch:21.03-py3'
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
        sh 'python -c "import torchvision; print(torchvision.__version__)"'
      }
    }

    stage('Uninstall torchtext') {
      steps {
        sh 'pip uninstall -y torchtext'
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

    stage('PyTorch STFT Patch check') {
      steps {
        sh 'python /home/TestData/check_stft_patch.py --dir .'
      }
    }

    stage('Code formatting checks') {
      steps {
        sh 'python setup.py style'
      }
    }
    stage('Installation') {
      steps {
        sh './reinstall.sh release'
      }
    }

    stage('Install nemo_tools requirements') {
      steps {
        sh 'bash nemo_tools/setup.sh'
      }
    }

    stage('PyTorch Lightning version') {
      steps {
        sh 'python -c "import pytorch_lightning; print(pytorch_lightning.__version__)"'
      }
    }

    stage('L0: Unit Tests GPU') {
      steps {
        sh 'pytest -m "not pleasefixme" --with_downloads'
      }
    }

    stage('L0: Unit Tests CPU') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      steps {
        sh 'CUDA_VISIBLE_DEVICES="" pytest -m "not pleasefixme" --cpu --with_downloads'
      }
    }

    stage('L0: Computer Vision Integration') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage ('MNIST image classification with LeNet-5 Integration Test - on CPU') {
          steps {
            sh 'cd examples/cv && \
            python mnist_lenet5_image_classification_pure_lightning.py trainer.gpus=0 \
            trainer.accelerator=null \
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
    //       branch 'main'
    //       changeRequest target: 'main'
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
    //       branch 'dev
    //       changeRequest target: 'main'
    //     }
    //   }
    //   steps {
    //     sh 'pytest -m "system and not pleasefixme" --cpu'
    //   }
    // }

    stage('L2: NeMo tools') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('L2: pynini export') {
          steps {
            sh 'cd tools/text_denormalization && python pynini_export.py /home/TestData/nlp/text_denorm/output/ && ls -R /home/TestData/nlp/text_denorm/output/ && echo ".far files created "|| exit 1'
            sh 'cd tools/text_denormalization && cp *.grm /home/TestData/nlp/text_denorm/output/'
            sh 'ls -R /home/TestData/nlp/text_denorm/output/'
            sh 'cd nemo_tools/text_denormalization/ &&  python run_predict.py --input=/home/TestData/nlp/text_denorm/ci/test.txt --output=/home/TestData/nlp/text_denorm/output/test.pynini.txt --verbose'
            sh 'cmp --silent /home/TestData/nlp/text_denorm/output/test.pynini.txt /home/TestData/nlp/text_denorm/ci/test_goal_py.txt || exit 1'
            sh 'rm -rf /home/TestData/nlp/text_denorm/output/*'
          }
        }
      }
    }

    stage('L2: ASR dev run') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
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
        //         stage('Speech to Text - DALI AudioToMelSpectrogramPreprocessor') {
        //           steps {
        //             sh 'python examples/asr/speech_to_text.py \
        //             model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
        //             +model.train_ds.use_dali=True \
        //             model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
        //             +model.validation_ds.use_dali=True \
        //             trainer.gpus=[0] \
        //             +trainer.fast_dev_run=True \
        //             exp_manager.exp_dir=examples/asr/speech_to_text_results'
        //             sh 'rm -rf examples/asr/speech_to_text_results'
        //           }
        //         }
        //         stage('Speech to Text - DALI AudioToMFCCPreprocessor') {
        //           steps {
        //             sh 'python examples/asr/speech_to_text.py \
        //             model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
        //             +model.train_ds.use_dali=True \
        //             model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
        //             +model.validation_ds.use_dali=True \
        //             model.preprocessor._target_=nemo.collections.asr.modules.AudioToMFCCPreprocessor \
        //             trainer.gpus=[0] \
        //             +trainer.fast_dev_run=True \
        //             exp_manager.exp_dir=examples/asr/speech_to_text_results'
        //             sh 'rm -rf examples/asr/speech_to_text_results'
        //           }
        //         }
        stage('Speech to Label') {
          steps {
            sh 'python examples/asr/speech_to_label.py \
            model.train_ds.manifest_filepath=/home/TestData/speech_commands/train_manifest.json \
            model.validation_ds.manifest_filepath=/home/TestData/speech_commands/test_manifest.json \
            model.test_ds.manifest_filepath=/home/TestData/speech_commands/test_manifest.json \
            trainer.gpus=[1] \
            +trainer.fast_dev_run=True \
            model.preprocessor._target_=nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor \
            ~model.preprocessor.window_size \
            ~model.preprocessor.window_stride \
            ~model.preprocessor.window \
            ~model.preprocessor.n_mels \
            ~model.preprocessor.n_mfcc \
            ~model.preprocessor.n_fft \
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

        stage('Speaker Diarization Inference') {
          steps {
            sh 'python examples/speaker_recognition/speaker_diarize.py \
            diarizer.paths2audio_files=/home/TestData/an4_diarizer/audio_files.scp \
            diarizer.path2groundtruth_rttm_files=/home/TestData/an4_diarizer/rttm_files.scp \
            diarizer.speaker_embeddings.model_path=/home/TestData/an4_diarizer/spkr.nemo \
            diarizer.vad.model_path=/home/TestData/an4_diarizer/MatchboxNet_VAD_3x2.nemo \
            diarizer.out_dir=examples/speaker_recognition/speaker_diarization_results'
            sh 'rm -rf examples/speaker_recognition/speaker_diarization_results'
          }
        }

        stage('L2: Speech to Text WPE - CitriNet') {
          steps {
            sh 'python examples/asr/speech_to_text_bpe.py \
            --config-path="conf/citrinet/" --config-name="config_bpe" \
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

        stage('L2: Speech to Text WPE - Conformer') {
          steps {
            sh 'python examples/asr/speech_to_text_bpe.py \
            --config-path="conf/conformer" --config-name="conformer_ctc_bpe" \
            model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
            model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
            model.tokenizer.dir="/home/TestData/asr_tokenizers/an4_wpe_128/" \
            model.tokenizer.type="wpe" \
            model.train_ds.batch_size=10 \
            model.validation_ds.batch_size=10 \
            trainer.gpus=[1] \
            +trainer.fast_dev_run=True \
            exp_manager.exp_dir=examples/asr/speech_to_text_wpe_conformer_results'
            sh 'rm -rf examples/asr/speech_to_text_wpe_conformer_results'
          }
        }
      }
    }

//  TODO: UNCOMMENT TESTS AFTER 21.04 release (numba 0.53 min requirement)
//     stage('L2: ASR RNNT dev run') {
//       when {
//         anyOf {
//           branch 'main'
//           changeRequest target: 'main'
//         }
//       }
//       failFast true
//       parallel {
//         stage('Speech to Text - RNNT') {
//           steps {
//             sh 'python examples/asr/speech_to_text_rnnt.py \
//             model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
//             model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
//             model.train_ds.batch_size=8 \
//             trainer.gpus=[0] \
//             +trainer.fast_dev_run=True \
//             exp_manager.exp_dir=examples/asr/speech_to_text_rnnt_results'
//             sh 'rm -rf examples/asr/speech_to_text_rnnt_results'
//           }
//         }
//         stage('L2: Speech to Text RNNT WPE') {
//           steps {
//             sh 'python examples/asr/speech_to_text_rnnt_bpe.py \
//             --config-path="experimental/contextnet_rnnt/" --config-name="config_rnnt_bpe.yaml" \
//             model.train_ds.manifest_filepath=/home/TestData/an4_dataset/an4_train.json \
//             model.validation_ds.manifest_filepath=/home/TestData/an4_dataset/an4_val.json \
//             model.tokenizer.dir="/home/TestData/asr_tokenizers/an4_wpe_128/" \
//             model.tokenizer.type="wpe" \
//             trainer.gpus=[0] \
//             +trainer.fast_dev_run=True \
//             exp_manager.exp_dir=examples/asr/speech_to_text_rnnt_wpe_results'
//             sh 'rm -rf examples/asr/speech_to_text_rnnt_wpe_results'
//           }
//         }
//       }
//     }

    stage('L2: ASR Multi-dataloader dev run') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
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
            model.preprocessor._target_=nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor \
            ~model.preprocessor.window_size \
            ~model.preprocessor.window_stride \
            ~model.preprocessor.window \
            ~model.preprocessor.n_mels \
            ~model.preprocessor.n_mfcc \
            ~model.preprocessor.n_fft \
            exp_manager.exp_dir=examples/asr/speech_to_label_results'
            sh 'rm -rf examples/asr/speech_to_label_results'
          }
        }
      }
    }

    stage('L2: Speech Transcription') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('Speech to Text Transcribe') {
          steps {
            sh 'python examples/asr/transcribe_speech.py \
            pretrained_name="QuartzNet15x5Base-En" \
            audio_dir="/home/TestData/an4_transcribe/test_subset/" \
            cuda=true \
            amp=true'
            sh 'rm -rf examples/asr/speech_to_text_transcriptions.txt'
          }
        }
      }
    }

    stage('L2: Segmentation Tool') {
      when {
            anyOf {
              branch 'main'
              changeRequest target: 'main'
            }
      }
      stages {
        stage('Install ctc_segmentation requirements') {
            steps {
            sh 'cd tools/ctc_segmentation && \
            pip install -r requirements.txt && \
            DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg'
            }
        }

        stage('Parallel ctc_segmentation test') {
          failFast true
          parallel {
            stage('L2: Eng QN with .wav') {
              steps {
                sh 'cd tools/ctc_segmentation && \
            TIME=`date +"%Y-%m-%d-%T"` && \
            /bin/bash run_sample.sh \
            --MODEL_NAME_OR_PATH=QuartzNet15x5Base-En \
            --DATA_DIR=/home/TestData/ctc_segmentation/eng \
            --OUTPUT_DIR=/home/TestData/ctc_segmentation/eng/output${TIME} \
            --LANGUAGE=eng \
            --OFFSET=0 \
            --CUT_PREFIX=0 \
            --MIN_SEGMENT_LEN=0 \
            --AUDIO_FORMAT=.wav && \
            python /home/TestData/ctc_segmentation/verify_alignment.py \
            -r /home/TestData/ctc_segmentation/eng/eng_valid_segments.txt \
            -g /home/TestData/ctc_segmentation/eng/output${TIME}/verified_segments/nv_test_segments.txt && \
            rm -rf /home/TestData/ctc_segmentation/eng/output${TIME}'
              }
            }
            stage('L2: Ru QN with .mp3') {
              steps {
                sh 'cd tools/ctc_segmentation && \
            TIME=`date +"%Y-%m-%d-%T"` && \
            /bin/bash run_sample.sh \
            --MODEL_NAME_OR_PATH=/home/TestData/ctc_segmentation/QuartzNet15x5-Ru-e512-wer14.45.nemo \
            --DATA_DIR=/home/TestData/ctc_segmentation/ru \
            --OUTPUT_DIR=/home/TestData/ctc_segmentation/ru/output${TIME} \
            --LANGUAGE=ru \
            --OFFSET=0 \
            --CUT_PREFIX=0 \
            --MIN_SEGMENT_LEN=0 \
            --AUDIO_FORMAT=.mp3 \
            --ADDITIONAL_SPLIT_SYMBOLS=";" && \
            python /home/TestData/ctc_segmentation/verify_alignment.py \
            -r /home/TestData/ctc_segmentation/ru/valid_ru_segments.txt \
            -g /home/TestData/ctc_segmentation/ru/output${TIME}/verified_segments/ru_segments.txt && \
            rm -rf /home/TestData/ctc_segmentation/ru/output${TIME}'
              }
            }
          }
        }
      }
    }

    stage('L2: Multi-GPU Megatron finetuning') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('L2: Cased Megatron finetuning on MRPC') {
          steps {
            sh 'cd examples/nlp/glue_benchmark && \
        python glue_benchmark.py \
        model.dataset.data_dir=/home/TestData/nlp/glue_fake/MRPC \
        trainer.gpus=[0,1] \
        +trainer.fast_dev_run=true \
        model.dataset.use_cache=false \
        model.language_model.pretrained_model_name=megatron-bert-345m-cased \
        trainer.accelerator=ddp \
        exp_manager=null'
          }
        }
      }
    }

    stage('L2: SGD-QA') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('L2: SGD-QA') {
          steps {
            sh 'cd examples/nlp/dialogue_state_tracking && \
        python sgd_qa.py \
        model.dataset.data_dir=/home/TestData/nlp/sgd_small \
        model.dataset.dialogues_example_dir=sgd_outputs \
        model.dataset.task_name=debug_sample \
        trainer.max_steps=1 \
        trainer.max_epochs=1 \
        model.train_ds.batch_size=2 \
        model.validation_ds.batch_size=2 \
        model.test_ds.batch_size=2 \
        model.nemo_path=null \
        trainer.val_check_interval=0.0 \
        trainer.gpus=[0,1] \
        model.dataset.use_cache=false \
        model.language_model.pretrained_model_name=bert-base-cased \
        trainer.accelerator=ddp \
        exp_manager=null  && \
        rm -rf sgd_outputs'
          }
        }
        stage('GLUE STS-b with AlBERT') {
          steps {
            sh 'python examples/nlp/glue_benchmark/glue_benchmark.py \
            model.dataset.use_cache=false \
            model.task_name=sts-b \
            model.dataset.data_dir=/home/TestData/nlp/glue_fake/STS-B \
            trainer.gpus=[1] \
            +trainer.fast_dev_run=True \
            model.language_model.pretrained_model_name=albert-base-v1 \
            exp_manager=null'
          }
        }
      }
    }

    stage('L2: Parallel BERT SQUAD v1.1 / v2.0') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
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
            model.test_ds.file=/home/TestData/nlp/squad_mini/v1.1/dev-v1.1.json \
            model.train_ds.batch_size=2 \
            model.train_ds.num_samples=2 \
            model.validation_ds.batch_size=2 \
            model.validation_ds.num_samples=2 \
            model.test_ds.num_samples=2 \
            model.test_ds.batch_size=2 \
            trainer.max_epochs=1 \
            +trainer.max_steps=1 \
            model.language_model.pretrained_model_name=bert-base-uncased \
            model.dataset.version_2_with_negative=false \
            trainer.precision=16 \
            trainer.amp_level=O1 \
            trainer.gpus=[0] \
            exp_manager=null'
          }
        }
        stage('BERT SQUAD 2.0') {
          // Cannot do fast_dev_run because squad needs whole dev dataset
          steps {
            sh 'cd examples/nlp/question_answering && \
            python question_answering_squad.py \
            model.train_ds.file=/home/TestData/nlp/squad_mini/v2.0/train-v2.0.json \
            model.dataset.use_cache=false \
            model.train_ds.batch_size=2 \
            model.train_ds.num_samples=2 \
            model.validation_ds.batch_size=2 \
            model.validation_ds.num_samples=2 \
            trainer.max_epochs=1 \
            +trainer.max_steps=1 \
            model.validation_ds.file=/home/TestData/nlp/squad_mini/v2.0/dev-v2.0.json \
            model.language_model.pretrained_model_name=bert-base-uncased \
            model.dataset.version_2_with_negative=true \
            trainer.precision=16 \
            trainer.amp_level=O1 \
            trainer.gpus=[1] \
            exp_manager=null'
          }
        }
      }
    }
    // Runs out of memory on the 12G TITAN V (GPU 0 on main CI)
    stage('L2: MegaBERT Token Classification') {
      when {
        anyOf {
          branch 'v1.0.0b2'
          changeRequest target: 'v1.0.0b2'
        }
      }
      failFast true
      steps {
        sh 'cd examples/nlp/token_classification && \
        python token_classification_train.py \
        model.dataset.data_dir=/home/TestData/nlp/token_classification_punctuation/ \
        model.language_model.pretrained_model_name=megatron-bert-345m-uncased \
        model.train_ds.batch_size=10 \
        model.dataset.max_seq_length=50 \
        model.dataset.use_cache=false \
        trainer.accelerator=ddp \
        trainer.precision=16 \
        trainer.amp_level=O1 \
        trainer.gpus=[1] \
        +trainer.fast_dev_run=true \
        exp_manager=null'
      }
    }
    stage('L2: Parallel SQUAD v1.1 & v2.0') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('SQUAD v2.0 with Megatron with ckpt & config') {
          // Cannot do fast_dev_run because squad needs whole dev dataset
          steps {
            sh 'cd examples/nlp/question_answering && \
            python question_answering_squad.py \
            model.train_ds.file=/home/TestData/nlp/squad_mini/v2.0/train-v2.0.json \
            model.dataset.use_cache=false \
            model.train_ds.batch_size=1 \
            model.train_ds.num_samples=1 \
            model.validation_ds.batch_size=1 \
            model.validation_ds.num_samples=1 \
            trainer.accelerator=ddp \
            trainer.max_epochs=1 \
            +trainer.max_steps=1 \
            model.validation_ds.file=/home/TestData/nlp/squad_mini/v2.0/dev-v2.0.json \
            model.language_model.pretrained_model_name=megatron-bert-uncased  \
            model.language_model.lm_checkpoint=/home/TestData/nlp/megatron_345m_uncased/model_optim_rng.pt \
            model.language_model.config_file=/home/TestData/nlp/megatron_345m_uncased/345m_config.json \
            model.dataset.version_2_with_negative=true \
            trainer.precision=16 \
            trainer.amp_level=O1 \
            trainer.gpus=[1] \
            exp_manager=null'
          }
        }
        stage('RoBERTa SQUAD 1.1') {
          // Cannot do fast_dev_run because squad needs whole dev dataset
          steps {
            sh 'cd examples/nlp/question_answering && \
            python question_answering_squad.py \
            model.train_ds.file=/home/TestData/nlp/squad_mini/v1.1/train-v1.1.json \
            model.dataset.use_cache=false \
            model.train_ds.batch_size=2 \
            model.train_ds.num_samples=2 \
            model.validation_ds.batch_size=2 \
            model.validation_ds.num_samples=2 \
            trainer.max_epochs=1 \
            +trainer.max_steps=1 \
            model.validation_ds.file=/home/TestData/nlp/squad_mini/v1.1/dev-v1.1.json \
            model.language_model.pretrained_model_name=roberta-base \
            model.dataset.version_2_with_negative=false \
            trainer.precision=16 \
            trainer.amp_level=O1 \
            trainer.gpus=[0] \
            exp_manager=null'
          }
        }
        stage ('Text Classification with BERT Test') {
          steps {
            sh 'cd examples/nlp/text_classification && \
            python text_classification_with_bert.py \
            model.dataset.num_classes=6 \
            model.train_ds.file_path=/home/TestData/nlp/retail_text_classification/train.tsv \
            model.validation_ds.file_path=/home/TestData/nlp/retail_text_classification/dev.tsv \
            model.language_model.pretrained_model_name=bert-base-uncased \
            model.train_ds.batch_size=10 \
            model.dataset.max_seq_length=50 \
            model.dataset.use_cache=false \
            trainer.gpus=[0] \
            +trainer.fast_dev_run=true \
            exp_manager=null'
          }
        }
        stage('L2: Intent and Slot Classification') {
          steps {
            sh 'cd examples/nlp/intent_slot_classification && \
            python intent_slot_classification.py \
            model.data_dir=/home/TestData/nlp/retail \
            model.validation_ds.prefix=dev \
            model.test_ds.prefix=dev \
            trainer.gpus=[0] \
            +trainer.fast_dev_run=true \
            exp_manager.exp_dir=checkpoints'
            sh 'rm -rf checkpoints'
          }
        }
      }
    }
    // TODO: fix model parallel for PTL 1.2
    // stage('L2: Model Parallel Size 2 Megatron Text Classification') {
    //   when {
    //     anyOf{
    //       branch 'main'
    //       changeRequest target: 'main'
    //     }
    //   }
    //   failFast true
    //   steps{
    //     sh 'cd examples/nlp/text_classification && \
    //     python text_classification_with_bert.py \
    //     exp_manager.create_checkpoint_callback=false \
    //     exp_manager.exp_dir=exp_mp_2_megatron_bert \
    //     trainer.gpus=[0,1] \
    //     trainer.num_nodes=1 \
    //     trainer.precision=16 \
    //     trainer.gradient_clip_val=1.0 \
    //     ~trainer.amp_level \
    //     +trainer.replace_sampler_ddp=false \
    //     +trainer.fast_dev_run=true \
    //     model.dataset.num_classes=6 \
    //     model.train_ds.file_path=/home/TestData/nlp/retail_text_classification/train.tsv \
    //     model.train_ds.batch_size=4 \
    //     model.language_model.pretrained_model_name=megatron-bert-uncased \
    //     model.language_model.config_file=/home/TestData/nlp/mp_2_bert_toy/config.json \
    //     model.language_model.lm_checkpoint=/home/TestData/nlp/mp_2_bert_toy/iter_2000000 \
    //     model.nemo_path=null \
    //     ~model.infer_samples \
    //     exp_manager=null'
    //   }
    // }

    stage('L2: Parallel NLP Examples 2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage ('NER finetuning from pretrained Test') {
          steps {
            sh 'cd examples/nlp/token_classification && \
            python token_classification_train.py \
            pretrained_model=ner_en_bert \
            model.dataset.data_dir=/home/TestData/nlp/ner/ \
            model.train_ds.batch_size=2 \
            model.dataset.use_cache=false \
            trainer.gpus=[0] \
            +trainer.fast_dev_run=true \
            model.dataset.class_balancing="weighted_loss" \
            exp_manager.exp_dir=null'
          }
        }
        stage ('Punctuation and capitalization finetuning from pretrained test') {
          steps {
            sh 'cd examples/nlp/token_classification && \
            python punctuation_capitalization_train.py \
            pretrained_model=punctuation_en_bert \
            model.dataset.data_dir=/home/TestData/nlp/token_classification_punctuation/ \
            trainer.gpus=[1] \
            +trainer.fast_dev_run=true \
            model.dataset.use_cache=false \
            exp_manager.exp_dir=null'
          }
        }
        stage ('NER with TurkuNLP/bert-base-finnish-cased-v1') {
          steps {
            sh 'cd examples/nlp/token_classification && \
            python token_classification_train.py \
            model.dataset.data_dir=/home/TestData/nlp/token_classification_punctuation/ \
            trainer.gpus=[0] \
            +trainer.fast_dev_run=true \
            model.dataset.use_cache=false \
            model.language_model.pretrained_model_name="TurkuNLP/bert-base-finnish-cased-v1" \
            exp_manager.exp_dir=null'
          }
        }
        stage('Evaluation script for Token Classification') {
          steps {
            sh 'python examples/nlp/token_classification/token_classification_evaluate.py \
            model.dataset.data_dir=/home/TestData/nlp/ner/ \
            pretrained_model=/home/TestData/nlp/pretrained_models/NER_Model_with_BERT_base_uncased.nemo && \
            rm -rf nemo_experiments'
          }
        }
        stage('Evaluation script for Punctuation') {
          steps {
            sh 'python examples/nlp/token_classification/punctuation_capitalization_evaluate.py \
            model.dataset.data_dir=/home/TestData/nlp/token_classification_punctuation/ \
            pretrained_model=/home/TestData/nlp/pretrained_models/Punctuation_Capitalization_with_DistilBERT_base_uncased.nemo && \
            rm -rf nemo_experiments'
          }
        }
        stage('L2: Punctuation & Capitalization, 2GPUs with DistilBERT') {
          steps {
            sh 'cd examples/nlp/token_classification && \
            python punctuation_capitalization_train.py \
            model.dataset.data_dir=/home/TestData/nlp/token_classification_punctuation/ \
            model.language_model.pretrained_model_name=distilbert-base-uncased \
            model.dataset.use_cache=false \
            trainer.gpus=[0,1] \
            trainer.accelerator=ddp \
            +trainer.fast_dev_run=true \
            exp_manager=null'
          }
        }
      }
    }

    stage('L2: Parallel Pretraining BERT pretraining from Text/Preprocessed') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
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
              model.train_ds.batch_size=32 \
              model.validation_ds.data_file=/home/TestData/nlp/wikitext-2/valid.txt  \
              model.validation_ds.batch_size=32 \
              model.language_model.config_file=/home/TestData/nlp/bert_configs/bert_3200.json \
              model.optim.lr=0.01 \
              model.optim.sched.warmup_ratio=0.1 \
              model.tokenizer.tokenizer_name=sentencepiece \
              model.tokenizer.tokenizer_model=/home/TestData/nlp/wikitext-2/tokenizer_bpe_v3193/tokenizer.model \
              model.mask_prob=0.15 \
              model.short_seq_prob=0.1 \
              exp_manager.exp_dir=PretrainingBERTFromText \
              '
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
              model.language_model.lm_checkpoint=/home/TestData/nlp/bert_ckpts/nemo1.0/bert_base_uncased_mlm_final_1074591_nemo1.0.pt \
              model.language_model.config_file=/home/TestData/nlp/bert_configs/uncased_L-12_H-768_A-12.json \
              model.optim.lr=0.875e-4 \
              model.optim.weight_decay=0.01 \
              model.optim.sched.warmup_ratio=0.01 \
              exp_manager.exp_dir=PretrainingBERTFromPreprocessed \
              exp_manager.create_checkpoint_callback=False \
              '
              sh 'rm -rf examples/nlp/language_modeling/PretrainingBERTFromPreprocessed'
              sh 'ls -lha examples/nlp/language_modeling'
            }
        }
        stage('L2: Pretraining BERT pretraining from Text with char tokenizer') {
            steps {
              sh 'cd examples/nlp/language_modeling && \
              python bert_pretraining.py \
              --config-name=bert_pretraining_from_text_config.yaml \
              trainer.gpus=[0] \
              trainer.precision=16 \
              trainer.amp_level=O1 \
              +trainer.fast_dev_run=true \
              model.train_ds.data_file=/home/TestData/nlp/wikitext-2/train.txt  \
              model.train_ds.batch_size=32 \
              model.validation_ds.data_file=/home/TestData/nlp/wikitext-2/valid.txt  \
              model.validation_ds.batch_size=32 \
              model.language_model.config_file=/home/TestData/nlp/bert_configs/bert_3200.json \
              model.optim.lr=0.01 \
              model.optim.sched.warmup_ratio=0.1 \
              model.tokenizer.tokenizer_name=char \
              model.tokenizer.vocab_file=/home/TestData/nlp/vocabs/mini_vocab.txt \
              model.mask_prob=0.15 \
              model.short_seq_prob=0.1 \
              exp_manager.exp_dir=PretrainingBERTFromTextchartok \
              '
              sh 'rm -rf examples/nlp/language_modeling/PretrainingBERTFromTextchartok'
            }
        }
        stage('L2: Pretraining BERT pretraining from Text with word tokenizer') {
            steps {
              sh 'cd examples/nlp/language_modeling && \
              python bert_pretraining.py \
              --config-name=bert_pretraining_from_text_config.yaml \
              trainer.gpus=[1] \
              trainer.precision=16 \
              trainer.amp_level=O1 \
              +trainer.fast_dev_run=true \
              model.train_ds.data_file=/home/TestData/nlp/wikitext-2/train.txt  \
              model.train_ds.batch_size=32 \
              model.validation_ds.data_file=/home/TestData/nlp/wikitext-2/valid.txt  \
              model.validation_ds.batch_size=32 \
              model.language_model.config_file=/home/TestData/nlp/bert_configs/bert_3200.json \
              model.optim.lr=0.01 \
              model.optim.sched.warmup_ratio=0.1 \
              model.tokenizer.tokenizer_name=word \
              model.tokenizer.vocab_file=/home/TestData/nlp/vocabs/mini_vocab.txt \
              model.mask_prob=0.15 \
              model.short_seq_prob=0.1 \
              exp_manager.exp_dir=PretrainingBERTFromTextwordtok \
              '
              sh 'rm -rf examples/nlp/language_modeling/PretrainingBERTFromTextwordtok'
            }
        }
      }
    }

    stage('L2: NMT Attention is All You Need') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('L2: NMT Training Post-LN') {
            steps {
              sh 'cd examples/nlp/machine_translation && \
              python enc_dec_nmt.py \
              --config-path=conf \
              --config-name=aayn_base \
              model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
              model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.test_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.test_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.encoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/tt_tokenizer.BPE.4096.model \
              model.decoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/tt_tokenizer.BPE.4096.model \
              trainer.gpus=[0] \
              +trainer.fast_dev_run=true \
              exp_manager=null \
              '
            }
        }

        stage('L2: NMT Training Pre-LN') {
            steps {
              sh 'cd examples/nlp/machine_translation && \
              python enc_dec_nmt.py \
              --config-path=conf \
              --config-name=aayn_base \
              model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
              model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.test_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.test_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.encoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/tt_tokenizer.BPE.4096.model \
              model.decoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/tt_tokenizer.BPE.4096.model \
              model.encoder.pre_ln=true \
              model.decoder.pre_ln=true \
              trainer.gpus=[1] \
              +trainer.fast_dev_run=true \
              exp_manager=null \
              '
            }
        }

        stage('L2: NMT Inference') {
            steps {
              sh 'cd examples/nlp/machine_translation && \
              python nmt_transformer_infer.py \
              --model=/home/TestData/nlp/nmt/toy_data/TransformerLargeDe-En.nemo \
              --srctext=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.test.src \
              --tgtout=/home/TestData/nlp/nmt/toy_data/out.txt \
              --target_lang en \
              --source_lang de \
              '
            }
        }
      }
    }

    stage('L2: NMT with HuggingFace') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('L2: NMT Pretrained HF Encoder') {
            steps {
              sh 'cd examples/nlp/machine_translation && \
              python enc_dec_nmt.py \
              --config-path=conf \
              --config-name=huggingface \
              model.shared_tokenizer=False \
              model.encoder_tokenizer.library=huggingface \
              model.encoder.library=huggingface \
              model.encoder.model_name=bert-base-cased \
              model.encoder.pretrained=true \
              model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
              model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.test_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.test_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.decoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/tt_tokenizer.BPE.4096.model \
              trainer.gpus=[0] \
              +trainer.fast_dev_run=true \
              exp_manager=null \
              '
            }
        }

        stage('L2: NMT Custom HF Encoder') {
            steps {
              sh 'cd examples/nlp/machine_translation && \
              python enc_dec_nmt.py \
              --config-path=conf \
              --config-name=huggingface \
              model.shared_tokenizer=True \
              model.encoder_tokenizer.library=yttm \
              model.encoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/tt_tokenizer.BPE.4096.model \
              model.encoder.library=huggingface \
              model.encoder.model_name=null \
              model.encoder.pretrained=false \
              +model.encoder._target_=transformers.BertConfig \
              +model.encoder.hidden_size=1536 \
              model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
              model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.test_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.test_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.decoder_tokenizer.tokenizer_model=/home/TestData/nlp/nmt/toy_data/tt_tokenizer.BPE.4096.model \
              trainer.gpus=[1] \
              +trainer.fast_dev_run=true \
              exp_manager=null \
              '
            }
        }
      }
    }

    stage('L2: NMT Tarred Dataset Creation') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('L2: NMT Auto Tarred Dataset Creation') {
            steps {
              sh 'cd examples/nlp/machine_translation && \
              python enc_dec_nmt.py \
              --config-path=conf \
              --config-name=aayn_base \
              do_training=false \
              model.preproc_out_dir=$PWD/preproc_out_dir \
              model.train_ds.use_tarred_dataset=true \
              model.train_ds.n_preproc_jobs=2 \
              model.train_ds.lines_per_dataset_fragment=500 \
              model.train_ds.num_batches_per_tarfile=10 \
              model.train_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.train_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
              model.validation_ds.src_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.validation_ds.tgt_file_name=/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              model.encoder_tokenizer.vocab_size=2000 \
              model.decoder_tokenizer.vocab_size=2000 \
              ~model.test_ds \
              trainer.gpus=[0] \
              +trainer.fast_dev_run=true \
              exp_manager=null \
              '
            }
        }

        stage('L2: NMT Script Tarred Dataset Creation') {
            steps {
              sh 'cd examples/nlp/machine_translation && \
              python create_tarred_parallel_dataset.py \
              --src_fname /home/TestData/nlp/nmt/toy_data/wmt14-de-en.src \
              --tgt_fname /home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref \
              --out_dir $PWD/preproc_out_dir \
              --encoder_tokenizer_vocab_size=2000 \
              --decoder_tokenizer_vocab_size=2000 \
              --tokens_in_batch=1000 \
              --lines_per_dataset_fragment=500 \
              --num_batches_per_tarfile=10 \
              --n_preproc_jobs=2 \
              '
            }
        }
      }
    }

    stage('L2: TTS Fast dev runs 1') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
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
            trainer.accelerator=null \
            trainer.max_epochs=-1 \
            model.train_ds.dataloader_params.batch_size=12 \
            model.validation_ds.dataloader_params.batch_size=12 \
            ~trainer.check_val_every_n_epoch'
          }
        }
        // stage('FastPitch') {
        //   steps {
        //     sh 'python examples/tts/fastpitch.py \
        //     train_dataset=/home/TestData/an4_dataset/an4_train.json \
        //     validation_datasets=/home/TestData/an4_dataset/an4_val.json \
        //     trainer.gpus="[0]" \
        //     +trainer.fast_dev_run=True \
        //     trainer.accelerator=null \
        //     trainer.max_epochs=-1 \
        //     model.train_ds.batch_size=12 \
        //     model.train_ds.num_workers=1 \
        //     model.validation_ds.batch_size=12 \
        //     model.validation_ds.num_workers=1 \
        //     ~trainer.check_val_every_n_epoch'
        //   }
        // }
        stage('WaveGlow') {
          steps {
            sh 'python examples/tts/waveglow.py \
            train_dataset=/home/TestData/an4_dataset/an4_train.json \
            validation_datasets=/home/TestData/an4_dataset/an4_val.json \
            trainer.gpus="[1]" \
            +trainer.fast_dev_run=True \
            trainer.accelerator=null \
            trainer.max_epochs=-1 \
            model.train_ds.dataloader_params.batch_size=4 \
            model.validation_ds.dataloader_params.batch_size=4 \
            ~trainer.check_val_every_n_epoch'
          }
        }
      }
    }

    stage('L2: TTS Fast dev runs 2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }

      parallel {
        stage('MelGAN') {
          steps {
            sh 'python examples/tts/melgan.py \
            train_dataset=/home/TestData/an4_dataset/an4_train.json \
            validation_datasets=/home/TestData/an4_dataset/an4_val.json \
            trainer.gpus="[0]" \
            +trainer.fast_dev_run=True \
            trainer.accelerator=ddp \
            trainer.max_epochs=-1 \
            model.train_ds.dataloader_params.batch_size=4 \
            model.validation_ds.dataloader_params.batch_size=4 \
            ~trainer.check_val_every_n_epoch'
          }
        }
        stage('SqueezeWave') {
          steps {
            sh 'python examples/tts/squeezewave.py \
            train_dataset=/home/TestData/an4_dataset/an4_train.json \
            validation_datasets=/home/TestData/an4_dataset/an4_val.json \
            trainer.gpus="[0]" \
            +trainer.fast_dev_run=True \
            trainer.accelerator=null \
            trainer.max_epochs=-1 \
            model.train_ds.dataloader_params.batch_size=4 \
            model.validation_ds.dataloader_params.batch_size=4 \
            ~trainer.check_val_every_n_epoch'
          }
        }
        stage('GlowTTS') {
          steps {
            sh 'python examples/tts/glow_tts.py \
            train_dataset=/home/TestData/an4_dataset/an4_train.json \
            validation_datasets=/home/TestData/an4_dataset/an4_val.json \
            trainer.gpus="[1]" \
            +trainer.fast_dev_run=True \
            trainer.accelerator=null \
            trainer.max_epochs=-1 \
            model.train_ds.batch_size=4 \
            model.validation_ds.batch_size=4 \
            ~trainer.check_val_every_n_epoch'
          }
        }
      }
    }

    stage('L??: Speech Checkpoints tests') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('QuartzNet15x5Base-En') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES=0 python examples/asr/speech_to_text_infer.py --asr_model QuartzNet15x5Base-En --dataset /home/TestData/librispeech/librivox-dev-other.json --wer_tolerance 0.1012 --batch_size 64'
          }
        }
        stage('Tacotron2_WaveGlow_Jasper') {
          steps {
            sh 'CUDA_VISIBLE_DEVICES=1 python examples/tts/test_tts_infer.py --wer_tolerance 0.25 --debug --trim'
          }
        }
      }
    }
  }

  post {
    always {
      sh 'chmod -R 777 .'
      cleanWs()
    }
  }
}
