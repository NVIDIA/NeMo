pipeline {
  agent any
  environment {
      PATH="/home/mrjenkins/anaconda3/envs/py37p1.4.0/bin:$PATH"
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
        sh 'pip install -r requirements/requirements_test.txt'
      }
    }
    stage('Code formatting checks') {
      steps {
        sh 'python setup.py style'
      }
    }
    stage('Unittests Core') {
      steps {
        sh './reinstall.sh && python -m unittest tests/core/*.py'
      }
    }
    stage('Unittests ASR') {
      steps {
        sh 'python -m unittest tests/asr/*.py'
      }
    }
    stage('Unittests NLP') {
      steps {
        sh 'python -m unittest tests/nlp/*.py'
      }
    }
    stage('Unittests TTS') {
      steps {
        sh 'python -m unittest tests/tts/*.py'
      }
    }

    stage('Parallel Stage1') {
      failFast true
      parallel {
        stage('Simplest test') {
          steps {
            sh 'cd examples/start_here && CUDA_VISIBLE_DEVICES=0 python simplest_example.py'
          }
        }
        stage ('Chatbot test') {
          steps {
            sh 'cd examples/start_here && CUDA_VISIBLE_DEVICES=1 python chatbot_example.py'
          }
        }
        stage ('NMT test') {
          steps {
            sh 'cd examples/nlp && CUDA_VISIBLE_DEVICES=0 python machine_translation_tutorial.py'
          }
        }
      }
    }

    stage('Parallel Stage2') {
      failFast true
      parallel {
        stage('Jasper AN4 O1') {
          steps {
            sh 'cd examples/asr && CUDA_VISIBLE_DEVICES=0 python jasper_an4.py --amp_opt_level=O1 --num_epochs=35 --test_after_training --work_dir=O1'
          }
        }
        stage('Jasper AN4 O2') {
          steps {
            sh 'cd examples/asr && CUDA_VISIBLE_DEVICES=1 python jasper_an4.py --amp_opt_level=O2 --num_epochs=35 --test_after_training --work_dir=O2'
          }
        }
      }
    }

    stage('Parallel Stage3') {
      failFast true
      parallel {
        stage('GAN O1') {
          steps {
            sh 'cd examples/image && CUDA_VISIBLE_DEVICES=0 python gan.py --amp_opt_level=O1 --num_epochs=3'
          }
        }
        stage('GAN O2') {
          steps {
            sh 'cd examples/image && CUDA_VISIBLE_DEVICES=1 python gan.py --amp_opt_level=O2 --num_epochs=3'
          }
        }
      }
    }

    stage('Multi-GPU test') {
      failFast true
      parallel {
        stage('Jasper AN4 2 GPUs') {
          steps {
            sh 'cd examples/asr && CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 jasper_an4.py --num_epochs=40 --batch_size=24 --work_dir=multi_gpu --test_after_training'
          }
        }
      }
    }

    stage('TTS Tests') {
      failFast true
      steps {
        sh 'cd examples/tts && CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 tacotron2.py --max_steps=51 --model_config=configs/tacotron2.yaml --train_dataset=/home/mrjenkins/TestData/an4_dataset/an4_train.json --amp_opt_level=O1 --eval_freq=50'
        sh 'cd examples/tts && TTS_CHECKPOINT_DIR=$(ls | grep "Tacotron2") && echo $TTS_CHECKPOINT_DIR && LOSS=$(cat $TTS_CHECKPOINT_DIR/log_globalrank-0_localrank-0.txt | grep -o -E "Loss[ :0-9.]+" | grep -o -E "[0-9.]+" | tail -n 1) && echo $LOSS && if [ $(echo "$LOSS > 3.0" | bc -l) -eq 1 ]; then echo "FAILURE" && exit 1; else echo "SUCCESS"; fi'
        // sh 'cd examples/tts && TTS_CHECKPOINT_DIR=$(ls | grep "Tacotron2") && cp ../asr/multi_gpu/checkpoints/* $TTS_CHECKPOINT_DIR/checkpoints'
        // sh 'CUDA_VISIBLE_DEVICES=0 python tacotron2_an4_test.py --model_config=configs/tacotron2.yaml --eval_dataset=/home/mrjenkins/TestData/an4_dataset/an4_train.json --jasper_model_config=../asr/configs/jasper_an4.yaml --load_dir=$TTS_CHECKPOINT_DIR/checkpoints'
      }
    }

  }

  post {
    always {
        cleanWs()
    }
  }
}
