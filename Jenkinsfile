pipeline {
  agent any
  environment {
      PATH="/home/mrjenkins/anaconda3/envs/py36p1.3.1c10/bin:$PATH"
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
    stage('PEP8 Checks') {
      steps {
        sh 'pycodestyle . --exclude=./scripts/get_librispeech_data.py,./scripts/process_beam_dump.py,./tests/other/jasper.py,./tests/other/jasper_zero_dl.py,./collections/nemo_nlp/nemo_nlp/utils/metrics/sacrebleu.py,./collections/nemo_nlp/nemo_nlp/utils/metrics/fairseq_tokenizer.py,./nemo/setup.py,./docs/sources/source/conf.py,./docs/sources/source/tutorials/infer.py,./docs/sources/source/tutorials/test.py,./collections/nemo_nlp/build'
      }
    } 

    stage('Unittests') {
      steps {
        sh './reinstall.sh && python -m unittest tests/*.py'
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
            sh 'cd examples/nlp && CUDA_VISIBLE_DEVICES=0 python nmt_tutorial.py'
          }
        }
      }
    }

    stage('Parallel Stage2') {
      failFast true
      parallel {
        stage('Jasper AN4 O1') {
          steps {
            sh 'cd examples/asr && CUDA_VISIBLE_DEVICES=0 python jasper_an4.py --amp_opt_level=O1 --num_epochs=40'
          }
        }
        stage('Jasper AN4 O2') {
          steps {
            sh 'cd examples/asr && CUDA_VISIBLE_DEVICES=1 python jasper_an4.py --amp_opt_level=O2 --num_epochs=40'
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
            sh 'cd examples/image && CUDA_VISIBLE_DEVICES=0 python gan.py --amp_opt_level=O2 --num_epochs=3'
          }
        }
      }
    }

    stage('Multi-GPU test') {
      failFast true
      parallel {
        stage('Jasper AN4 2 GPUs') {
          steps {
            sh 'cd examples/asr && CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2  jasper_an4.py --amp_opt_level=O2 --num_epochs=40'
          }
        }
      }
    }
  }
  post {
    always {
        cleanWs()
    }
  }
}
