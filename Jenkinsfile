def pytorch_container = 'nvcr.io/nvidia/pytorch:21.02-py3'

pipeline {
  agent none
  options {
    timeout(time: 1, unit: 'HOURS')
    disableConcurrentBuilds()
  }
  stages {
    // stage('PyTorch Container Setup') {
    //   agent {
    //     docker {
    //       image "${pytorch_container}"
    //       args '--device=/dev/nvidia0 --gpus all --user 0:128 -v /home/TestData:/home/TestData -v $HOME/.cache/torch:/root/.cache/torch --shm-size=8g'
    //       // args '--device=/dev/nvidia0 --gpus all -v /home/TestData:/home/TestData -v $HOME/.cache/torch:/root/.cache/torch --shm-size=8g'
    //     }
    //   }
    //   stages {
    //     stage('Install test requirements') {
    //       steps {
    //         sh 'apt-get update'
    //         sh 'apt-get install -y bc'
    //         sh 'pip install -r requirements/requirements_test.txt'
    //       }
    //     }
    //   }
    // }
    stage('PyTorch Container') {
      environment {
        UID = sh (returnStdout: true, script: 'id -u').trim()
        GID = sh (returnStdout: true, script: 'id -g').trim()
      }
      agent {
        docker {
          image "${pytorch_container}"
          // args '--device=/dev/nvidia0 --gpus all --user 0:128 -v /home/TestData:/home/TestData -v $HOME/.cache/torch:/root/.cache/torch --shm-size=8g'
          args '--device=/dev/nvidia0 --gpus all --group-add 999 --mount type=tmpfs,destination=/.cache --mount type=tmpfs,destination=/.local -v /home/TestData:/home/TestData -v $HOME/.cache/torch:/root/.cache/torch -v /var/run/docker.sock:/var/run/docker.sock -v /usr/bin/docker:/usr/bin/docker --shm-size=8g'
        }
      }
      stages {
        stage('Install test requirements') {
          steps {
            // sh 'ls -ltrh /var/run/docker.sock'
            // sh 'id'
            // sh 'env'
            // sh 'docker ps -q'
            sh 'docker exec --user 0:128 $(docker ps -q) apt-get update'
            sh 'docker exec --user 0:128 $(docker ps -q) apt-get install -y bc'
            sh 'docker exec --user 0:128 $(docker ps -q) pip install -r requirements/requirements_test.txt'
            sh 'docker exec --user 0:128 $(docker ps -q) chmod 777 -R /opt/conda/bin'
            sh 'docker exec --user 0:128 $(docker ps -q) chmod 777 /'
            // script {
            //   docker.image("${pytorch_container}").inside("--user 0:128") {
            //     sh 'apt-get update'
            //     sh 'apt-get install -y bc'
            //     sh 'pip install -r requirements/requirements_test.txt'
            //     sh 'ls /opt/conda/bin | grep isort'
            //   }
            // }
            // sh 'ls /opt/conda/bin | grep isort'
          }
        }
        stage('PyTorch version') {
          steps {
            sh 'python -c "import torch; print(torch.__version__)"'
            sh 'python -c "import torchtext; print(torchtext.__version__)"'
            sh 'python -c "import torchvision; print(torchvision.__version__)"'
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
            sh 'pytest -m "unit and not skipduringci and not pleasefixme"'
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
            sh 'CUDA_VISIBLE_DEVICES="" pytest -m "unit and not pleasefixme" --cpu'
          }
        }

        stage('pynini export') {
          steps {
            sh 'cd tools/text_denormalization && python pynini_export.py /home/TestData/nlp/text_denorm/output/ && ls -R /home/TestData/nlp/text_denorm/output/ && echo ".far files created "|| exit 1'
            sh 'cd tools/text_denormalization && cp *.grm /home/TestData/nlp/text_denorm/output/'
            sh 'ls -R /home/TestData/nlp/text_denorm/output/'
            sh 'cd nemo_tools/text_denormalization/ &&  python run_predict.py --input=/home/TestData/nlp/text_denorm/ci/test.txt --output=/home/TestData/nlp/text_denorm/output/test.pynini.txt --verbose'
            sh 'cmp --silent /home/TestData/nlp/text_denorm/output/test.pynini.txt /home/TestData/nlp/text_denorm/ci/test_goal_pynini.txt || exit 1'
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
      }
    }

    stage('Text denorm') {
      stages {
        stage('sparrowhawk test') {
          agent {
                docker {
                  image 'gitlab-master.nvidia.com:5005/yangzhang/text_normalization/sparrowhawk:latest'
                  args '--user 0:128 -v /home/TestData:/home/TestData --shm-size=8g'
                }
          }
          options {
            skipDefaultCheckout true
          }
          steps {
            sh 'cd /home/TestData/nlp/text_denorm/ci/ && bash setup_sparrowhawk.sh /home/TestData/nlp/text_denorm/output/ || exit 2'
            sh 'cd /work_dir/sparrowhawk/documentation/grammars && normalizer_main --config=sparrowhawk_configuration.ascii_proto --multi_line_text < /home/TestData/nlp/text_denorm/ci/test.txt > /home/TestData/nlp/text_denorm/output/test.sparrowhawk.txt'
            sh 'cmp --silent /home/TestData/nlp/text_denorm/output/test.sparrowhawk.txt /home/TestData/nlp/text_denorm/ci/test_goal_sh.txt || exit 1'
            sh 'rm -rf /home/TestData/nlp/text_denorm/output/*'
          }
        }
      }
    }
  }

  post {
    always {
      node(null) {
        script {
          docker.image("${pytorch_container}").inside("""--user 0:128 --entrypoint=''""") {
            sh 'chmod -R 777 .'
            cleanWs()
          }
        }
      }
    }
  }
}
