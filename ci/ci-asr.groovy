@Library('blossom-github-lib@master') 
import ipp.blossom.*

podTemplate(cloud:'sc-ipp-blossom-prod', yaml : """
apiVersion: v1
kind: Pod
metadata:
  labels:
    some-label: some-label-value
spec:
  volumes:
  - name: scratch
    nfs:
      server: ipp1-cdot01-col01
      path: /vol/scratch1/scratch.okuchaiev_blossom
  containers:
  - name: latestdlfw
    image: nvcr.io/nvidia/pytorch:22.09-py3
    command:
    - cat
    volumeMounts:
    - name: scratch
      mountPath: /testdata
    resources:
          limits:
             nvidia.com/gpu: 2
    restartPolicy: Never
    backoffLimit: 4
    tty: true
    shm-size: 32g
  nodeSelector:
    kubernetes.io/os: linux
    nvidia.com/gpu_type: "Tesla_T4x4"
    nvidia.com/node_type: gpu_tester
    nvidia.com/driver_version: "510.20"
"""
)   {
      node(POD_LABEL) {
          def githubHelper
          stage('Get Token') {
              withCredentials([usernamePassword(credentialsId: 'GHAtoken', passwordVariable: 'GIT_PASSWORD', usernameVariable: 'GIT_USERNAME')]) {
                  // create new instance of helper object
                  githubHelper = GithubHelper.getInstance("${GIT_PASSWORD}", githubData)
              }
              
          }
          def stageName = '' 
          try {
              currentBuild.description = githubHelper.getBuildDescription()
              container('latestdlfw') {
                stage('Code checkout') {
                    // update status on github
                    githubHelper.updateCommitStatus("$BUILD_URL", "$stageName Running", GitHubCommitState.PENDING)
                    checkout changelog: true, poll: true, scm: [$class: 'GitSCM', branches: [[name: "pr/"+githubHelper.getPRNumber()]],
                    doGenerateSubmoduleConfigurations: false,
                    submoduleCfg: [],
                    userRemoteConfigs: [[credentialsId: 'github-token', url: githubHelper.getCloneUrl(), refspec: '+refs/pull/'+githubHelper.getPRNumber()+'/head:refs/remotes/origin/pr/'+githubHelper.getPRNumber()]]]
                }

                stage('Code Requirements') {
                        sh "apt-get update && \
                            apt-get install -y bc && \
                            pip install -r requirements/requirements_test.txt
                }

                stage('Installation') {
                  sh "git config --global --add safe.directory '*' && \
                      nvidia-smi && \
                      ./reinstall.sh release"
                }

                parallel( //USE CUDA_VISIBLE_DEVICES to execute 2 single GPU tests in parallel here
                [
                    "L2: Speech to Text": {
                        sh 'CUDA_VISIBLE_DEVICES=0 python examples/asr/asr_ctc/speech_to_text_ctc.py \
                            model.train_ds.manifest_filepath=/testdata/TestData/an4_dataset/an4_train.json \
                            model.validation_ds.manifest_filepath=/testdata/TestData/an4_dataset/an4_val.json \
                            trainer.devices=[0] \
                            trainer.accelerator="gpu" \
                            +trainer.fast_dev_run=True \
                            exp_manager.exp_dir=examples/asr/speech_to_text_results'
                        sh 'rm -rf examples/asr/speech_to_text_results'
                    },
                    "L2: Speech to Text EMA": {
                        sh 'CUDA_VISIBLE_DEVICES=0 python examples/asr/asr_ctc/speech_to_text_ctc.py \
                            model.train_ds.manifest_filepath=/testdata/TestData/an4_dataset/an4_train.json \
                            model.validation_ds.manifest_filepath=/testdata/TestData/an4_dataset/an4_val.json \
                            trainer.devices=[0] \
                            trainer.accelerator="gpu" \
                            +trainer.fast_dev_run=True \
                            +exp_manager.ema.enable=True \
                            exp_manager.exp_dir=examples/asr/speech_to_text_results'
                        sh 'rm -rf examples/asr/speech_to_text_results'
                    },
                    "L2: Speech to Text WPE - CitriNet": {
                        sh 'CUDA_VISIBLE_DEVICES=1 python examples/asr/asr_ctc/speech_to_text_ctc_bpe.py \
                            --config-path="../conf/citrinet/" --config-name="config_bpe" \
                            model.train_ds.manifest_filepath=/testdata/TestData/an4_dataset/an4_train.json \
                            model.validation_ds.manifest_filepath=/testdata/TestData/an4_dataset/an4_val.json \
                            model.tokenizer.dir="/testdata/TestData/asr_tokenizers/an4_wpe_128/" \
                            model.tokenizer.type="wpe" \
                            trainer.devices=[0] \
                            trainer.accelerator="gpu" \
                            +trainer.fast_dev_run=True \
                            exp_manager.exp_dir=examples/asr/speech_to_text_wpe_results'
                        sh 'rm -rf examples/asr/speech_to_text_wpe_results'
                    },
                    "L2: Speech Pre-training - CitriNet": {
                        sh 'CUDA_VISIBLE_DEVICES=1 python examples/asr/speech_pretraining/speech_pre_training.py \
                            --config-path="../conf/ssl/citrinet/" --config-name="citrinet_ssl_ci" \
                            model.train_ds.manifest_filepath=/testdata/TestData/an4_dataset/an4_train.json \
                            model.validation_ds.manifest_filepath=/testdata/TestData/an4_dataset/an4_val.json \
                            trainer.devices=[0] \
                            trainer.accelerator="gpu" \
                            +trainer.fast_dev_run=True \
                            exp_manager.exp_dir=examples/asr/speech_pre_training_results'
                        sh 'rm -rf examples/asr/speech_pre_training_results'
                    },
                    "L2: Speech Pre-training - Wav2Vec": {
                        sh 'CUDA_VISIBLE_DEVICES=1 python examples/asr/speech_pretraining/speech_pre_training.py \
                            --config-path="../conf/ssl/wav2vec/" --config-name="wav2vec_ci" \
                            model.train_ds.manifest_filepath=/testdata/TestData/an4_dataset/an4_train.json \
                            model.validation_ds.manifest_filepath=/testdata/TestData/an4_dataset/an4_val.json \
                            trainer.devices=[0] \
                            trainer.accelerator="gpu" \
                            +trainer.fast_dev_run=True \
                            exp_manager.exp_dir=examples/asr/speech_pre_training_results'
                        sh 'rm -rf examples/asr/speech_pre_training_results'
                    },
                    "L2: Speech to Text WPE - Conformer": {
                        sh 'CUDA_VISIBLE_DEVICES=0 python examples/asr/asr_ctc/speech_to_text_ctc_bpe.py \
                            --config-path="../conf/conformer" --config-name="conformer_ctc_bpe" \
                            model.train_ds.manifest_filepath=/testdata/TestData/an4_dataset/an4_train.json \
                            model.validation_ds.manifest_filepath=/testdata/TestData/an4_dataset/an4_val.json \
                            model.tokenizer.dir="/testdata/TestData/asr_tokenizers/an4_wpe_128/" \
                            model.tokenizer.type="wpe" \
                            model.train_ds.batch_size=4 \
                            model.validation_ds.batch_size=4 \
                            trainer.devices=[0] \
                            trainer.accelerator="gpu" \
                            +trainer.fast_dev_run=True \
                            exp_manager.exp_dir=examples/asr/speech_to_text_wpe_conformer_results'
                        sh 'rm -rf examples/asr/speech_to_text_wpe_conformer_results'
                    }
                ]
                )//end of parallel

                parallel( //USE CUDA_VISIBLE_DEVICES to execute 2 single GPU tests in parallel here
                [
                    "L2: Speech to Text WPE - Squeezeformer": {
                        sh 'CUDA_VISIBLE_DEVICES=0 python examples/asr/asr_ctc/speech_to_text_ctc_bpe.py \
                            --config-path="../conf/squeezeformer" --config-name="squeezeformer_ctc_bpe" \
                            model.train_ds.manifest_filepath=/testdata/TestData/an4_dataset/an4_train.json \
                            model.validation_ds.manifest_filepath=/testdata/TestData/an4_dataset/an4_val.json \
                            model.tokenizer.dir="/testdata/TestData/asr_tokenizers/an4_wpe_128/" \
                            model.tokenizer.type="wpe" \
                            model.encoder.d_model=144 \
                            model.train_ds.batch_size=4 \
                            model.validation_ds.batch_size=4 \
                            trainer.devices=[0] \
                            trainer.accelerator="gpu" \
                            +trainer.fast_dev_run=True \
                            exp_manager.exp_dir=examples/asr/speech_to_text_wpe_squeezeformer_results'
                        sh 'rm -rf examples/asr/speech_to_text_wpe_squeezeformer_results'
                    }
                ]
                )//end of parallel


              }
              githubHelper.updateCommitStatus("$BUILD_URL", "Complete", GitHubCommitState.SUCCESS)
          }
          catch (Exception ex){
              currentBuild.result = 'FAILURE'
              println ex
              githubHelper.updateCommitStatus("$BUILD_URL", "$stageName Failed", GitHubCommitState.FAILURE)
          }
          
      }
  }