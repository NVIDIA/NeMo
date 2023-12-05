pipeline {
  agent {
        docker {
          image 'nvcr.io/nvidia/pytorch:23.09-py3'
          args '--device=/dev/nvidia0 --gpus all --user 0:128 -v /home/TestData:/home/TestData -v $HOME/.cache:/root/.cache --shm-size=8g --env TRANSFORMERS_OFFLINE=0 --env HYDRA_FULL_ERROR=1'
        }
  }
  options {
    timeout(time: 8, unit: 'HOURS')
    disableConcurrentBuilds(abortPrevious: true)
  }

  stages {

    stage('Add git safe directory'){
      steps{
        sh 'git config --global --add safe.directory /var/lib/jenkins/workspace/NeMo_$GIT_BRANCH'
        sh 'git config --global --add safe.directory /raid/JenkinsWorkDir/workspace/NeMo_$GIT_BRANCH'
        sh 'git config --global --add safe.directory /mnt/D3/JenkinsWorkDir/workspace/NeMo_$GIT_BRANCH'
      }
    }

    stage('nvidia-smi'){
      steps{
        sh 'nvidia-smi'
      }
    }

    stage('PyTorch version') {
      steps {
        sh 'python -c "import torch; print(torch.__version__)"'
        sh 'python -c "import torchvision; print(torchvision.__version__)"'
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

    stage('Copyright Headers check') {
      steps {
        sh 'python tests/check_copyright_header.py --dir .'
      }
    }

    stage('NeMo Installation') {
      steps {
        sh './reinstall.sh release'
      }
    }

    // megatron-core 0.3 has been pinned in the requirements, this should not be needed on r1.21.0
    stage('Megatron Core installation') {
      steps {
        sh 'git clone https://github.com/NVIDIA/Megatron-LM.git && \
            cd Megatron-LM && \
            git checkout 973330e9c3681604703bf1eb6b5a265d1b9b9b38 && \
            pip install -e .'
      }
    }

    stage('Transformer Engine installation') {
      steps {
         sh 'git clone https://github.com/NVIDIA/TransformerEngine.git && \
             cd TransformerEngine && \
             git fetch origin e6676c53f26f6ef072943c909d136cf2a39c1d90 && \
             git checkout FETCH_HEAD && \
             git submodule init && git submodule update && \
             NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi pip install .'
      }
    }

    stage('PyTorch Lightning version') {
      steps {
        sh 'python -c "import pytorch_lightning; print(pytorch_lightning.__version__)"'
      }
    }

    stage('PyTorch Lightning DDP Checks') {
      steps {
        sh 'CUDA_VISIBLE_DEVICES="0,1" python "tests/core_ptl/check_for_ranks.py"'
      }
    }

    stage('Basic Import Checks') {
      steps {
        sh 'python -c "import nemo.collections.asr as nemo_asr"'
        sh 'python -c "import nemo.collections.nlp as nemo_nlp"'
        sh 'python -c "import nemo.collections.tts as nemo_tts"'
      }
    }

    stage('L2: Megatron Bert Pretraining and Resume Training with Pipeline Paralleism') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_bert_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/bert_pretrain_results \
        model.pipeline_model_parallel_size=2 \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.optim.sched.warmup_steps=2 \
        model.optim.sched.constant_steps=2 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.data.seq_length=128 \
        model.tokenizer.vocab_file=/home/TestData/nlp/megatron_bert/data/bert/vocab.txt \
        model.num_layers=8 \
        model.hidden_size=256 \
        model.num_attention_heads=8 \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_bert/data/bert/simple_wiki_bert_preproc_text_sentence,.5,/home/TestData/nlp/megatron_bert/data/bert/simple_wiki_bert_preproc_text_sentence] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/bert_index_mappings"
        sh "python examples/nlp/language_modeling/megatron_bert_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=20 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/bert_pretrain_results \
        exp_manager.resume_if_exists=True \
        model.pipeline_model_parallel_size=2 \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.optim.sched.warmup_steps=2 \
        model.optim.sched.constant_steps=2 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.data.seq_length=128 \
        model.tokenizer.vocab_file=/home/TestData/nlp/megatron_bert/data/bert/vocab.txt \
        model.num_layers=8 \
        model.hidden_size=256 \
        model.num_attention_heads=8 \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_bert/data/bert/simple_wiki_bert_preproc_text_sentence,.5,/home/TestData/nlp/megatron_bert/data/bert/simple_wiki_bert_preproc_text_sentence] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/bert_index_mappings"
        sh "rm -rf examples/nlp/language_modeling/bert_pretrain_results"
        sh "rm -rf examples/nlp/language_modeling/bert_index_mappings"
      }
    }
    stage('L2: Megatron Bert Pretraining and Resume Training') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_bert_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/bert_pretrain_results \
        model.tensor_model_parallel_size=2 \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.sequence_parallel=True \
        model.optim.sched.warmup_steps=2 \
        model.optim.sched.constant_steps=2 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.data.seq_length=128 \
        model.tokenizer.vocab_file=/home/TestData/nlp/megatron_bert/data/bert/vocab.txt \
        model.num_layers=8 \
        model.hidden_size=256 \
        model.num_attention_heads=8 \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_bert/data/bert/simple_wiki_bert_preproc_text_sentence,.5,/home/TestData/nlp/megatron_bert/data/bert/simple_wiki_bert_preproc_text_sentence] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/bert_index_mappings"
        sh "python examples/nlp/language_modeling/megatron_bert_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=20 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/bert_pretrain_results \
        exp_manager.resume_if_exists=True \
        model.tensor_model_parallel_size=2 \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.optim.sched.warmup_steps=2 \
        model.optim.sched.constant_steps=2 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.data.seq_length=128 \
        model.tokenizer.vocab_file=/home/TestData/nlp/megatron_bert/data/bert/vocab.txt \
        model.num_layers=8 \
        model.hidden_size=256 \
        model.num_attention_heads=8 \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_bert/data/bert/simple_wiki_bert_preproc_text_sentence,.5,/home/TestData/nlp/megatron_bert/data/bert/simple_wiki_bert_preproc_text_sentence] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/bert_index_mappings"
        sh "rm -rf examples/nlp/language_modeling/bert_pretrain_results"
        sh "rm -rf examples/nlp/language_modeling/bert_index_mappings"
      }
    }
    stage('L2: Megatron Core Bert Pretraining and Resume Training') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_bert_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/bert_pretrain_results \
        model.mcore_bert=True \
        model.tensor_model_parallel_size=2 \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.sequence_parallel=True \
        model.optim.sched.warmup_steps=2 \
        model.optim.sched.constant_steps=2 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.data.seq_length=128 \
        model.tokenizer.vocab_file=/home/TestData/nlp/megatron_bert/data/bert/vocab.txt \
        model.num_layers=8 \
        model.hidden_size=256 \
        model.num_attention_heads=8 \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_bert/data/bert/simple_wiki_bert_preproc_text_sentence,.5,/home/TestData/nlp/megatron_bert/data/bert/simple_wiki_bert_preproc_text_sentence] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/bert_index_mappings"
        sh "python examples/nlp/language_modeling/megatron_bert_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=20 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/bert_pretrain_results \
        exp_manager.resume_if_exists=True \
        model.mcore_bert=True \
        model.tensor_model_parallel_size=2 \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.optim.sched.warmup_steps=2 \
        model.optim.sched.constant_steps=2 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.data.seq_length=128 \
        model.tokenizer.vocab_file=/home/TestData/nlp/megatron_bert/data/bert/vocab.txt \
        model.num_layers=8 \
        model.hidden_size=256 \
        model.num_attention_heads=8 \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_bert/data/bert/simple_wiki_bert_preproc_text_sentence,.5,/home/TestData/nlp/megatron_bert/data/bert/simple_wiki_bert_preproc_text_sentence] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/bert_index_mappings"
        sh "rm -rf examples/nlp/language_modeling/bert_pretrain_results"
        sh "rm -rf examples/nlp/language_modeling/bert_index_mappings"
      }
    }
    stage('L2: Megatron RETRO Pretraining and Resume Training') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_retro_pretraining.py \
        trainer.devices=2 \
        trainer.num_nodes=1 \
        trainer.accelerator=gpu \
        trainer.accumulate_grad_batches=1 \
        trainer.limit_val_batches=2 \
        exp_manager.resume_if_exists=True \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        trainer.val_check_interval=10 \
        exp_manager.exp_dir=examples/nlp/language_modeling/retro_results \
        model.data.data_prefix='' \
        model.data.knn_index='' \
        model.data.retrieval_prefix='' \
        model.tensor_model_parallel_size=2 \
        model.micro_batch_size=4 \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.optim.sched.warmup_steps=2 \
        model.optim.sched.constant_steps=2 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.chunk_size=32 \
        model.enc_num_layers=2 \
        model.dec_num_layers=2 \
        model.enc_cross_attention=[1] \
        model.dec_cross_attention=[1] \
        +model.data.mock=True"
        sh "python examples/nlp/language_modeling/megatron_retro_pretraining.py \
        trainer.devices=2 \
        trainer.num_nodes=1 \
        trainer.accelerator=gpu \
        trainer.accumulate_grad_batches=1 \
        trainer.limit_val_batches=2 \
        exp_manager.resume_if_exists=True \
        trainer.max_steps=20 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        trainer.val_check_interval=10 \
        exp_manager.exp_dir=examples/nlp/language_modeling/retro_results \
        model.data.data_prefix='' \
        model.data.knn_index='' \
        model.data.retrieval_prefix='' \
        model.tensor_model_parallel_size=2 \
        model.micro_batch_size=4 \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.optim.sched.warmup_steps=2 \
        model.optim.sched.constant_steps=2 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.chunk_size=32 \
        model.enc_num_layers=2 \
        model.dec_num_layers=2 \
        model.enc_cross_attention=[1] \
        model.dec_cross_attention=[1] \
        +model.data.mock=True"
        sh "rm -rf examples/nlp/language_modeling/retro_results"
      }
    }
    stage('L2: Megatron RETRO muTransfer Pretraining Performance') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
            sh "python examples/nlp/language_modeling/megatron_retro_mutransfer_pretrain.py \
                trainer.devices=2 \
                trainer.num_nodes=1 \
                trainer.accelerator=gpu \
                trainer.accumulate_grad_batches=1 \
                trainer.max_steps=100 \
                trainer.log_every_n_steps=1 \
                trainer.precision=16 \
                trainer.val_check_interval=100 \
                trainer.limit_val_batches=0 \
                trainer.gradient_clip_val=1.0 \
                +trainer.num_sanity_val_steps=0 \
                exp_manager.exp_dir=examples/nlp/language_modeling/retro_results/ \
                +exp_manager.version=smalltest \
                model.data.neighbors=2 \
                model.megatron_amp_O2=False \
                model.apply_query_key_layer_scaling=False \
                model.tensor_model_parallel_size=1 \
                model.optim.name=muadamw \
                model.optim.weight_decay=0.1 \
                model.optim.betas=[0.9,0.95] \
                model.optim.lr=6e-4 \
                model.optim.sched.warmup_steps=1000 \
                model.optim.sched.constant_steps=0 \
                model.optim.sched.min_lr=6e-5 \
                model.add_position_embedding=False \
                model.enc_num_layers=2 \
                model.dec_num_layers=6 \
                model.enc_cross_attention=[0] \
                model.dec_cross_attention=[3,5] \
                model.hidden_size=96 \
                model.ffn_hidden_size=384 \
                model.init_method_std=0.023 \
                model.num_attention_heads=12 \
                model.max_position_embeddings=1024 \
                model.encoder_seq_length=1024 \
                model.tokenizer.library=megatron \
                model.tokenizer.type=GPT2BPETokenizer \
                model.tokenizer.merge_file=/home/TestData/nlp/megatron_retro/gpt2-merges.txt \
                model.tokenizer.vocab_file=/home/TestData/nlp/megatron_retro/gpt2-vocab.json \
                model.data.data_prefix=[/home/TestData/nlp/megatron_retro/retro_wiki_test_text_document] \
                model.data.knn_index=[/home/TestData/nlp/megatron_retro/knn2_map_wiki_test.idx] \
                model.data.retrieval_prefix=/home/TestData/nlp/megatron_retro/retro_wiki_test_text_document \
                model.data.index_mapping_dir=/home/TestData/nlp/megatron_retro \
                model.data.num_workers=8 \
                model.micro_batch_size=8 \
                model.normalization=rmsnorm \
                model.transformer_block_type=pre_ln \
                model.bias_activation_fusion=True \
                model.bias_dropout_add_fusion=False \
                model.masked_softmax_fusion=True \
                model.hidden_dropout=0 \
                model.attention_dropout=0 \
                model.fp32_residual_connection=True \
                model.shape_file=/home/TestData/nlp/megatron_retro/o1_rel_shape_info_tiny.yaml"
        sh '''python -c "import pandas as pd
import pathlib
from pandas.testing import assert_frame_equal
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch
if not (torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name()):
    import sys
    sys.exit(0)
event_file = list(pathlib.Path('examples/nlp/language_modeling/retro_results/megatron_retro/smalltest').glob('events.out.tfevents*'))[0]
ea = EventAccumulator(str(event_file)).Reload()
vals = []
for i in ea.Scalars('reduced_train_loss'):
    vals.append(i.value)
training_curve = pd.DataFrame({'loss': vals})
gt_curve = pd.read_csv('/home/TestData/nlp/megatron_retro/expected_learning_curve.csv')
assert_frame_equal(training_curve, gt_curve, rtol=1e-3, atol=1e-3)"'''
        sh "rm -rf examples/nlp/language_modeling/retro_results"
      }
    }
    stage('L2: BioMegatron Bert NER Task') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/token_classification/token_classification_train.py \
        exp_manager.exp_dir=examples/nlp/language_modeling/token_classification_results \
        trainer.max_epochs=1 \
        model.dataset.data_dir=/home/TestData/nlp/ner \
        model.language_model.pretrained_model_name=biomegatron345m_biovocab_30k_cased \
        model.tokenizer.tokenizer_name=null"
        sh "rm -rf examples/nlp/language_modeling/token_classification_results"
      }
    }
    stage('L2: Megatron GPT Pretraining and Resume Training TP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=2 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=3 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/gpt_pretrain_results \
        model.tensor_model_parallel_size=2 \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.optim.sched.warmup_steps=1 \
        model.optim.sched.constant_steps=1 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.data.seq_length=128 \
        model.normalization=rmsnorm \
        model.bias=False \
        model.bias_activation_fusion=False \
        model.bias_dropout_add_fusion=False \
        model.tokenizer.vocab_file=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
        model.tokenizer.merge_file=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
        model.num_layers=8 \
        model.hidden_size=256 \
        model.num_attention_heads=8 \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_granularity='full' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document,.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/gpt_index_mappings"
        sh "python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=2 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=6 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/gpt_pretrain_results \
        exp_manager.resume_if_exists=True \
        model.tensor_model_parallel_size=2 \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.optim.sched.warmup_steps=2 \
        model.optim.sched.constant_steps=2 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.data.seq_length=128 \
        model.normalization=rmsnorm \
        model.bias=False \
        model.bias_activation_fusion=False \
        model.bias_dropout_add_fusion=False \
        model.tokenizer.vocab_file=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
        model.tokenizer.merge_file=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
        model.num_layers=8 \
        model.hidden_size=256 \
        model.num_attention_heads=8 \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_granularity='full' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document,.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/gpt_index_mappings"
        sh "rm -rf examples/nlp/language_modeling/gpt_pretrain_results"
        sh "rm -rf examples/nlp/language_modeling/gpt_index_mappings"
      }
    }
    stage('L2: Megatron GPT with Rope Pretraining and Resume Training TP=2') {
     when {
       anyOf {
         branch 'main'
         changeRequest target: 'main'
       }
     }
     failFast true
     steps {
       sh "python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
       trainer.devices=2 \
       trainer.accelerator=gpu \
       trainer.log_every_n_steps=1 \
       trainer.val_check_interval=2 \
       trainer.limit_val_batches=2 \
       trainer.accumulate_grad_batches=1 \
       trainer.max_steps=3 \
       trainer.precision=16 \
       trainer.gradient_clip_val=1.0 \
       exp_manager.exp_dir=examples/nlp/language_modeling/gpt_pretrain_results \
       model.tensor_model_parallel_size=2 \
       model.optim.name=fused_adam \
       model.optim.lr=2e-4 \
       model.optim.sched.warmup_steps=1 \
       model.optim.sched.constant_steps=1 \
       model.optim.sched.min_lr=8e-5 \
       model.max_position_embeddings=128 \
       model.encoder_seq_length=128 \
       model.data.seq_length=128 \
       model.position_embedding_type=rope \
       model.rotary_percentage=0.5 \
       model.normalization=rmsnorm \
       model.bias=False \
       model.bias_activation_fusion=False \
       model.bias_dropout_add_fusion=False \
       model.tokenizer.vocab_file=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
       model.tokenizer.merge_file=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
       model.num_layers=8 \
       model.hidden_size=256 \
       model.num_attention_heads=8 \
       model.activations_checkpoint_method='block' \
       model.activations_checkpoint_granularity='full' \
       model.activations_checkpoint_num_layers=1 \
       model.data.data_prefix=[.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document,.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document] \
       model.data.index_mapping_dir=examples/nlp/language_modeling/gpt_index_mappings"
        // commented out to save time on github ci @adithyare
        //sh "python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
        //trainer.devices=2 \
        //trainer.accelerator=gpu \
        //trainer.log_every_n_steps=1 \
        //trainer.val_check_interval=2 \
        //trainer.limit_val_batches=1 \
        //trainer.accumulate_grad_batches=1 \
        //trainer.max_steps=6 \
        //trainer.precision=16 \
        //trainer.gradient_clip_val=1.0 \
        //exp_manager.exp_dir=examples/nlp/language_modeling/gpt_pretrain_results \
        //exp_manager.resume_if_exists=True \
        //model.tensor_model_parallel_size=2 \
        //model.optim.name=fused_adam \
        //model.optim.lr=2e-4 \
        //model.optim.sched.warmup_steps=2 \
        //model.optim.sched.constant_steps=2 \
        //model.optim.sched.min_lr=8e-5 \
        //model.max_position_embeddings=128 \
        //model.encoder_seq_length=128 \
        //model.data.seq_length=128 \
        //model.position_embedding_type=rope \
        //model.rotary_percentage=0.5 \
        //model.normalization=rmsnorm \
        //model.bias=False \
        //model.bias_activation_fusion=False \
        //model.bias_dropout_add_fusion=False \
        //model.tokenizer.vocab_file=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
        //model.tokenizer.merge_file=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
        //model.num_layers=8 \
        //model.hidden_size=256 \
        //model.num_attention_heads=8 \
        //model.activations_checkpoint_method='block' \
        //model.activations_checkpoint_granularity='full' \
        //model.activations_checkpoint_num_layers=1 \
        //model.data.data_prefix=[.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document,.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document] \
        //model.data.index_mapping_dir=examples/nlp/language_modeling/gpt_index_mappings"
       sh "rm -rf examples/nlp/language_modeling/gpt_pretrain_results"
       sh "rm -rf examples/nlp/language_modeling/gpt_index_mappings"
      }
     }

    // This test requires Ampere but some of the test GPUs are Volta
    // Need to add a check for compute capability before uncommenting this test
    // stage('L2: Megatron GPT with Rope Pretraining using Flash Attention and Resume Training TP=2') {
    //   when {
    //     anyOf {
    //       branch 'main'
    //       changeRequest target: 'main'
    //     }
    //   }
    //   failFast true
    //   steps {
    //     sh "python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
    //     trainer.devices=2 \
    //     trainer.accelerator=gpu \
    //     trainer.log_every_n_steps=1 \
    //     trainer.val_check_interval=2 \
    //     trainer.limit_val_batches=2 \
    //     trainer.accumulate_grad_batches=1 \
    //     trainer.max_steps=3 \
    //     trainer.precision=16 \
    //     trainer.gradient_clip_val=1.0 \
    //     exp_manager.exp_dir=examples/nlp/language_modeling/gpt_pretrain_results \
    //     model.tensor_model_parallel_size=2 \
    //     model.optim.name=fused_adam \
    //     model.optim.lr=2e-4 \
    //     model.optim.sched.warmup_steps=1 \
    //     model.optim.sched.constant_steps=1 \
    //     model.optim.sched.min_lr=8e-5 \
    //     model.max_position_embeddings=128 \
    //     model.encoder_seq_length=128 \
    //     model.data.seq_length=128 \
    //     model.position_embedding_type=rope \
    //     model.rotary_percentage=0.5 \
    //     model.normalization=rmsnorm \
    //     model.bias=False \
    //     model.bias_activation_fusion=False \
    //     model.bias_dropout_add_fusion=False \
    //     model.tokenizer.vocab_file=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
    //     model.tokenizer.merge_file=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
    //     model.num_layers=8 \
    //     model.hidden_size=256 \
    //     model.num_attention_heads=8 \
    //     model.activations_checkpoint_method='block' \
    //     model.activations_checkpoint_granularity='full' \
    //     model.activations_checkpoint_num_layers=1 \
    //     model.data.data_prefix=[.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document,.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document] \
    //     model.data.index_mapping_dir=examples/nlp/language_modeling/gpt_index_mappings \
    //     model.use_flash_attention=True "
    //     // commented out to save time on github ci @adithyare
    //     //sh "python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
    //     //trainer.devices=2 \
    //     //trainer.accelerator=gpu \
    //     //trainer.log_every_n_steps=1 \
    //     //trainer.val_check_interval=2 \
    //     //trainer.limit_val_batches=1 \
    //     //trainer.accumulate_grad_batches=1 \
    //     //trainer.max_steps=6 \
    //     //trainer.precision=16 \
    //     //trainer.gradient_clip_val=1.0 \
    //     //exp_manager.exp_dir=examples/nlp/language_modeling/gpt_pretrain_results \
    //     //exp_manager.resume_if_exists=True \
    //     //model.tensor_model_parallel_size=2 \
    //     //model.optim.name=fused_adam \
    //     //model.optim.lr=2e-4 \
    //     //model.optim.sched.warmup_steps=2 \
    //     //model.optim.sched.constant_steps=2 \
    //     //model.optim.sched.min_lr=8e-5 \
    //     //model.max_position_embeddings=128 \
    //     //model.encoder_seq_length=128 \
    //     //model.data.seq_length=128 \
    //     //model.position_embedding_type=rope \
    //     //model.rotary_percentage=0.5 \
    //     //model.normalization=rmsnorm \
    //     //model.bias=False \
    //     //model.bias_activation_fusion=False \
    //     //model.bias_dropout_add_fusion=False \
    //     //model.tokenizer.vocab_file=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
    //     //model.tokenizer.merge_file=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
    //     //model.num_layers=8 \
    //     //model.hidden_size=256 \
    //     //model.num_attention_heads=8 \
    //     //model.activations_checkpoint_method='block' \
    //     //model.activations_checkpoint_granularity='full' \
    //     //model.activations_checkpoint_num_layers=1 \
    //     //model.data.data_prefix=[.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document,.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document] \
    //     //model.data.index_mapping_dir=examples/nlp/language_modeling/gpt_index_mappings \
    //     //model.use_flash_attention=True"
    //     sh "rm -rf examples/nlp/language_modeling/gpt_pretrain_results"
    //     sh "rm -rf examples/nlp/language_modeling/gpt_index_mappings"
    //   }
    // }
    stage('L2: Megatron GPT with ALiBi Pretraining and Resume Training TP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=2 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=3 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/gpt_pretrain_results \
        model.tensor_model_parallel_size=2 \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.optim.sched.warmup_steps=1 \
        model.optim.sched.constant_steps=1 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.data.seq_length=128 \
        model.position_embedding_type=alibi \
        model.normalization=rmsnorm \
        model.bias=False \
        model.bias_activation_fusion=False \
        model.bias_dropout_add_fusion=False \
        model.tokenizer.vocab_file=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
        model.tokenizer.merge_file=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
        model.num_layers=8 \
        model.hidden_size=256 \
        model.num_attention_heads=8 \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_granularity='full' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document,.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/gpt_index_mappings"
        // not testing resume functionality to save time on ci @adithyare
        //sh "python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
        //trainer.devices=2 \
        //trainer.accelerator=gpu \
        //trainer.log_every_n_steps=1 \
        //trainer.val_check_interval=2 \
        //trainer.limit_val_batches=1 \
        //trainer.accumulate_grad_batches=1 \
        //trainer.max_steps=6 \
        //trainer.precision=16 \
        //trainer.gradient_clip_val=1.0 \
        //exp_manager.exp_dir=examples/nlp/language_modeling/gpt_pretrain_results \
        //exp_manager.resume_if_exists=True \
        //model.tensor_model_parallel_size=2 \
        //model.optim.name=fused_adam \
        //model.optim.lr=2e-4 \
        //model.optim.sched.warmup_steps=2 \
        //model.optim.sched.constant_steps=2 \
        //model.optim.sched.min_lr=8e-5 \
        //model.max_position_embeddings=128 \
        //model.encoder_seq_length=128 \
        //model.data.seq_length=128 \
        //model.position_embedding_type=alibi \
        //model.normalization=rmsnorm \
        //model.bias=False \
        //model.bias_activation_fusion=False \
        //model.bias_dropout_add_fusion=False \
        //model.tokenizer.vocab_file=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
        //model.tokenizer.merge_file=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
        //model.num_layers=8 \
        //model.hidden_size=256 \
        //model.num_attention_heads=8 \
        //model.activations_checkpoint_method='block' \
        //model.activations_checkpoint_granularity='full' \
        //model.activations_checkpoint_num_layers=1 \
        //model.data.data_prefix=[.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document,.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document] \
        //model.data.index_mapping_dir=examples/nlp/language_modeling/gpt_index_mappings"
        sh "rm -rf examples/nlp/language_modeling/gpt_pretrain_results"
        sh "rm -rf examples/nlp/language_modeling/gpt_index_mappings"
      }
    }
    stage('L2: Megatron GPT with KERPLE Pretraining and Resume Training TP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=2 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=3 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/gpt_pretrain_results \
        model.tensor_model_parallel_size=2 \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.optim.sched.warmup_steps=1 \
        model.optim.sched.constant_steps=1 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.data.seq_length=128 \
        model.position_embedding_type=kerple \
        model.normalization=rmsnorm \
        model.bias=False \
        model.bias_activation_fusion=False \
        model.bias_dropout_add_fusion=False \
        model.tokenizer.vocab_file=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
        model.tokenizer.merge_file=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
        model.num_layers=8 \
        model.hidden_size=256 \
        model.num_attention_heads=8 \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_granularity='full' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document,.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/gpt_index_mappings"
        // commented out to save time on github ci @adithyare
        //sh "python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
        //trainer.devices=2 \
        //trainer.accelerator=gpu \
        //trainer.log_every_n_steps=1 \
        //trainer.val_check_interval=2 \
        //trainer.limit_val_batches=1 \
        //trainer.accumulate_grad_batches=1 \
        //trainer.max_steps=6 \
        //trainer.precision=16 \
        //trainer.gradient_clip_val=1.0 \
        //exp_manager.exp_dir=examples/nlp/language_modeling/gpt_pretrain_results \
        //exp_manager.resume_if_exists=True \
        //model.tensor_model_parallel_size=2 \
        //model.optim.name=fused_adam \
        //model.optim.lr=2e-4 \
        //model.optim.sched.warmup_steps=2 \
        //model.optim.sched.constant_steps=2 \
        //model.optim.sched.min_lr=8e-5 \
        //model.max_position_embeddings=128 \
        //model.encoder_seq_length=128 \
        //model.data.seq_length=128 \
        //model.position_embedding_type=kerple \
        //model.normalization=rmsnorm \
        //model.bias=False \
        //model.bias_activation_fusion=False \
        //model.bias_dropout_add_fusion=False \
        //model.tokenizer.vocab_file=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
        //model.tokenizer.merge_file=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
        //model.num_layers=8 \
        //model.hidden_size=256 \
        //model.num_attention_heads=8 \
        //model.activations_checkpoint_method='block' \
        //model.activations_checkpoint_granularity='full' \
        //model.activations_checkpoint_num_layers=1 \
        //model.data.data_prefix=[.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document,.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document] \
        //model.data.index_mapping_dir=examples/nlp/language_modeling/gpt_index_mappings"
        sh "rm -rf examples/nlp/language_modeling/gpt_pretrain_results"
        sh "rm -rf examples/nlp/language_modeling/gpt_index_mappings"
      }
    }
    stage('L2: Megatron GPT Pretraining and Resume Training PP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
        trainer.devices=2 \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=2 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=3 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/gpt_pretrain_results \
        model.pipeline_model_parallel_size=2 \
        model.tensor_model_parallel_size=1 \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.optim.sched.warmup_steps=1 \
        model.optim.sched.constant_steps=1 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.activation=fast-swiglu \
        model.bias_activation_fusion=False \
        model.hidden_dropout=0.0 \
        model.attention_dropout=0.0 \
        model.transformer_block_type=normformer \
        model.headscale=True \
        model.data.seq_length=128 \
        model.tokenizer.vocab_file=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
        model.tokenizer.merge_file=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
        model.num_layers=8 \
        model.hidden_size=256 \
        model.num_attention_heads=8 \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document,.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/gpt_index_mappings"
        sh "python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
        trainer.devices=2 \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=2 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=6 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/gpt_pretrain_results \
        exp_manager.resume_if_exists=True \
        model.pipeline_model_parallel_size=2 \
        model.tensor_model_parallel_size=1 \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.optim.sched.warmup_steps=2 \
        model.optim.sched.constant_steps=2 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=128 \
        model.encoder_seq_length=128 \
        model.activation=fast-swiglu \
        model.bias_activation_fusion=False \
        model.hidden_dropout=0.0 \
        model.attention_dropout=0.0 \
        model.transformer_block_type=normformer \
        model.headscale=True \
        model.data.seq_length=128 \
        model.tokenizer.vocab_file=/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json \
        model.tokenizer.merge_file=/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt \
        model.num_layers=8 \
        model.hidden_size=256 \
        model.num_attention_heads=8 \
        model.activations_checkpoint_method='block' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document,.5,/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/gpt_index_mappings"
        sh "rm -rf examples/nlp/language_modeling/gpt_pretrain_results"
        sh "rm -rf examples/nlp/language_modeling/gpt_index_mappings"
      }
    }
    // @athitten Remove /home/TestData/nlp/megatron_sft/trec.jsonl for validation and test file until we have support for multiple dataloaders in lightning 2.0
    stage('L2: Megatron GPT Finetuning PP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/tuning/megatron_gpt_peft_tuning.py \
        trainer.devices=2 \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=2 \
        +trainer.limit_val_batches=2 \
        trainer.max_steps=3 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/gpt_sft_results \
        model.pipeline_model_parallel_size=2 \
        model.tensor_model_parallel_size=1 \
        model.restore_from_path=/home/TestData/nlp/megatron_gpt/PP2/gpt_pp2_tp1.nemo \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.peft.peft_scheme=null \
        model.data.train_ds.micro_batch_size=1 \
        model.data.train_ds.global_batch_size=4 \
        model.data.train_ds.file_names=[/home/TestData/nlp/megatron_sft/quarel.jsonl,/home/TestData/nlp/megatron_sft/trec.jsonl] \
        model.data.train_ds.concat_sampling_probabilities=[0.3,0.7] \
        model.data.train_ds.num_workers=0 \
        model.data.test_ds.micro_batch_size=1 \
        model.data.test_ds.global_batch_size=1 \
        model.data.test_ds.file_names=[/home/TestData/nlp/megatron_sft/quarel.jsonl] \
        model.data.test_ds.names=[quarel] \
        model.data.validation_ds.micro_batch_size=1 \
        model.data.validation_ds.global_batch_size=1 \
        model.data.validation_ds.num_workers=0 \
        model.data.validation_ds.file_names=[/home/TestData/nlp/megatron_sft/quarel.jsonl] \
        model.data.validation_ds.names=[quarel]"
        sh "python examples/nlp/language_modeling/tuning/megatron_gpt_peft_tuning.py \
        trainer.devices=2 \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=1 \
        +trainer.limit_val_batches=2 \
        trainer.max_steps=3 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/gpt_sft_results \
        model.pipeline_model_parallel_size=2 \
        model.tensor_model_parallel_size=1 \
        model.restore_from_path=/home/TestData/nlp/megatron_gpt/PP2/gpt_pp2_tp1.nemo \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.peft.peft_scheme=null \
        model.data.train_ds.micro_batch_size=1 \
        model.data.train_ds.global_batch_size=4 \
        model.data.train_ds.file_names=[/home/TestData/nlp/megatron_sft/quarel.jsonl,/home/TestData/nlp/megatron_sft/trec.jsonl] \
        model.data.train_ds.concat_sampling_probabilities=[0.3,0.7] \
        model.data.train_ds.num_workers=0 \
        model.data.test_ds.micro_batch_size=1 \
        model.data.test_ds.global_batch_size=1 \
        model.data.test_ds.file_names=[/home/TestData/nlp/megatron_sft/quarel.jsonl] \
        model.data.test_ds.names=[quarel] \
        model.data.validation_ds.micro_batch_size=1 \
        model.data.validation_ds.global_batch_size=1 \
        model.data.validation_ds.num_workers=0 \
        model.data.validation_ds.file_names=[/home/TestData/nlp/megatron_sft/quarel.jsonl] \
        model.data.validation_ds.names=[quarel]"
        sh "rm -rf examples/nlp/language_modeling/gpt_sft_results"
      }
    }
    stage('L2: Megatron GPT Finetuning StarCoder PP=1') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/tuning/megatron_gpt_sft.py \
        trainer.devices=1 \
        trainer.num_nodes=1 \
        trainer.precision=32 \
        trainer.max_steps=4 \
        trainer.val_check_interval=4 \
        trainer.enable_checkpointing=False \
        +trainer.limit_val_batches=2 \
        +trainer.limit_test_batches=2 \
        exp_manager.checkpoint_callback_params.save_best_model=False \
        exp_manager.exp_dir=examples/nlp/language_modeling/gpt_sft_results \
        model.optim.name=distributed_fused_adam \
        model.restore_from_path=/home/TestData/nlp/megatron_gpt/starcoder-ci-nemo/megatron_starcoder_tp1_pp1.nemo \
        model.tensor_model_parallel_size=1 \
        model.pipeline_model_parallel_size=1 \
        model.data.train_ds.file_names=[/home/TestData/nlp/megatron_sft/quarel.jsonl] \
        model.data.train_ds.num_workers=0 \
        model.data.test_ds.file_names=[/home/TestData/nlp/megatron_sft/quarel.jsonl] \
        model.data.validation_ds.num_workers=0 \
        model.data.validation_ds.file_names=[/home/TestData/nlp/megatron_sft/quarel.jsonl] \
        model.data.test_ds.num_workers=0 \
        model.data.train_ds.concat_sampling_probabilities=[1.0]"
        sh "rm -rf examples/nlp/language_modeling/gpt_sft_results"
      }
    }
    stage('L2: Megatron GPT PEFT Lora PP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "rm -rf examples/nlp/language_modeling/gpt_peft_lora_results_pp2"
        sh "python examples/nlp/language_modeling/tuning/megatron_gpt_peft_tuning.py \
        trainer.devices=2 \
        trainer.log_every_n_steps=1 \
        trainer.max_epochs=9999 \
        trainer.max_steps=3 \
        trainer.val_check_interval=3 \
        ++trainer.limit_val_batches=2 \
        trainer.precision=16 \
        exp_manager.exp_dir=examples/nlp/language_modeling/gpt_peft_lora_results_pp2 \
        model.pipeline_model_parallel_size=2 \
        model.tensor_model_parallel_size=1 \
        model.restore_from_path=/home/TestData/nlp/megatron_gpt/PP2/gpt_pp2_tp1.nemo \
        model.peft.peft_scheme='lora' \
        model.answer_only_loss=True \
        model.micro_batch_size=1 \
        model.global_batch_size=1 \
        model.data.train_ds.file_names=[/home/TestData/nlp/megatron_sft/quarel.jsonl] \
        model.data.train_ds.concat_sampling_probabilities=[1.0] \
        model.data.train_ds.num_workers=0 \
        model.data.validation_ds.num_workers=0 \
        model.data.validation_ds.file_names=[/home/TestData/nlp/megatron_sft/quarel.jsonl] \
        model.data.validation_ds.names=[quarel]"
        sh "rm -rf examples/nlp/language_modeling/gpt_peft_lora_results_pp2"
      }
    }
    stage('L2: Megatron GPT PEFT Lora TP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "rm -rf /home/TestData/nlp/lora_tuning_tp2"
        sh "python examples/nlp/language_modeling/tuning/megatron_gpt_peft_tuning.py \
        trainer.devices=2 \
        trainer.log_every_n_steps=1 \
        trainer.max_epochs=9999 \
        trainer.max_steps=3 \
        trainer.val_check_interval=3 \
        ++trainer.limit_val_batches=2 \
        trainer.precision=16 \
        exp_manager.exp_dir=/home/TestData/nlp/lora_tuning_tp2 \
        model.pipeline_model_parallel_size=1 \
        model.tensor_model_parallel_size=2 \
        model.restore_from_path=/home/TestData/nlp/megatron_gpt/TP2/megatron_gpt_tp2.nemo \
        model.peft.peft_scheme='lora' \
        model.answer_only_loss=True \
        model.micro_batch_size=1 \
        model.global_batch_size=1 \
        model.data.train_ds.file_names=[/home/TestData/nlp/megatron_sft/quarel.jsonl] \
        model.data.train_ds.concat_sampling_probabilities=[1.0] \
        model.data.train_ds.num_workers=0 \
        model.data.validation_ds.num_workers=0 \
        model.data.validation_ds.file_names=[/home/TestData/nlp/megatron_sft/quarel.jsonl] \
        model.data.validation_ds.names=[quarel]"
        sh "python examples/nlp/language_modeling/tuning/megatron_gpt_peft_eval.py \
        model.restore_from_path=/home/TestData/nlp/megatron_gpt/TP2/megatron_gpt_tp2.nemo \
        model.peft.restore_from_path=/home/TestData/nlp/lora_tuning_tp2/megatron_gpt_peft_lora_tuning/checkpoints/megatron_gpt_peft_lora_tuning.nemo \
        model.peft.restore_from_ckpt_name=null \
        model.peft.restore_from_hparams_path=null \
        model.tensor_model_parallel_size=2 \
        trainer.devices=2 \
        model.data.test_ds.file_names=[/home/TestData/nlp/megatron_sft/quarel_4.jsonl] \
        model.data.test_ds.names=['quarel4'] \
        model.global_batch_size=2 \
        model.micro_batch_size=1 \
        model.data.test_ds.tokens_to_generate=10 \
        model.data.test_ds.write_predictions_to_file=True \
        model.data.test_ds.output_file_path_prefix='/home/TestData/nlp/lora_tuning_tp2/out' \
        inference.greedy=True \
        inference.repetition_penalty=1.0 \
        inference.outfile_path='/home/TestData/nlp/lora_tuning_tp2/out.jsonl'"
        sh "rm -rf /home/TestData/nlp/lora_tuning_tp2"
      }
    }
    stage('L2: Megatron GPT Eval') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps{
        sh "python examples/nlp/language_modeling/megatron_gpt_eval.py \
            gpt_model_file=/home/TestData/nlp/megatron_gpt/125M/megatron_gpt.nemo \
            prompts=['How to fix GPU memory? A:'] \
            tensor_model_parallel_size=1 \
            inference.tokens_to_generate=32 \
            trainer.precision=32"
      }
    }
    stage('L2: Megatron GPT Eval PP2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_gpt_eval.py \
            gpt_model_file=/home/TestData/nlp/megatron_gpt/PP2/gpt_pp2_tp1.nemo \
            server=False \
            tensor_model_parallel_size=1 \
            pipeline_model_parallel_size=2 \
            trainer.devices=2 \
            trainer.num_nodes=1 \
            trainer.precision=32"
      }
    }
    stage('L2: Megatron GPT SFT Eval (inference seq len > training seq len)') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps{
        sh "python examples/nlp/language_modeling/tuning/megatron_gpt_peft_eval.py \
            model.restore_from_path=/home/TestData/nlp/megatron_gpt_sft/megatron_gpt_rope_sft.nemo \
            model.peft.restore_from_path=null \
            model.data.test_ds.file_names=['/home/TestData/nlp/megatron_gpt_sft/sample.jsonl'] \
            model.data.test_ds.names=['test'] \
            model.data.test_ds.global_batch_size=1 \
            model.data.test_ds.micro_batch_size=1 \
            model.data.test_ds.tokens_to_generate=30 \
            model.data.test_ds.max_seq_length=6000 \
            model.data.test_ds.write_predictions_to_file=True \
            model.data.test_ds.output_file_path_prefix='examples/nlp/language_modeling/out' \
            inference.greedy=True \
            inference.repetition_penalty=1.0 \
            inference.outfile_path='examples/nlp/language_modeling/out.jsonl' && \
            rm -rf examples/nlp/language_modeling/out.jsonl"
      }
    }

    // TODO: Add this test back. Test was failing on CI machines due to HW error
    // stage('L2: Megatron GPT Convert from Megatron-LM checkpoing and Eval') {
    //   when {
    //     anyOf {
    //       branch 'main'
    //       changeRequest target: 'main'
    //     }
    //   }
    //   failFast true
    //   steps {
    //     sh "python -m torch.distributed.launch --nproc_per_node=2 \
    //     examples/nlp/language_modeling/megatron_lm_ckpt_to_nemo.py \
    //     --checkpoint_folder=/home/TestData/nlp/megatron_gpt/data/gpt/iter_0008700 \
    //     --checkpoint_name=model_optim_rng.pt \
    //     --hparams_file=/home/TestData/nlp/megatron_gpt/data/gpt/iter_0008700/hparams.yaml \
    //     --nemo_file_path=examples/nlp/language_modeling/small_gpt.nemo \
    //     --model_type=gpt \
    //     --pipeline_model_parallel_size=1 \
    //     --gpus_per_node=2 \
    //     --tensor_model_parallel_size=2"
    //     sh "python examples/nlp/language_modeling/megatron_gpt_eval.py \
    //     --gpt_model_file=examples/nlp/language_modeling/small_gpt.nemo \
    //     --tokens_to_generate=32 \
    //     --tensor_model_parallel_size=2 \
    //     --prompt='This is a test.'"
    //     sh "rm examples/nlp/language_modeling/small_gpt.nemo"
    //   }
    // }
    stage('L2: Megatron Change Partitions') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel{
        stage('Reduce TP Num Partitions (2 to 1) and PP Num Partitions (1 to 2)'){
          steps{
            sh "python examples/nlp/language_modeling/megatron_change_num_partitions.py \
                --model_file \
                /home/TestData/nlp/megatron_gpt/TP2/megatron_gpt_tp2.nemo \
                --target_file \
                /home/TestData/nlp/megatron_gpt/TP2-Temp/test-reduce.nemo \
                --tensor_model_parallel_size \
                2 \
                --target_tensor_model_parallel_size \
                1 \
                --pipeline_model_parallel_size \
                1 \
                --target_pipeline_model_parallel_size \
                2"
            sh "rm /home/TestData/nlp/megatron_gpt/TP2-Temp/test-reduce.nemo"
          }
        }
        stage('Increase TP Num Partitions (2 to 4) and PP Num Partitions (1 to 2)'){
          steps{
            sh "python examples/nlp/language_modeling/megatron_change_num_partitions.py \
                --model_file \
                /home/TestData/nlp/megatron_gpt/TP2/megatron_gpt_tp2.nemo \
                --target_file \
                /home/TestData/nlp/megatron_gpt/TP2-Temp/test-increase.nemo \
                --tensor_model_parallel_size \
                2 \
                --target_tensor_model_parallel_size \
                4 \
                --pipeline_model_parallel_size \
                1 \
                --target_pipeline_model_parallel_size \
                1"
            sh "rm /home/TestData/nlp/megatron_gpt/TP2-Temp/test-increase.nemo"
          }
        }
      }
    }
    stage('L2: Megatron T5 Pretraining and Resume Training TP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_t5_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/t5_pretrain_results \
        model.tensor_model_parallel_size=2 \
        model.seq_length=128 \
        model.encoder.num_layers=4 \
        model.encoder.hidden_size=64 \
        model.encoder.num_attention_heads=8 \
        model.encoder.activation='swiglu' \
        model.encoder.masked_softmax_fusion=False \
        model.encoder.bias_activation_fusion=False \
        model.encoder.activations_checkpoint_method='block' \
        model.encoder.activations_checkpoint_num_layers=1 \
        model.encoder.position_embedding_type=relative \
        model.decoder.num_layers=2 \
        model.decoder.hidden_size=64 \
        model.decoder.num_attention_heads=8 \
        model.decoder.activation='fast-swiglu' \
        model.decoder.masked_softmax_fusion=False \
        model.decoder.bias_activation_fusion=False \
        model.decoder.activations_checkpoint_method='block' \
        model.decoder.activations_checkpoint_num_layers=1 \
        model.encoder.transformer_block_type='pre_ln' \
        model.decoder.transformer_block_type='pre_ln' \
        model.data.data_prefix=[.5,/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src,.5,/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/t5_index_mappings \
        model.data.data_impl=text_mmap \
        +model.data.data_impl_kwargs.newline_int=10 \
        +model.data.data_impl_kwargs.header_lines=0 \
        +model.data.data_impl_kwargs.workers=null \
        +model.data.data_impl_kwargs.sort_dataset_paths=False \
        model.share_token_embeddings=False \
        model.share_decoder_tokens_head_embeddings=False"
        sh "python examples/nlp/language_modeling/megatron_t5_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=1 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/t5_pretrain_results \
        exp_manager.resume_if_exists=True \
        model.tensor_model_parallel_size=2 \
        model.seq_length=128 \
        model.encoder.num_layers=4 \
        model.encoder.hidden_size=64 \
        model.encoder.num_attention_heads=8 \
        model.encoder.activation='swiglu' \
        model.encoder.masked_softmax_fusion=False \
        model.encoder.bias_activation_fusion=False \
        model.encoder.activations_checkpoint_method='block' \
        model.encoder.activations_checkpoint_num_layers=1 \
        model.encoder.position_embedding_type=relative \
        model.decoder.num_layers=2 \
        model.decoder.hidden_size=64 \
        model.decoder.num_attention_heads=8 \
        model.decoder.activation='fast-swiglu' \
        model.decoder.masked_softmax_fusion=False \
        model.decoder.bias_activation_fusion=False \
        model.decoder.activations_checkpoint_method='block' \
        model.decoder.activations_checkpoint_num_layers=1 \
        model.encoder.transformer_block_type='pre_ln' \
        model.decoder.transformer_block_type='pre_ln' \
        model.data.data_prefix=[.5,/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src,.5,/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/t5_index_mappings \
        model.data.data_impl=text_mmap \
        +model.data.data_impl_kwargs.newline_int=10 \
        +model.data.data_impl_kwargs.header_lines=0 \
        +model.data.data_impl_kwargs.workers=null \
        +model.data.data_impl_kwargs.sort_dataset_paths=False \
        model.share_token_embeddings=False \
        model.share_decoder_tokens_head_embeddings=False"
        sh "rm -rf examples/nlp/language_modeling/t5_pretrain_results"
        sh "rm -rf examples/nlp/language_modeling/t5_index_mappings"
      }
    }
    stage('L2: Megatron T5 with ALiBi Pretraining and Resume Training TP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_t5_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/t5_pretrain_results \
        model.tensor_model_parallel_size=2 \
        model.seq_length=128 \
        model.encoder.num_layers=4 \
        model.encoder.hidden_size=64 \
        model.encoder.num_attention_heads=8 \
        model.encoder.activation='swiglu' \
        model.encoder.masked_softmax_fusion=False \
        model.encoder.bias_activation_fusion=False \
        model.encoder.activations_checkpoint_method='block' \
        model.encoder.activations_checkpoint_num_layers=1 \
        model.encoder.position_embedding_type=alibi \
        model.decoder.num_layers=2 \
        model.decoder.hidden_size=64 \
        model.decoder.num_attention_heads=8 \
        model.decoder.activation='swiglu' \
        model.decoder.masked_softmax_fusion=False \
        model.decoder.bias_activation_fusion=False \
        model.decoder.activations_checkpoint_method='block' \
        model.decoder.activations_checkpoint_num_layers=1 \
        model.encoder.transformer_block_type='pre_ln' \
        model.decoder.transformer_block_type='pre_ln' \
        model.data.data_prefix=[.5,/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src,.5,/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/t5_index_mappings \
        model.data.data_impl=text_mmap \
        +model.data.data_impl_kwargs.newline_int=10 \
        +model.data.data_impl_kwargs.header_lines=0 \
        +model.data.data_impl_kwargs.workers=null \
        +model.data.data_impl_kwargs.sort_dataset_paths=False \
        model.share_token_embeddings=False \
        model.share_decoder_tokens_head_embeddings=False"
        sh "python examples/nlp/language_modeling/megatron_t5_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=1 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/t5_pretrain_results \
        exp_manager.resume_if_exists=True \
        model.tensor_model_parallel_size=2 \
        model.seq_length=128 \
        model.encoder.num_layers=4 \
        model.encoder.hidden_size=64 \
        model.encoder.num_attention_heads=8 \
        model.encoder.activation='swiglu' \
        model.encoder.masked_softmax_fusion=False \
        model.encoder.bias_activation_fusion=False \
        model.encoder.activations_checkpoint_method='block' \
        model.encoder.activations_checkpoint_num_layers=1 \
        model.encoder.position_embedding_type=alibi \
        model.decoder.num_layers=2 \
        model.decoder.hidden_size=64 \
        model.decoder.num_attention_heads=8 \
        model.decoder.activation='swiglu' \
        model.decoder.masked_softmax_fusion=False \
        model.decoder.bias_activation_fusion=False \
        model.decoder.activations_checkpoint_method='block' \
        model.decoder.activations_checkpoint_num_layers=1 \
        model.encoder.transformer_block_type='pre_ln' \
        model.decoder.transformer_block_type='pre_ln' \
        model.data.data_prefix=[.5,/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src,.5,/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/t5_index_mappings \
        model.data.data_impl=text_mmap \
        +model.data.data_impl_kwargs.newline_int=10 \
        +model.data.data_impl_kwargs.header_lines=0 \
        +model.data.data_impl_kwargs.workers=null \
        +model.data.data_impl_kwargs.sort_dataset_paths=False \
        model.share_token_embeddings=False \
        model.share_decoder_tokens_head_embeddings=False"
        sh "rm -rf examples/nlp/language_modeling/t5_pretrain_results"
        sh "rm -rf examples/nlp/language_modeling/t5_index_mappings"
      }
    }
    stage('L2: Megatron T5 with KERPLE Pretraining and Resume Training TP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_t5_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/t5_pretrain_results \
        model.tensor_model_parallel_size=2 \
        model.seq_length=128 \
        model.encoder.num_layers=4 \
        model.encoder.hidden_size=64 \
        model.encoder.num_attention_heads=8 \
        model.encoder.activation='swiglu' \
        model.encoder.masked_softmax_fusion=False \
        model.encoder.bias_activation_fusion=False \
        model.encoder.activations_checkpoint_method='block' \
        model.encoder.activations_checkpoint_num_layers=1 \
        model.encoder.position_embedding_type=kerple \
        model.decoder.num_layers=2 \
        model.decoder.hidden_size=64 \
        model.decoder.num_attention_heads=8 \
        model.decoder.activation='swiglu' \
        model.decoder.masked_softmax_fusion=False \
        model.decoder.bias_activation_fusion=False \
        model.decoder.activations_checkpoint_method='block' \
        model.decoder.activations_checkpoint_num_layers=1 \
        model.encoder.transformer_block_type='pre_ln' \
        model.decoder.transformer_block_type='pre_ln' \
        model.data.data_prefix=[.5,/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src,.5,/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/t5_index_mappings \
        model.data.data_impl=text_mmap \
        +model.data.data_impl_kwargs.newline_int=10 \
        +model.data.data_impl_kwargs.header_lines=0 \
        +model.data.data_impl_kwargs.workers=null \
        +model.data.data_impl_kwargs.sort_dataset_paths=False \
        model.share_token_embeddings=False \
        model.share_decoder_tokens_head_embeddings=False"
        sh "python examples/nlp/language_modeling/megatron_t5_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=1 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/t5_pretrain_results \
        exp_manager.resume_if_exists=True \
        model.tensor_model_parallel_size=2 \
        model.seq_length=128 \
        model.encoder.num_layers=4 \
        model.encoder.hidden_size=64 \
        model.encoder.num_attention_heads=8 \
        model.encoder.activation='swiglu' \
        model.encoder.masked_softmax_fusion=False \
        model.encoder.bias_activation_fusion=False \
        model.encoder.activations_checkpoint_method='block' \
        model.encoder.activations_checkpoint_num_layers=1 \
        model.encoder.position_embedding_type=kerple \
        model.decoder.num_layers=2 \
        model.decoder.hidden_size=64 \
        model.decoder.num_attention_heads=8 \
        model.decoder.activation='swiglu' \
        model.decoder.masked_softmax_fusion=False \
        model.decoder.bias_activation_fusion=False \
        model.decoder.activations_checkpoint_method='block' \
        model.decoder.activations_checkpoint_num_layers=1 \
        model.encoder.transformer_block_type='pre_ln' \
        model.decoder.transformer_block_type='pre_ln' \
        model.data.data_prefix=[.5,/home/TestData/nlp/nmt/toy_data/wmt14-de-en.src,.5,/home/TestData/nlp/nmt/toy_data/wmt14-de-en.ref] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/t5_index_mappings \
        model.data.data_impl=text_mmap \
        +model.data.data_impl_kwargs.newline_int=10 \
        +model.data.data_impl_kwargs.header_lines=0 \
        +model.data.data_impl_kwargs.workers=null \
        +model.data.data_impl_kwargs.sort_dataset_paths=False \
        model.share_token_embeddings=False \
        model.share_decoder_tokens_head_embeddings=False"
        sh "rm -rf examples/nlp/language_modeling/t5_pretrain_results"
        sh "rm -rf examples/nlp/language_modeling/t5_index_mappings"
      }
    }
    stage('L2: Megatron T5 Pretraining and Resume Training PP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_t5_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/t5_pretrain_results \
        model.pipeline_model_parallel_size=2 \
        model.pipeline_model_parallel_split_rank=1 \
        model.seq_length=256 \
        model.encoder.num_layers=4 \
        model.decoder.num_layers=1 \
        model.encoder.hidden_size=64 \
        model.decoder.hidden_size=64 \
        model.encoder.num_attention_heads=8 \
        model.decoder.num_attention_heads=8 \
        model.decoder.ffn_hidden_size=2048 \
        model.encoder.activation='gelu' \
        model.encoder.activations_checkpoint_method='block' \
        model.encoder.activations_checkpoint_num_layers=1 \
        model.encoder.transformer_block_type='pre_ln' \
        model.decoder.transformer_block_type='post_ln' \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document,.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/t5_index_mappings"
        sh "python examples/nlp/language_modeling/megatron_t5_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=1 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/t5_pretrain_results \
        exp_manager.resume_if_exists=True \
        model.pipeline_model_parallel_size=2 \
        model.pipeline_model_parallel_split_rank=1 \
        model.seq_length=256 \
        model.encoder.num_layers=4 \
        model.decoder.num_layers=1 \
        model.encoder.hidden_size=64 \
        model.decoder.hidden_size=64 \
        model.encoder.num_attention_heads=8 \
        model.decoder.num_attention_heads=8 \
        model.decoder.ffn_hidden_size=2048 \
        model.encoder.activation='gelu' \
        model.encoder.activations_checkpoint_method='block' \
        model.encoder.activations_checkpoint_num_layers=1 \
        model.encoder.transformer_block_type='pre_ln' \
        model.decoder.transformer_block_type='post_ln' \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document,.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/t5_index_mappings"
        sh "rm -rf examples/nlp/language_modeling/t5_pretrain_results"
        sh "rm -rf examples/nlp/language_modeling/t5_index_mappings"
      }
    }
    stage('L2: Megatron T5 w/ Mixture of Expert Pretraining') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_t5_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/t5_pretrain_results \
        model.pipeline_model_parallel_split_rank=1 \
        model.seq_length=256 \
        model.encoder.num_layers=4 \
        model.decoder.num_layers=1 \
        model.encoder.num_moe_experts=4 \
        model.decoder.num_moe_experts=4 \
        model.encoder.moe_frequency=3 \
        model.decoder.moe_frequency=1 \
        model.encoder.hidden_size=64 \
        model.decoder.hidden_size=64 \
        model.encoder.num_attention_heads=8 \
        model.decoder.num_attention_heads=8 \
        model.decoder.ffn_hidden_size=2048 \
        model.encoder.activation='gelu' \
        model.encoder.activations_checkpoint_method='block' \
        model.encoder.activations_checkpoint_num_layers=1 \
        model.encoder.transformer_block_type='pre_ln' \
        model.decoder.transformer_block_type='post_ln' \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document,.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/t5_index_mappings"
        sh "rm -rf examples/nlp/language_modeling/t5_pretrain_results"
        sh "rm -rf examples/nlp/language_modeling/t5_index_mappings"
      }
    }

    stage('L2: Megatron UL2 Pretraining and Resume Training TP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_t5_pretraining.py -cn megatron_ul2_config \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/t5_pretrain_results \
        model.tensor_model_parallel_size=2 \
        model.seq_length=128 \
        model.encoder.num_layers=4 \
        model.encoder.hidden_size=64 \
        model.encoder.num_attention_heads=8 \
        model.encoder.activation='swiglu' \
        model.encoder.bias_activation_fusion=False \
        model.encoder.activations_checkpoint_method='block' \
        model.encoder.activations_checkpoint_num_layers=1 \
        model.encoder.transformer_block_type='normformer' \
        model.encoder.headscale=True \
        model.decoder.num_layers=4 \
        model.decoder.hidden_size=64 \
        model.decoder.num_attention_heads=8 \
        model.decoder.activation='geglu' \
        model.decoder.bias_activation_fusion=False \
        model.decoder.activations_checkpoint_method='block' \
        model.decoder.activations_checkpoint_num_layers=1 \
        model.decoder.transformer_block_type='normformer' \
        model.decoder.headscale=False \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document,.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/t5_index_mappings"
        sh "python examples/nlp/language_modeling/megatron_t5_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=1 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/t5_pretrain_results \
        exp_manager.resume_if_exists=True \
        model.tensor_model_parallel_size=2 \
        model.seq_length=128 \
        model.encoder.num_layers=4 \
        model.encoder.hidden_size=64 \
        model.encoder.num_attention_heads=8 \
        model.encoder.activation='swiglu' \
        model.encoder.bias_activation_fusion=False \
        model.encoder.activations_checkpoint_method='block' \
        model.encoder.activations_checkpoint_num_layers=1 \
        model.encoder.transformer_block_type='normformer' \
        model.encoder.headscale=True \
        model.decoder.num_layers=4 \
        model.decoder.hidden_size=64 \
        model.decoder.num_attention_heads=8 \
        model.decoder.activation='geglu' \
        model.decoder.bias_activation_fusion=False \
        model.decoder.activations_checkpoint_method='block' \
        model.decoder.activations_checkpoint_num_layers=1 \
        model.decoder.transformer_block_type='normformer' \
        model.decoder.headscale=False \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document,.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document] \
        model.data.index_mapping_dir=examples/nlp/language_modeling/t5_index_mappings"
        sh "rm -rf examples/nlp/language_modeling/t5_pretrain_results"
        sh "rm -rf examples/nlp/language_modeling/t5_index_mappings"
      }
    }
    stage('L2: Megatron T5 Eval') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps{
        sh "python examples/nlp/language_modeling/megatron_t5_eval.py \
            --model_file \
            /home/TestData/nlp/megatron_t5/8m/megatron_t5_8m-refactor.nemo \
            --prompt \
            'How do I fix my GPU memory issue? I am seeing <mask> out of memory.' \
            --tensor_model_parallel_size 1"
      }
    }
    stage('L2: Megatron BART Pretraining and Resume Training, TP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_bart_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=2 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=3 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/bart_pretrain_results \
        model.tensor_model_parallel_size=2 \
        model.seq_length=128 \
        model.encoder.num_layers=4 \
        model.encoder.hidden_size=64 \
        model.encoder.num_attention_heads=8 \
        model.encoder.activation='reglu' \
        model.encoder.bias_activation_fusion=False \
        model.encoder.activations_checkpoint_method='block' \
        model.encoder.activations_checkpoint_num_layers=1 \
        model.decoder.num_layers=4 \
        model.decoder.hidden_size=64 \
        model.decoder.num_attention_heads=8 \
        model.decoder.activation='reglu' \
        model.decoder.bias_activation_fusion=False \
        model.decoder.activations_checkpoint_method='block' \
        model.decoder.activations_checkpoint_num_layers=1 \
        model.data.data_prefix='{train:[1.0,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document],test:[/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document], validation:[/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document]}'"
        sh "python examples/nlp/language_modeling/megatron_bart_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=2 \
        trainer.limit_val_batches=5 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=6 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/bart_pretrain_results \
        exp_manager.resume_if_exists=True \
        model.tensor_model_parallel_size=2 \
        model.seq_length=128 \
        model.encoder.num_layers=4 \
        model.encoder.hidden_size=64 \
        model.encoder.num_attention_heads=8 \
        model.encoder.activation='reglu' \
        model.encoder.bias_activation_fusion=False \
        model.encoder.activations_checkpoint_method='block' \
        model.encoder.activations_checkpoint_num_layers=1 \
        model.decoder.num_layers=4 \
        model.decoder.hidden_size=64 \
        model.decoder.num_attention_heads=8 \
        model.decoder.activation='reglu' \
        model.decoder.bias_activation_fusion=False \
        model.decoder.activations_checkpoint_method='block' \
        model.decoder.activations_checkpoint_num_layers=1 \
        model.data.data_prefix='{train:[1.0,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document],test:[/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document], validation:[/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document]}'"
        sh "rm -rf examples/nlp/language_modeling/bart_pretrain_results"
      }
    }
    stage('L2: Megatron BART Pretraining and Resume Training, PP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "python examples/nlp/language_modeling/megatron_bart_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=10 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/bart_pretrain_results \
        model.pipeline_model_parallel_size=2 \
        model.pipeline_model_parallel_split_rank=1 \
        model.seq_length=256 \
        model.encoder.num_layers=4 \
        model.encoder.hidden_size=64 \
        model.encoder.num_attention_heads=8 \
        model.encoder.activation='geglu' \
        model.encoder.bias_activation_fusion=False \
        model.encoder.activations_checkpoint_method='block' \
        model.encoder.activations_checkpoint_num_layers=1 \
        model.decoder.num_layers=4 \
        model.decoder.hidden_size=64 \
        model.decoder.num_attention_heads=8 \
        model.decoder.activation='geglu' \
        model.decoder.bias_activation_fusion=False \
        model.decoder.activations_checkpoint_method='block' \
        model.decoder.activations_checkpoint_num_layers=1 \
        model.data.respect_document_boundaries=False \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document,.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document]"
        sh "python examples/nlp/language_modeling/megatron_bart_pretraining.py \
        trainer.devices=2 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=1 \
        trainer.val_check_interval=1 \
        trainer.limit_val_batches=2 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=examples/nlp/language_modeling/bart_pretrain_results \
        exp_manager.resume_if_exists=True \
        model.pipeline_model_parallel_size=2 \
        model.pipeline_model_parallel_split_rank=1 \
        model.seq_length=256 \
        model.encoder.num_layers=4 \
        model.encoder.hidden_size=64 \
        model.encoder.num_attention_heads=8 \
        model.encoder.activation='geglu' \
        model.encoder.bias_activation_fusion=False \
        model.encoder.activations_checkpoint_method='block' \
        model.encoder.activations_checkpoint_num_layers=1 \
        model.decoder.num_layers=4 \
        model.decoder.hidden_size=64 \
        model.decoder.num_attention_heads=8 \
        model.decoder.activation='geglu' \
        model.decoder.bias_activation_fusion=False \
        model.decoder.activations_checkpoint_method='block' \
        model.decoder.activations_checkpoint_num_layers=1 \
        model.data.respect_document_boundaries=False \
        model.data.data_prefix=[.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document,.5,/home/TestData/nlp/megatron_t5/data/pile_val_small_bert_tokenizer_text_document]"
        sh "rm -rf examples/nlp/language_modeling/bart_pretrain_results"
      }
    }
    stage('L2: Megatron T5 GLUE/XNLI Finetuning') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        // TODO(Oktai15): update it in 1.8.0 version
        stage('T5 GLUE RTE') {
          steps {
            sh "python examples/nlp/language_modeling/megatron_t5_seq2seq_finetune.py \
            trainer.devices=1 \
            trainer.accelerator=gpu \
            trainer.log_every_n_steps=1 \
            trainer.val_check_interval=1 \
            +trainer.limit_val_batches=2 \
            +trainer.limit_test_batches=2 \
            trainer.accumulate_grad_batches=1 \
            trainer.max_steps=2 \
            trainer.precision=16 \
            exp_manager.exp_dir=examples/nlp/language_modeling/t5_glue_results \
            model.restore_from_path=/home/TestData/nlp/megatron_t5/8m/megatron_t5_8m-refactor.nemo \
            model.pipeline_model_parallel_size=1 \
            model.pipeline_model_parallel_split_rank=0 \
            model.data.train_ds.task_name=rte \
            model.data.train_ds.global_batch_size=4 \
            model.data.train_ds.micro_batch_size=2 \
            model.data.validation_ds.global_batch_size=2 \
            model.data.validation_ds.micro_batch_size=2 \
            model.data.train_ds.file_path=/home/TestData/nlp/megatron_t5/data/train_ci.tsv \
            model.data.validation_ds.task_name=rte \
            model.data.validation_ds.file_path=/home/TestData/nlp/megatron_t5/data/dev_ci.tsv \
            "
            sh "rm -rf examples/nlp/language_modeling/t5_glue_results"
          }
        }
        stage('T5 GLUE XNLI') {
          steps {
            sh "python examples/nlp/language_modeling/megatron_t5_seq2seq_finetune.py \
            -cn megatron_t5_config_finetune_glue_xnli \
            trainer.devices=1 \
            trainer.accelerator=gpu \
            trainer.log_every_n_steps=1 \
            trainer.val_check_interval=1 \
            +trainer.limit_val_batches=2 \
            +trainer.limit_test_batches=2 \
            trainer.accumulate_grad_batches=1 \
            trainer.max_steps=2 \
            trainer.precision=16 \
            exp_manager.exp_dir=examples/nlp/language_modeling/t5_xnli_results \
            model.restore_from_path=/home/TestData/nlp/megatron_t5/8m/megatron_t5_8m-refactor.nemo \
            model.pipeline_model_parallel_size=1 \
            model.pipeline_model_parallel_split_rank=0 \
            model.data.train_ds.global_batch_size=4 \
            model.data.train_ds.micro_batch_size=2 \
            model.data.validation_ds.global_batch_size=2 \
            model.data.validation_ds.micro_batch_size=2 \
            model.data.test_ds.global_batch_size=2 \
            model.data.test_ds.micro_batch_size=2 \
            model.data.train_ds.task_name=rte \
            model.data.train_ds.file_path=/home/TestData/nlp/megatron_t5/data/train_ci.tsv \
            model.data.validation_ds.task_name=xnli \
            model.data.validation_ds.file_path=/home/TestData/nlp/megatron_t5/data/xnli_dev_ci.tsv \
            model.data.test_ds.task_name=xnli \
            model.data.test_ds.file_path=/home/TestData/nlp/megatron_t5/data/xnli_dev_ci.tsv \
            "
            sh "rm -rf examples/nlp/language_modeling/t5_xnli_results"
          }
        }
      }
    }

    stage('L2: Megatron T5 PEFT Lora TP=2') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      steps {
        sh "rm -rf /home/TestData/nlp/t5_lora_tuning_tp2"
        sh "python examples/nlp/language_modeling/tuning/megatron_t5_peft_tuning.py \
        trainer.devices=2 \
        trainer.log_every_n_steps=1 \
        trainer.max_epochs=9999 \
        trainer.max_steps=3 \
        trainer.val_check_interval=3 \
        ++trainer.limit_val_batches=2 \
        trainer.precision=16 \
        exp_manager.exp_dir=/home/TestData/nlp/t5_lora_tuning_tp2 \
        model.pipeline_model_parallel_size=1 \
        model.tensor_model_parallel_size=2 \
        model.restore_from_path=/home/TestData/nlp/megatron_t5/8m/megatron_t5_8m_tp2.nemo \
        model.peft.peft_scheme='lora' \
        model.answer_only_loss=True \
        model.micro_batch_size=1 \
        model.global_batch_size=1 \
        model.data.train_ds.file_names=[/home/TestData/nlp/megatron_sft/quarel.jsonl] \
        model.data.train_ds.concat_sampling_probabilities=[1.0] \
        model.data.train_ds.num_workers=0 \
        model.data.validation_ds.num_workers=0 \
        model.data.validation_ds.file_names=[/home/TestData/nlp/megatron_sft/quarel.jsonl] \
        model.data.validation_ds.names=[quarel]"
        sh "python examples/nlp/language_modeling/tuning/megatron_t5_peft_eval.py \
        model.restore_from_path=/home/TestData/nlp/megatron_t5/8m/megatron_t5_8m_tp2.nemo \
        model.peft.restore_from_path=/home/TestData/nlp/t5_lora_tuning_tp2/megatron_t5_peft_lora_tuning/checkpoints/megatron_t5_peft_lora_tuning.nemo \
        model.peft.restore_from_ckpt_name=null \
        model.peft.restore_from_hparams_path=null \
        model.tensor_model_parallel_size=2 \
        trainer.devices=2 \
        model.data.test_ds.file_names=[/home/TestData/nlp/megatron_sft/quarel_4.jsonl] \
        model.data.test_ds.names=['quarel4'] \
        model.global_batch_size=2 \
        model.micro_batch_size=1 \
        model.data.test_ds.tokens_to_generate=10 \
        model.data.test_ds.write_predictions_to_file=True \
        model.data.test_ds.output_file_path_prefix='/home/TestData/nlp/t5_lora_tuning_tp2/out' \
        inference.greedy=True \
        inference.repetition_penalty=1.0 \
        inference.outfile_path='/home/TestData/nlp/t5_lora_tuning_tp2/out.jsonl'"
        sh "rm -rf /home/TestData/nlp/t5_lora_tuning_tp2"
      }
    }


    stage('L2: Megatron Mock Data Generation') {
      when {
        anyOf {
          branch 'main'
          changeRequest target: 'main'
        }
      }
      failFast true
      parallel {
        stage('MockGPTDataset') {
          steps {
            sh "python examples/nlp/language_modeling/megatron_gpt_pretraining.py \
            trainer.max_steps=10 \
            trainer.limit_val_batches=7 \
            trainer.val_check_interval=10 \
            exp_manager.exp_dir=examples/nlp/language_modeling/gpt_pretrain_results \
            model.data.data_impl=mock \
            model.data.data_prefix=[] \
            "
            sh "rm -rf examples/nlp/language_modeling/gpt_pretrain_results"
          }
        }
        stage('MockT5Dataset') {
          steps {
            sh "python examples/nlp/language_modeling/megatron_t5_pretraining.py \
            trainer.max_steps=10 \
            trainer.limit_val_batches=3 \
            trainer.val_check_interval=10 \
            exp_manager.exp_dir=examples/nlp/language_modeling/t5_pretrain_results \
            model.data.data_impl=mock \
            model.data.data_prefix=[] \
            "
            sh "rm -rf examples/nlp/language_modeling/t5_pretrain_results"
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
            trainer.devices=[0] \
            trainer.accelerator="gpu" \
            +trainer.limit_train_batches=1 +trainer.limit_val_batches=1 trainer.max_epochs=1 \
            trainer.strategy=auto \
            model.decoder.decoder_rnn_dim=256 \
            model.decoder.attention_rnn_dim=1024 \
            model.decoder.prenet_dim=128 \
            model.postnet.postnet_n_convolutions=3 \
            model.train_ds.dataloader_params.batch_size=4 \
            model.train_ds.dataloader_params.num_workers=0 \
            model.validation_ds.dataloader_params.batch_size=4 \
            model.validation_ds.dataloader_params.num_workers=0 \
            ~model.text_normalizer \
            ~model.text_normalizer_call_kwargs \
            ~trainer.check_val_every_n_epoch \
            '
          }
        }
        stage('WaveGlow') {
          steps {
            sh 'python examples/tts/waveglow.py \
            train_dataset=/home/TestData/an4_dataset/an4_train.json \
            validation_datasets=/home/TestData/an4_dataset/an4_val.json \
            trainer.devices="[0]" \
            +trainer.limit_train_batches=1 +trainer.limit_val_batches=1 trainer.max_epochs=1 \
            trainer.strategy=auto \
            model.train_ds.dataloader_params.batch_size=4 \
            model.train_ds.dataloader_params.num_workers=0 \
            model.validation_ds.dataloader_params.batch_size=4 \
            model.validation_ds.dataloader_params.num_workers=0 \
            model.waveglow.n_flows=4 \
            model.waveglow.n_wn_layers=2 \
            model.waveglow.n_wn_channels=32 \
            ~trainer.check_val_every_n_epoch'
          }
        }
        stage('FastPitch') {
          steps {
            sh 'python examples/tts/fastpitch.py \
            --config-name fastpitch_align_v1.05 \
            train_dataset=/home/TestData/an4_dataset/an4_train.json \
            validation_datasets=/home/TestData/an4_dataset/an4_val.json \
            sup_data_path=/home/TestData/an4_dataset/beta_priors \
            trainer.devices="[0]" \
            +trainer.limit_train_batches=1 \
            +trainer.limit_val_batches=1 \
            trainer.max_epochs=1 \
            trainer.strategy=auto \
            model.pitch_mean=212.35873413085938 \
            model.pitch_std=68.52806091308594 \
            model.train_ds.dataloader_params.batch_size=4 \
            model.train_ds.dataloader_params.num_workers=0 \
            model.validation_ds.dataloader_params.batch_size=4 \
            model.validation_ds.dataloader_params.num_workers=0 \
            model.symbols_embedding_dim=64 \
            model.input_fft.d_inner=384 \
            model.input_fft.n_layer=2 \
            model.output_fft.d_inner=384 \
            model.output_fft.n_layer=2 \
            ~trainer.check_val_every_n_epoch \
            ~model.text_normalizer \
            ~model.text_normalizer_call_kwargs'
          }
        }
        stage('RADTTS') {
          steps {
            sh 'python examples/tts/radtts.py \
            train_dataset=/home/TestData/an4_dataset/an4_train.json \
            validation_datasets=/home/TestData/an4_dataset/an4_val.json \
            sup_data_path=/home/TestData/an4_dataset/radtts_beta_priors \
            trainer.devices="[0]" \
            +trainer.limit_train_batches=1 \
            +trainer.limit_val_batches=1 \
            trainer.max_epochs=1 \
            trainer.strategy=auto \
            model.pitch_mean=212.35873413085938 \
            model.pitch_std=68.52806091308594 \
            model.train_ds.dataloader_params.batch_size=4 \
            model.train_ds.dataloader_params.num_workers=0 \
            model.validation_ds.dataloader_params.batch_size=4 \
            model.validation_ds.dataloader_params.num_workers=0 \
            export_dir=/home/TestData/radtts_test \
            model.optim.lr=0.0001 \
            model.modelConfig.decoder_use_partial_padding=True \
            ~trainer.check_val_every_n_epoch \
            ~model.text_normalizer \
            ~model.text_normalizer_call_kwargs'
          }
        }
        stage('Mixer-TTS') {
          steps {
            sh 'python examples/tts/mixer_tts.py \
            train_dataset=/home/TestData/an4_dataset/an4_train.json \
            validation_datasets=/home/TestData/an4_dataset/an4_val.json \
            sup_data_path=/home/TestData/an4_dataset/sup_data \
            trainer.devices="[0]" \
            +trainer.limit_train_batches=1 \
            +trainer.limit_val_batches=1 \
            trainer.max_epochs=1 \
            trainer.strategy=auto \
            model.pitch_mean=212.35873413085938 \
            model.pitch_std=68.52806091308594 \
            model.train_ds.dataloader_params.batch_size=4 \
            model.train_ds.dataloader_params.num_workers=0 \
            model.validation_ds.dataloader_params.batch_size=4 \
            model.validation_ds.dataloader_params.num_workers=0 \
            ~trainer.check_val_every_n_epoch \
            ~model.text_normalizer \
            ~model.text_normalizer_call_kwargs'
          }
        }
        stage('Hifigan') {
          steps {
            sh 'python examples/tts/hifigan.py \
            train_dataset=/home/TestData/an4_dataset/an4_train.json \
            validation_datasets=/home/TestData/an4_dataset/an4_val.json \
            trainer.devices="[0]" \
            +trainer.limit_train_batches=1 \
            +trainer.limit_val_batches=1 \
            +trainer.max_epochs=1 \
            trainer.strategy=auto \
            model.train_ds.dataloader_params.batch_size=4 \
            model.train_ds.dataloader_params.num_workers=0 \
            model.validation_ds.dataloader_params.batch_size=4 \
            model.validation_ds.dataloader_params.num_workers=0 \
            model.generator.upsample_initial_channel=64 \
            +model.debug=true \
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
      steps {
        sh 'CUDA_VISIBLE_DEVICES=0 python examples/asr/speech_to_text_eval.py \
            pretrained_name=QuartzNet15x5Base-En  \
            dataset_manifest=/home/TestData/librispeech/librivox-dev-other.json \
            batch_size=64 \
            tolerance=0.1012'
        sh 'rm -f examples/asr/evaluation_transcripts.json'
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
