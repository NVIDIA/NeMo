pipeline {
  agent any
  environment {
      PATH="/home/mrjenkins/anaconda3/envs/py37p1.12c10/bin:$PATH"
  }
  stages {
    stage('PEP8 Checks') {
      steps {
        sh 'pycodestyle . --exclude=./scripts/get_librispeech_data.py,./scripts/process_beam_dump.py,./examples/nlp/end_of_sentence_tagging_with_bert.py,./examples/nlp/squad_with_pretrained_bert.py,./examples/nlp/transformer_translation.py,./tests/other/jasper.py,./tests/other/jasper_zero_dl.py,./collections/nemo_nlp/nemo_nlp/data/datasets/question_answering.py,./collections/nemo_nlp/nemo_nlp/data/datasets/token_classification.py,./collections/nemo_nlp/nemo_nlp/externals/bleu.py,./collections/nemo_nlp/nemo_nlp/externals/fairseq_tokenizer.py,./collections/nemo_nlp/nemo_nlp/externals/file_utils.py,./collections/nemo_nlp/nemo_nlp/externals/run_squad.py,./collections/nemo_nlp/nemo_nlp/externals/sacrebleu.py,./collections/nemo_nlp/nemo_nlp/externals/tokenization.py,./nemo/setup.py,./docs/sources/source/conf.py,./docs/sources/source/tutorials/infer.py,./docs/sources/source/tutorials/test.py,./collections/nemo_nlp/build'
      }
    } 
    stage('Unittests') {
      steps {
        sh './reinstall.sh && python -m unittest tests/*.py'
      }
    }
    parallel {
      stage('Simplest test') {
        steps {
          sh 'cd examples/start_here && python simplest_example.py'        
        }
      }
      stage ('Chatbot test') {
        steps {
          sh 'cd examples/start_here && python chatbot_example.py'
        }
      }
   }
  }
}
