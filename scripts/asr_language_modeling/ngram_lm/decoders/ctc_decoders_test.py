import numpy as np
import pickle
import unittest

from ctc_decoders import Scorer, ctc_beam_search_decoder


def load_test_sample(pickle_file):
  with open(pickle_file, 'rb') as f:
    seq, label = pickle.load(f, encoding='bytes')
  return seq, label 


def softmax(x):
  m = np.expand_dims(np.max(x, axis=-1), -1)
  e = np.exp(x - m)
  return e / np.expand_dims(e.sum(axis=-1), -1)


class CTCCustomDecoderTests(unittest.TestCase):

  def setUp(self):
    self.seq, self.label = load_test_sample('ctc-test.pickle')
    self.vocab = [
        " ",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "'"
    ]
    self.beam_width = 16
    self.tol = 1e-3


  def test_decoders(self):
    '''
    Test custom CTC decoder with LM rescoring. It should yield 'ten seconds'.
    '''
    logits = self.seq
    seq_len = [self.seq.shape[0]]

    scorer = Scorer(alpha=2.0, beta=0.5,
        model_path='ctc-test-lm.binary', 
        vocabulary=self.vocab)
    res = ctc_beam_search_decoder(softmax(self.seq.squeeze()), self.vocab,
                                  beam_size=self.beam_width,
                                  ext_scoring_func=scorer)
    res_prob, decoded_text = res[0]
    self.assertTrue( abs(4.0845 + res_prob) < self.tol )
    self.assertTrue( decoded_text == self.label )


if __name__ == '__main__':
  unittest.main()

