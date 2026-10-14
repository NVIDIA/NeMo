Phoneme loss weighting for normalization vocabulary
==================================================

EasyMagpie can amplify phoneme cross-entropy for complete word/phrase
pronunciations from a YAML lexicon. This supports text-only pretraining and
speech batches using ``MagpieTTSLhotseMultiturnDataset``. Enable it with:

.. code-block:: bash

   python examples/tts/easy_magpietts.py \
     model.phoneme_loss_lexicon_path=/absolute/path/phoneme_loss_lexicon_en.yaml \
     model.phoneme_lexicon_loss_scale=2.0

The supplied English lexicon is
``examples/tts/conf/magpietts/phoneme_loss_lexicon_en.yaml``. It records each
written word/phrase, its categories, and EN-US IPA pronunciations generated with
eSpeak-NG 1.52. It covers compositional number vocabulary, ordinals, fractions,
dates/times, currencies, units, symbols, spelled letters, and common URL/email
reading terms. Arbitrary domain labels, names, identifiers, and foreign words
are an open vocabulary; the file is extensible and cannot enumerate every
possible literal entity. The pronunciation vocabulary is EN-US; matching supports
English language IDs, including ``en``, ``en-US``, and ``en-GB``.

Written entries use canonical forms: ``B`` and ``E`` contain their letter-name
IPA; respellings such as ``bee`` and ``ee`` are excluded. ``AM``, ``PM``, and
calendar-era abbreviations are pronounced as spelled letters. Generic context
labels such as ``model``, ``website``, and ``phone`` are excluded because they
are not normalization expansions. Genuine spoken normalization terms, such as
``point``, ``dollars``, ``extension``, and ``underscore``, remain included.
Removing letter-name respellings does not distinguish the IPA of letter B from
ordinary "bee"; canonical letters still match their complete pronunciations.

The YAML schema is:

.. code-block:: yaml

   schema_version: 1
   languages: [en]
   entries:
     - word: million
       categories: [cardinal, currency]
       pronunciations:
         en-US: ["mˈɪliən"]

Model construction loads the YAML once, while restored artifacts still exist.
Workers receive the compiled CPU lexicon and tokenize its variants once. They match complete words/phrases
in the ground-truth IPA, ignoring primary/secondary stress, and project matching
character spans onto the full-sentence BPE tokenizer offsets. This handles
context-dependent BPE segmentation without matching individual phonemes in
unrelated words. A BPE token crossing a match boundary receives the multiplier
if any of its characters overlap the match. Overlapping matches receive the
scale once. No eSpeak calls, decoding, CPU transfers, or string matching occur
inside the GPU loss computation.

The batch carries a Boolean mask aligned with its original phoneme tokens.
The model stacks and shifts this mask exactly like the phoneme targets.
BOS, EOS, padding, and ignored languages do not receive extra weight.
Positions removed by dropout or truncation carry no extra weight. Existing
loss masks and corruption/dropout rules still apply.
The YAML is registered as a model artifact and is included in saved ``.nemo``
models when the feature is enabled.

The loss uses the original valid-token denominator:

.. math::

   L = \frac{\sum_i m_i [1 + (s - 1) q_i] \operatorname{CE}_i}{\sum_i m_i}

Here ``q`` is the lexicon match mask, ``m`` is the existing effective loss mask,
and ``s`` is ``phoneme_lexicon_loss_scale``. Thus a scale of two doubles the
loss and gradient contribution at matching positions. It also increases the
overall training loss; ``phoneme_loss_weight`` remains a separate multiplier
for the entire phoneme loss.

The default scale is ``1.0`` and the default path is ``null``. This bypasses
lexicon loading/matching and preserves the original loss. Scales must be finite
and at least one. Validation remains unweighted, and inference behavior and
model parameters are unchanged.

When enabled, Lightning logs these metrics to the configured logger, including
Weights & Biases:

* ``train/phoneme_lexicon_matched_tokens``: target tokens matching the lexicon,
  before effective loss filters.
* ``train/phoneme_amplified_tokens``: matched tokens whose loss was actually
  multiplied, after the existing loss masks and dropout/corruption rules.
* ``train/phoneme_loss_tokens``: positions included in the effective phoneme
  loss denominator, including special tokens that the existing loss supervises.
* ``train/phoneme_amplified_percent``: amplified tokens divided by effective
  loss positions, times 100. This is zero when the phoneme loss is skipped.

Counts are summed across distributed ranks using one detached three-integer
all-reduce per training batch; the percentage uses the global numerator and
denominator. Existing logging cadence controls when the values appear in the
logger. Disabled weighting adds no coverage reductions or logging. A nonzero
matched count with zero amplified count indicates that existing loss rules
excluded those matches. These counters measure phoneme lexicon matches, not
semantic entity precision/recall.

IPA homophones such as two/to/too, four/for, eight/ate, letter U/you, and
month May/modal may cannot be distinguished using a phoneme lexicon alone.
Fused pronunciations, unlisted variants, and inconsistent target labels can
also limit matching. This weighting does not provide future text lookahead.
Evaluate numeric value preservation alongside pronunciation metrics before
concluding that an increased scale improves normalization quality.
