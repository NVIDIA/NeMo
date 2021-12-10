#ifndef CTC_BEAM_SEARCH_DECODER_H_
#define CTC_BEAM_SEARCH_DECODER_H_

#include <string>
#include <utility>
#include <vector>

#include "scorer.h"

/* CTC Beam Search Decoder

 * Parameters:
 *     probs_seq: 2-D vector that each element is a vector of probabilities
 *               over vocabulary of one time step.
 *     vocabulary: A vector of vocabulary.
 *     beam_size: The width of beam search.
 *     cutoff_prob: Cutoff probability for pruning.
 *     cutoff_top_n: Cutoff number for pruning.
 *     ext_scorer: External scorer to evaluate a prefix, which consists of
 *                 n-gram language model scoring and word insertion term.
 *                 Default null, decoding the input sample without scorer.
 * Return:
 *     A vector that each element is a pair of score  and decoding result,
 *     in desending order.
*/
std::vector<std::pair<double, std::string>> ctc_beam_search_decoder(
    const std::vector<std::vector<double>> &probs_seq,
    const std::vector<std::string> &vocabulary,
    size_t beam_size,
    double cutoff_prob = 1.0,
    size_t cutoff_top_n = 40,
    Scorer *ext_scorer = nullptr);


class BeamDecoder {
public:
  BeamDecoder(const std::vector<std::string> &vocabulary,
         size_t beam_size,
         double cutoff_prob = 1.0,
         size_t cutoff_top_n = 40,
         Scorer *ext_scorer = nullptr);
  ~BeamDecoder();

  // decode a frame
  std::vector<std::pair<double, std::string>> decode(const std::vector<std::vector<double>> &probs_seq);

  void get_word_timestamps(
      std::vector<std::tuple<std::string, uint32_t, uint32_t>>& words);

  void add_start_offset(int offset) { time_offset += offset; }
  void set_start_offset(int offset) { time_offset = offset; }

  // reset state
  void reset(bool keep_offset = false, bool keep_words = false);

private:
  Scorer *ext_scorer;
  size_t beam_size;
  double cutoff_prob;
  size_t cutoff_top_n;

  // state
  std::vector<std::string> vocabulary;
  size_t blank_id;
  int space_id;
  // for word timestamps
  int time_offset; // time offset to be added to all prefixes
  int prev_time_offset; // time offset of previous decode. Needed when decoding
                        // multiple sentences with VAD.
  int last_decoded_timestep; // timestep of the last parsed prefix
  std::vector<std::tuple<std::string, uint32_t, uint32_t>> prev_wordlist;
  std::vector<std::tuple<std::string, uint32_t, uint32_t>> wordlist;

  PathTrie *root;
  std::vector<PathTrie *> prefixes;
};



/* CTC Beam Search Decoder for batch data

 * Parameters:
 *     probs_seq: 3-D vector that each element is a 2-D vector that can be used
 *                by ctc_beam_search_decoder().
 *     vocabulary: A vector of vocabulary.
 *     beam_size: The width of beam search.
 *     num_processes: Number of threads for beam search.
 *     cutoff_prob: Cutoff probability for pruning.
 *     cutoff_top_n: Cutoff number for pruning.
 *     ext_scorer: External scorer to evaluate a prefix, which consists of
 *                 n-gram language model scoring and word insertion term.
 *                 Default null, decoding the input sample without scorer.
 * Return:
 *     A 2-D vector that each element is a vector of beam search decoding
 *     result for one audio sample.
*/
std::vector<std::vector<std::pair<double, std::string>>>
ctc_beam_search_decoder_batch(
    const std::vector<std::vector<std::vector<double>>> &probs_split,
    const std::vector<std::string> &vocabulary,
    size_t beam_size,
    size_t num_processes,
    double cutoff_prob = 1.0,
    size_t cutoff_top_n = 40,
    Scorer *ext_scorer = nullptr);

#endif  // CTC_BEAM_SEARCH_DECODER_H_
