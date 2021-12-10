#ifndef CTC_GREEDY_DECODER_H
#define CTC_GREEDY_DECODER_H

#include <string>
#include <vector>

/* CTC Greedy (Best Path) Decoder
 *
 * Parameters:
 *     probs_seq: 2-D vector that each element is a vector of probabilities
 *               over vocabulary of one time step.
 *     vocabulary: A vector of vocabulary.
 * Return:
 *     The decoding result in string
 */
std::string ctc_greedy_decoder(
    const std::vector<std::vector<double>>& probs_seq,
    const std::vector<std::string>& vocabulary);

#endif  // CTC_GREEDY_DECODER_H
