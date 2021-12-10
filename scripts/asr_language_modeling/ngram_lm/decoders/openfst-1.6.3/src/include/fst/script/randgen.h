// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_RANDGEN_H_
#define FST_SCRIPT_RANDGEN_H_

#include <ctime>

#include <fst/randgen.h>
#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

using RandGenArgs = args::Package<const FstClass &, MutableFstClass *, time_t,
                                  const RandGenOptions<RandArcSelection> &>;

template <class Arc>
void RandGen(RandGenArgs *args) {
  const Fst<Arc> &ifst = *(args->arg1.GetFst<Arc>());
  MutableFst<Arc> *ofst = args->arg2->GetMutableFst<Arc>();
  time_t seed = args->arg3;
  const RandGenOptions<RandArcSelection> &opts = args->arg4;
  if (opts.selector == UNIFORM_ARC_SELECTOR) {
    const UniformArcSelector<Arc> selector(seed);
    const RandGenOptions<UniformArcSelector<Arc>> ropts(
        selector, opts.max_length, opts.npath, opts.weighted,
        opts.remove_total_weight);
    RandGen(ifst, ofst, ropts);
  } else if (opts.selector == FAST_LOG_PROB_ARC_SELECTOR) {
    const FastLogProbArcSelector<Arc> selector(seed);
    const RandGenOptions<FastLogProbArcSelector<Arc>> ropts(
        selector, opts.max_length, opts.npath, opts.weighted,
        opts.remove_total_weight);
    RandGen(ifst, ofst, ropts);
  } else {
    const LogProbArcSelector<Arc> selector(seed);
    const RandGenOptions<LogProbArcSelector<Arc>> ropts(
        selector, opts.max_length, opts.npath, opts.weighted,
        opts.remove_total_weight);
    RandGen(ifst, ofst, ropts);
  }
}

void RandGen(const FstClass &ifst, MutableFstClass *ofst,
             time_t seed = time(nullptr),
             const RandGenOptions<RandArcSelection> &opts =
                 RandGenOptions<RandArcSelection>(UNIFORM_ARC_SELECTOR));

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_RANDGEN_H_
