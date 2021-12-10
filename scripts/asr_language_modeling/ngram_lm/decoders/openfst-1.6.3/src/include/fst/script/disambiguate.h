// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_DISAMBIGUATE_H_
#define FST_SCRIPT_DISAMBIGUATE_H_

#include <fst/disambiguate.h>
#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>
#include <fst/script/weight-class.h>

namespace fst {
namespace script {

// 1: Full signature.
struct DisambiguateOptions {
  float delta;
  const WeightClass &weight_threshold;
  int64 state_threshold;
  int64 subsequential_label;

  DisambiguateOptions(float d, const WeightClass &w,
                      int64 n = fst::kNoStateId, int64 l = 0)
      : delta(d), weight_threshold(w), state_threshold(n),
        subsequential_label(l) {}
};

using DisambiguateArgs1 = args::Package<const FstClass &, MutableFstClass *,
                                        const DisambiguateOptions &>;

template <class Arc>
void Disambiguate(DisambiguateArgs1 *args) {
  const Fst<Arc> &ifst = *(args->arg1.GetFst<Arc>());
  MutableFst<Arc> *ofst = args->arg2->GetMutableFst<Arc>();
  const DisambiguateOptions &opts = args->arg3;
  typename Arc::Weight weight_threshold =
      *(opts.weight_threshold.GetWeight<typename Arc::Weight>());
  fst::DisambiguateOptions<Arc> disargs(opts.delta, weight_threshold,
                                            opts.state_threshold,
                                            opts.subsequential_label);
  Disambiguate(ifst, ofst, disargs);
}

// 2: Signature with default WeightClass argument.
using DisambiguateArgs2 =
    args::Package<const FstClass &, MutableFstClass *, int64, int64>;

template <class Arc>
void Disambiguate(DisambiguateArgs2 *args) {
  const Fst<Arc> &ifst = *(args->arg1.GetFst<Arc>());
  MutableFst<Arc> *ofst = args->arg2->GetMutableFst<Arc>();
  typename Arc::Weight weight_threshold = Arc::Weight::Zero();
  fst::DisambiguateOptions<Arc> disargs(kDelta, weight_threshold,
                                            args->arg3, args->arg4);
  Disambiguate(ifst, ofst, disargs);
}

// 1
void Disambiguate(const FstClass &ifst, MutableFstClass *ofst,
                  const DisambiguateOptions &opts);

// 2
void Disambiguate(const FstClass &, MutableFstClass *ofst,
                  int64 n = fst::kNoStateId, int64 l = 0);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_DISAMBIGUATE_H_
