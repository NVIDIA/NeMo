// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_DETERMINIZE_H_
#define FST_SCRIPT_DETERMINIZE_H_

#include <fst/determinize.h>
#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>
#include <fst/script/weight-class.h>

namespace fst {
namespace script {

// 1: Full signature with DeterminizeOptions.
struct DeterminizeOptions {
  float delta;
  const WeightClass &weight_threshold;
  int64 state_threshold;
  int64 subsequential_label;
  DeterminizeType type;
  bool increment_subsequential_label;

  DeterminizeOptions(float d, const WeightClass &w,
                     int64 n = fst::kNoStateId, int64 l = 0,
                     DeterminizeType t = DETERMINIZE_FUNCTIONAL, bool i = false)
      : delta(d),
        weight_threshold(w),
        state_threshold(n),
        subsequential_label(l),
        type(t),
        increment_subsequential_label(i) {}
};

using DeterminizeArgs1 = args::Package<const FstClass &, MutableFstClass *,
                                       const DeterminizeOptions &>;

template <class Arc>
void Determinize(DeterminizeArgs1 *args) {
  const Fst<Arc> &ifst = *(args->arg1.GetFst<Arc>());
  MutableFst<Arc> *ofst = args->arg2->GetMutableFst<Arc>();
  const DeterminizeOptions &opts = args->arg3;
  typename Arc::Weight weight_threshold =
      *(opts.weight_threshold.GetWeight<typename Arc::Weight>());
  fst::DeterminizeOptions<Arc> detargs(
      opts.delta, weight_threshold, opts.state_threshold,
      opts.subsequential_label, opts.type, opts.increment_subsequential_label);
  Determinize(ifst, ofst, detargs);
}

// 2: Signature with default WeightClass argument.
using DeterminizeArgs2 =
    args::Package<const FstClass &, MutableFstClass *, float, int64, int64,
                  DeterminizeType, bool>;

template <class Arc>
void Determinize(DeterminizeArgs2 *args) {
  const Fst<Arc> &ifst = *(args->arg1.GetFst<Arc>());
  MutableFst<Arc> *ofst = args->arg2->GetMutableFst<Arc>();
  typename Arc::Weight weight_threshold = Arc::Weight::Zero();
  fst::DeterminizeOptions<Arc> detargs(args->arg3, weight_threshold,
                                           args->arg4, args->arg5, args->arg6,
                                           args->arg7);
  Determinize(ifst, ofst, detargs);
}

// 1
void Determinize(const FstClass &ifst, MutableFstClass *ofst,
                 const DeterminizeOptions &opts);

// 2
void Determinize(const FstClass &ifst, MutableFstClass *ofst,
                 float d = fst::kDelta, int64 n = fst::kNoStateId,
                 int64 l = 0, DeterminizeType t = DETERMINIZE_FUNCTIONAL,
                 bool i = false);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_DETERMINIZE_H_
