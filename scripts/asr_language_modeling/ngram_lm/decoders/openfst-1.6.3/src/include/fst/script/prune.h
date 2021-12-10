// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_PRUNE_H_
#define FST_SCRIPT_PRUNE_H_

#include <memory>
#include <vector>

#include <fst/arcfilter.h>
#include <fst/prune.h>
#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>
#include <fst/script/weight-class.h>

namespace fst {
namespace script {

struct PruneOptions {
  const WeightClass &weight_threshold;
  int64 state_threshold;
  std::unique_ptr<const std::vector<WeightClass>> distance;
  float delta;

  PruneOptions(const WeightClass &w, int64 s,
               std::vector<WeightClass> *d = nullptr, float e = kDelta)
      : weight_threshold(w), state_threshold(s), distance(d), delta(e) {}

 private:
  PruneOptions();  // Forbids this.
};

// This function transforms a script-land PruneOptions into a lib-land
// PruneOptions. If the original opts.distance is not null, a new distance will
// be created with new; the client is responsible for deleting this.

template <class Arc>
fst::PruneOptions<Arc, AnyArcFilter<Arc>> ConvertPruneOptions(
    const PruneOptions &opts) {
  using Weight = typename Arc::Weight;
  const auto weight_threshold = *(opts.weight_threshold.GetWeight<Weight>());
  std::unique_ptr<std::vector<Weight>> distance;
  if (opts.distance) {
    distance.reset(new std::vector<Weight>(opts.distance->size()));
    for (auto i = 0; i < opts.distance->size(); ++i)
      (*distance)[i] = *((*opts.distance)[i].GetWeight<Weight>());
  }
  return fst::PruneOptions<Arc, AnyArcFilter<Arc>>(
      weight_threshold, opts.state_threshold, AnyArcFilter<Arc>(),
      distance.get(), opts.delta);
}

// 1
using PruneArgs1 = args::Package<MutableFstClass *, const PruneOptions &>;

template <class Arc>
void Prune(PruneArgs1 *args) {
  MutableFst<Arc> *ofst = args->arg1->GetMutableFst<Arc>();
  fst::PruneOptions<Arc, AnyArcFilter<Arc>> opts =
      ConvertPruneOptions<Arc>(args->arg2);
  Prune(ofst, opts);
}

// 2
using PruneArgs2 =
    args::Package<const FstClass &, MutableFstClass *, const PruneOptions &>;

template <class Arc>
void Prune(PruneArgs2 *args) {
  const Fst<Arc> &ifst = *(args->arg1.GetFst<Arc>());
  MutableFst<Arc> *ofst = args->arg2->GetMutableFst<Arc>();
  fst::PruneOptions<Arc, AnyArcFilter<Arc>> opts =
      ConvertPruneOptions<Arc>(args->arg3);
  Prune(ifst, ofst, opts);
}

// 3
using PruneArgs3 = args::Package<const FstClass &, MutableFstClass *,
                                 const WeightClass &, int64, float>;

template <class Arc>
void Prune(PruneArgs3 *args) {
  const Fst<Arc> &ifst = *(args->arg1.GetFst<Arc>());
  MutableFst<Arc> *ofst = args->arg2->GetMutableFst<Arc>();
  typename Arc::Weight weight_threshold =
      *(args->arg3.GetWeight<typename Arc::Weight>());
  Prune(ifst, ofst, weight_threshold, args->arg4, args->arg5);
}

// 4
using PruneArgs4 =
    args::Package<MutableFstClass *, const WeightClass &, int64, float>;

template <class Arc>
void Prune(PruneArgs4 *args) {
  MutableFst<Arc> *fst = args->arg1->GetMutableFst<Arc>();
  typename Arc::Weight weight_threshold =
      *(args->arg2.GetWeight<typename Arc::Weight>());
  Prune(fst, weight_threshold, args->arg3, args->arg4);
}

// 1
void Prune(MutableFstClass *fst, const PruneOptions &opts);

// 2
void Prune(const FstClass &ifst, MutableFstClass *fst,
           const PruneOptions &opts);

// 3
void Prune(const FstClass &ifst, MutableFstClass *ofst,
           const WeightClass &weight_threshold,
           int64 state_threshold = kNoStateId,
           float delta = kDelta);

// 4
void Prune(MutableFstClass *fst, const WeightClass &weight_threshold,
           int64 state_threshold, float delta);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_PRUNE_H_
