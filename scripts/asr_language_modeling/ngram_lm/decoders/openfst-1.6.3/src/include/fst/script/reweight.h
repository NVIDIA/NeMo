// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_REWEIGHT_H_
#define FST_SCRIPT_REWEIGHT_H_

#include <vector>

#include <fst/reweight.h>
#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>
#include <fst/script/weight-class.h>

namespace fst {
namespace script {

using ReweightArgs =
    args::Package<MutableFstClass *, const std::vector<WeightClass> &,
                  ReweightType>;

template <class Arc>
void Reweight(ReweightArgs *args) {
  MutableFst<Arc> *fst = args->arg1->GetMutableFst<Arc>();
  using Weight = typename Arc::Weight;
  std::vector<Weight> potentials(args->arg2.size());
  for (auto i = 0; i < args->arg2.size(); ++i)
    potentials[i] = *(args->arg2[i].GetWeight<Weight>());
  Reweight(fst, potentials, args->arg3);
}

void Reweight(MutableFstClass *fst, const std::vector<WeightClass> &potential,
              ReweightType reweight_type);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_REWEIGHT_H_
