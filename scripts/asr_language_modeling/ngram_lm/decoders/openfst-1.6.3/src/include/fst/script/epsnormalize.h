// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_EPSNORMALIZE_H_
#define FST_SCRIPT_EPSNORMALIZE_H_

#include <fst/epsnormalize.h>
#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

using EpsNormalizeArgs =
    args::Package<const FstClass &, MutableFstClass *, EpsNormalizeType>;

template <class Arc>
void EpsNormalize(EpsNormalizeArgs *args) {
  const Fst<Arc> &ifst = *(args->arg1.GetFst<Arc>());
  MutableFst<Arc> *ofst = args->arg2->GetMutableFst<Arc>();
  EpsNormalize(ifst, ofst, args->arg3);
}

void EpsNormalize(const FstClass &ifst, MutableFstClass *ofst,
                  EpsNormalizeType norm_type = EPS_NORM_INPUT);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_EPSNORMALIZE_H_
