// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_MINIMIZE_H_
#define FST_SCRIPT_MINIMIZE_H_

#include <fst/minimize.h>
#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

using MinimizeArgs =
    args::Package<MutableFstClass *, MutableFstClass *, float, bool>;

template <class Arc>
void Minimize(MinimizeArgs *args) {
  MutableFst<Arc> *ofst1 = args->arg1->GetMutableFst<Arc>();
  MutableFst<Arc> *ofst2 = args->arg2 ?
                           args->arg2->GetMutableFst<Arc>() : nullptr;
  Minimize(ofst1, ofst2, args->arg3, args->arg4);
}

void Minimize(MutableFstClass *ofst1, MutableFstClass *ofst2 = nullptr,
              float delta = kDelta, bool allow_nondet = false);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_MINIMIZE_H_
