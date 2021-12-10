// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_SYNCHRONIZE_H_
#define FST_SCRIPT_SYNCHRONIZE_H_

#include <fst/synchronize.h>
#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

using SynchronizeArgs = args::Package<const FstClass &, MutableFstClass *>;

template <class Arc>
void Synchronize(SynchronizeArgs *args) {
  const Fst<Arc> &ifst = *(args->arg1.GetFst<Arc>());
  MutableFst<Arc> *ofst = args->arg2->GetMutableFst<Arc>();
  Synchronize(ifst, ofst);
}

void Synchronize(const FstClass &ifst, MutableFstClass *ofst);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_SYNCHRONIZE_H_
