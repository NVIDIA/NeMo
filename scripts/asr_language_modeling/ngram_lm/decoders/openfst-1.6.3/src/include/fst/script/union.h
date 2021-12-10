// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_UNION_H_
#define FST_SCRIPT_UNION_H_

#include <fst/union.h>
#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

using UnionArgs = args::Package<MutableFstClass *, const FstClass &>;

template <class Arc>
void Union(UnionArgs *args) {
  MutableFst<Arc> *fst1 = args->arg1->GetMutableFst<Arc>();
  const Fst<Arc> &fst2 = *(args->arg2.GetFst<Arc>());
  Union(fst1, fst2);
}

void Union(MutableFstClass *fst1, const FstClass &fst2);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_UNION_H_
