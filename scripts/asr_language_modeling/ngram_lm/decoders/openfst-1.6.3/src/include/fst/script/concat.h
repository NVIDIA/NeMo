// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_CONCAT_H_
#define FST_SCRIPT_CONCAT_H_

#include <fst/concat.h>
#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

using ConcatArgs1 = args::Package<MutableFstClass *, const FstClass &>;

template <class Arc>
void Concat(ConcatArgs1 *args) {
  MutableFst<Arc> *ofst = args->arg1->GetMutableFst<Arc>();
  const Fst<Arc> &ifst = *(args->arg2.GetFst<Arc>());
  Concat(ofst, ifst);
}

using ConcatArgs2 = args::Package<const FstClass &, MutableFstClass *>;

template <class Arc>
void Concat(ConcatArgs2 *args) {
  const Fst<Arc> &ifst = *(args->arg1.GetFst<Arc>());
  MutableFst<Arc> *ofst = args->arg2->GetMutableFst<Arc>();
  Concat(ifst, ofst);
}

void Concat(MutableFstClass *ofst, const FstClass &ifst);

void Concat(const FstClass &ifst, MutableFstClass *ofst);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_CONCAT_H_
