// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_REVERSE_H_
#define FST_SCRIPT_REVERSE_H_

#include <fst/reverse.h>
#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

using ReverseArgs = args::Package<const FstClass &, MutableFstClass *, bool>;

template <class Arc>
void Reverse(ReverseArgs *args) {
  const Fst<Arc> &ifst = *(args->arg1.GetFst<Arc>());
  MutableFst<Arc> *ofst = args->arg2->GetMutableFst<Arc>();
  bool require_superinitial = args->arg3;
  Reverse(ifst, ofst, require_superinitial);
}

void Reverse(const FstClass &ifst, MutableFstClass *ofst,
             bool require_superinitial = true);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_REVERSE_H_
