// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_CLOSURE_H_
#define FST_SCRIPT_CLOSURE_H_

#include <fst/closure.h>
#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

using ClosureArgs = args::Package<MutableFstClass *, const ClosureType>;

template <class Arc>
void Closure(ClosureArgs *args) {
  MutableFst<Arc> *fst = args->arg1->GetMutableFst<Arc>();
  Closure(fst, args->arg2);
}

void Closure(MutableFstClass *ofst, ClosureType closure_type);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_CLOSURE_H_
