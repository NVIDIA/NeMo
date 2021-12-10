// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_ISOMORPHIC_H_
#define FST_SCRIPT_ISOMORPHIC_H_

#include <fst/isomorphic.h>
#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

using IsomorphicInnerArgs =
    args::Package<const FstClass &, const FstClass &, float>;

using IsomorphicArgs = args::WithReturnValue<bool, IsomorphicInnerArgs>;

template <class Arc>
void Isomorphic(IsomorphicArgs *args) {
  const Fst<Arc> &fst1 = *(args->args.arg1.GetFst<Arc>());
  const Fst<Arc> &fst2 = *(args->args.arg2.GetFst<Arc>());
  args->retval = Isomorphic(fst1, fst2, args->args.arg3);
}

bool Isomorphic(const FstClass &fst1, const FstClass &fst2,
                float delta = kDelta);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_ISOMORPHIC_H_
