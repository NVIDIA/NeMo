// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_COMPOSE_H_
#define FST_SCRIPT_COMPOSE_H_

#include <fst/compose.h>
#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

using ComposeArgs1 = args::Package<const FstClass &, const FstClass &,
                                   MutableFstClass *, ComposeFilter>;

template <class Arc>
void Compose(ComposeArgs1 *args) {
  const Fst<Arc> &ifst1 = *(args->arg1.GetFst<Arc>());
  const Fst<Arc> &ifst2 = *(args->arg2.GetFst<Arc>());
  MutableFst<Arc> *ofst = args->arg3->GetMutableFst<Arc>();
  Compose(ifst1, ifst2, ofst, ComposeOptions(args->arg4));
}

using ComposeOptions = fst::ComposeOptions;

using ComposeArgs2 = args::Package<const FstClass &, const FstClass &,
                                   MutableFstClass *, const ComposeOptions &>;

template <class Arc>
void Compose(ComposeArgs2 *args) {
  const Fst<Arc> &ifst1 = *(args->arg1.GetFst<Arc>());
  const Fst<Arc> &ifst2 = *(args->arg2.GetFst<Arc>());
  MutableFst<Arc> *ofst = args->arg3->GetMutableFst<Arc>();
  Compose(ifst1, ifst2, ofst, args->arg4);
}

void Compose(const FstClass &ifst1, const FstClass &ifst2,
             MutableFstClass *ofst,
             const ComposeOptions &opts = fst::script::ComposeOptions());

void Compose(const FstClass &ifst1, const FstClass &ifst2,
             MutableFstClass *ofst, ComposeFilter compose_filter);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_COMPOSE_H_
