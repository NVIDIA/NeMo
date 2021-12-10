// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_PUSH_H_
#define FST_SCRIPT_PUSH_H_

#include <fst/push.h>
#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

// 1
using PushArgs1 = args::Package<MutableFstClass *, ReweightType, float, bool>;

template <class Arc>
void Push(PushArgs1 *args) {
  MutableFst<Arc> *ofst = args->arg1->GetMutableFst<Arc>();
  if (args->arg2 == REWEIGHT_TO_FINAL)
    fst::Push(ofst, REWEIGHT_TO_FINAL, args->arg3, args->arg4);
  else
    fst::Push(ofst, REWEIGHT_TO_INITIAL, args->arg3, args->arg4);
}

// 2
using PushArgs2 = args::Package<const FstClass &, MutableFstClass *, uint32,
                                ReweightType, float>;

template <class Arc>
void Push(PushArgs2 *args) {
  const Fst<Arc> &ifst = *(args->arg1.GetFst<Arc>());
  MutableFst<Arc> *ofst = args->arg2->GetMutableFst<Arc>();
  if (args->arg4 == REWEIGHT_TO_FINAL)
    fst::Push<Arc, REWEIGHT_TO_FINAL>(ifst, ofst, args->arg3, args->arg5);
  else
    fst::Push<Arc, REWEIGHT_TO_INITIAL>(ifst, ofst, args->arg3, args->arg5);
}

// 1
void Push(MutableFstClass *ofst, ReweightType type, float delta = kDelta,
          bool remove_total_weight = false);

// 2
void Push(const FstClass &ifst, MutableFstClass *ofst, uint32 flags,
          ReweightType dir, float delta);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_PUSH_H_
