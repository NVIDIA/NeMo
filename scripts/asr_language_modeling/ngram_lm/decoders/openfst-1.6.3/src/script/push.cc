// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/script/push.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

// 1
void Push(MutableFstClass *ofst, ReweightType dir, float delta,
          bool remove_total_weight) {
  PushArgs1 args(ofst, dir, delta, remove_total_weight);
  Apply<Operation<PushArgs1>>("Push", ofst->ArcType(), &args);
}

// 2
void Push(const FstClass &ifst, MutableFstClass *ofst, uint32 flags,
          ReweightType dir, float delta) {
  if (!ArcTypesMatch(ifst, *ofst, "Push")) {
    ofst->SetProperties(kError, kError);
    return;
  }
  PushArgs2 args(ifst, ofst, flags, dir, delta);
  Apply<Operation<PushArgs2>>("Push", ifst.ArcType(), &args);
}

REGISTER_FST_OPERATION(Push, StdArc, PushArgs1);
REGISTER_FST_OPERATION(Push, LogArc, PushArgs1);
REGISTER_FST_OPERATION(Push, Log64Arc, PushArgs1);

REGISTER_FST_OPERATION(Push, StdArc, PushArgs2);
REGISTER_FST_OPERATION(Push, LogArc, PushArgs2);
REGISTER_FST_OPERATION(Push, Log64Arc, PushArgs2);

}  // namespace script
}  // namespace fst
