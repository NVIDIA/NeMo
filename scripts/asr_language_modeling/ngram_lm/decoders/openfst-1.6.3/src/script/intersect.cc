// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/script/intersect.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

// 1
void Intersect(const FstClass &ifst1, const FstClass &ifst2,
               MutableFstClass *ofst, ComposeFilter compose_filter) {
  if (!ArcTypesMatch(ifst1, ifst2, "Intersect") ||
      !ArcTypesMatch(*ofst, ifst1, "Intersect")) {
    ofst->SetProperties(kError, kError);
    return;
  }
  IntersectArgs1 args(ifst1, ifst2, ofst, compose_filter);
  Apply<Operation<IntersectArgs1>>("Intersect", ifst1.ArcType(), &args);
}

// 2
void Intersect(const FstClass &ifst1, const FstClass &ifst2,
               MutableFstClass *ofst, const ComposeOptions &copts) {
  if (!ArcTypesMatch(ifst1, ifst2, "Intersect") ||
      !ArcTypesMatch(*ofst, ifst1, "Intersect")) {
    ofst->SetProperties(kError, kError);
    return;
  }
  IntersectArgs2 args(ifst1, ifst2, ofst, copts);
  Apply<Operation<IntersectArgs2>>("Intersect", ifst1.ArcType(), &args);
}

REGISTER_FST_OPERATION(Intersect, StdArc, IntersectArgs1);
REGISTER_FST_OPERATION(Intersect, LogArc, IntersectArgs1);
REGISTER_FST_OPERATION(Intersect, Log64Arc, IntersectArgs1);

REGISTER_FST_OPERATION(Intersect, StdArc, IntersectArgs2);
REGISTER_FST_OPERATION(Intersect, LogArc, IntersectArgs2);
REGISTER_FST_OPERATION(Intersect, Log64Arc, IntersectArgs2);

}  // namespace script
}  // namespace fst
