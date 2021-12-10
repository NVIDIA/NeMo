// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/script/difference.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

// 1
void Difference(const FstClass &ifst1, const FstClass &ifst2,
                MutableFstClass *ofst, ComposeFilter compose_filter) {
  if (!ArcTypesMatch(ifst1, ifst2, "Difference") ||
      !ArcTypesMatch(*ofst, ifst1, "Difference")) {
    ofst->SetProperties(kError, kError);
    return;
  }
  DifferenceArgs1 args(ifst1, ifst2, ofst, compose_filter);
  Apply<Operation<DifferenceArgs1>>("Difference", ifst1.ArcType(), &args);
}

// 2
void Difference(const FstClass &ifst1, const FstClass &ifst2,
                MutableFstClass *ofst, const ComposeOptions &copts) {
  if (!ArcTypesMatch(ifst1, ifst2, "Difference") ||
      !ArcTypesMatch(*ofst, ifst1, "Difference")) {
    ofst->SetProperties(kError, kError);
    return;
  }
  DifferenceArgs2 args(ifst1, ifst2, ofst, copts);
  Apply<Operation<DifferenceArgs2>>("Difference", ifst1.ArcType(), &args);
}

REGISTER_FST_OPERATION(Difference, StdArc, DifferenceArgs1);
REGISTER_FST_OPERATION(Difference, LogArc, DifferenceArgs1);
REGISTER_FST_OPERATION(Difference, Log64Arc, DifferenceArgs1);
REGISTER_FST_OPERATION(Difference, StdArc, DifferenceArgs2);
REGISTER_FST_OPERATION(Difference, LogArc, DifferenceArgs2);
REGISTER_FST_OPERATION(Difference, Log64Arc, DifferenceArgs2);

}  // namespace script
}  // namespace fst
