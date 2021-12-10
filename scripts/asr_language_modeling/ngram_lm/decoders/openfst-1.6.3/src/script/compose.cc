// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/script/compose.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

// 1
void Compose(const FstClass &ifst1, const FstClass &ifst2,
             MutableFstClass *ofst, ComposeFilter compose_filter) {
  if (!ArcTypesMatch(ifst1, ifst2, "Compose") ||
      !ArcTypesMatch(*ofst, ifst1, "Compose")) {
    ofst->SetProperties(kError, kError);
    return;
  }
  ComposeArgs1 args(ifst1, ifst2, ofst, compose_filter);
  Apply<Operation<ComposeArgs1>>("Compose", ifst1.ArcType(), &args);
}

// 2
void Compose(const FstClass &ifst1, const FstClass &ifst2,
             MutableFstClass *ofst, const ComposeOptions &copts) {
  if (!ArcTypesMatch(ifst1, ifst2, "Compose") ||
      !ArcTypesMatch(*ofst, ifst1, "Compose")) {
    ofst->SetProperties(kError, kError);
    return;
  }
  ComposeArgs2 args(ifst1, ifst2, ofst, copts);
  Apply<Operation<ComposeArgs2>>("Compose", ifst1.ArcType(), &args);
}

REGISTER_FST_OPERATION(Compose, StdArc, ComposeArgs1);
REGISTER_FST_OPERATION(Compose, LogArc, ComposeArgs1);
REGISTER_FST_OPERATION(Compose, Log64Arc, ComposeArgs1);

REGISTER_FST_OPERATION(Compose, StdArc, ComposeArgs2);
REGISTER_FST_OPERATION(Compose, LogArc, ComposeArgs2);
REGISTER_FST_OPERATION(Compose, Log64Arc, ComposeArgs2);

}  // namespace script
}  // namespace fst
