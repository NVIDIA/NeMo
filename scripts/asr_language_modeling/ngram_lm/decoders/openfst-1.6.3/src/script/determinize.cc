// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/script/determinize.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

// 1: Full signature with DeterminizeOptions.
void Determinize(const FstClass &ifst, MutableFstClass *ofst,
                 const DeterminizeOptions &opts) {
  if (!ArcTypesMatch(ifst, *ofst, "Determinize") ||
      !ofst->WeightTypesMatch(opts.weight_threshold, "Determinize")) {
    ofst->SetProperties(kError, kError);
    return;
  }
  DeterminizeArgs1 args(ifst, ofst, opts);
  Apply<Operation<DeterminizeArgs1>>("Determinize", ifst.ArcType(), &args);
}

// 2: Signature with default WeightClass argument.
void Determinize(const FstClass &ifst, MutableFstClass *ofst, float d,
                 int64 n, int64 l, DeterminizeType t, bool i) {
  if (!ArcTypesMatch(ifst, *ofst, "Determinize")) {
    ofst->SetProperties(kError, kError);
    return;
  }
  DeterminizeArgs2 args(ifst, ofst, d, n, l, t, i);
  Apply<Operation<DeterminizeArgs2>>("Determinize", ifst.ArcType(), &args);
}

REGISTER_FST_OPERATION(Determinize, StdArc, DeterminizeArgs1);
REGISTER_FST_OPERATION(Determinize, LogArc, DeterminizeArgs1);
REGISTER_FST_OPERATION(Determinize, Log64Arc, DeterminizeArgs1);

REGISTER_FST_OPERATION(Determinize, StdArc, DeterminizeArgs2);
REGISTER_FST_OPERATION(Determinize, LogArc, DeterminizeArgs2);
REGISTER_FST_OPERATION(Determinize, Log64Arc, DeterminizeArgs2);

}  // namespace script
}  // namespace fst
