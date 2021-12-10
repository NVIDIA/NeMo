// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/script/prune.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

// 1
void Prune(MutableFstClass *fst, const PruneOptions &opts) {
  if (!fst->WeightTypesMatch(opts.weight_threshold, "Prune")) {
    fst->SetProperties(kError, kError);
    return;
  }
  PruneArgs1 args(fst, opts);
  Apply<Operation<PruneArgs1>>("Prune", fst->ArcType(), &args);
}

// 2
void Prune(const FstClass &ifst, MutableFstClass *ofst,
           const PruneOptions &opts) {
  if (!ArcTypesMatch(ifst, *ofst, "Prune") ||
      !ofst->WeightTypesMatch(opts.weight_threshold, "Prune")) {
    ofst->SetProperties(kError, kError);
    return;
  }
  PruneArgs2 args(ifst, ofst, opts);
  Apply<Operation<PruneArgs2>>("Prune", ofst->ArcType(), &args);
}

// 3
void Prune(const FstClass &ifst, MutableFstClass *ofst,
           const WeightClass &weight_threshold,
           int64 state_threshold, float delta) {
  if (!ArcTypesMatch(ifst, *ofst, "Prune") ||
      !ofst->WeightTypesMatch(weight_threshold, "Prune")) {
    ofst->SetProperties(kError, kError);
    return;
  }
  PruneArgs3 args(ifst, ofst, weight_threshold, state_threshold, delta);
  Apply<Operation<PruneArgs3>>("Prune", ifst.ArcType(), &args);
}

// 4
void Prune(MutableFstClass *fst, const WeightClass &weight_threshold,
           int64 state_threshold, float delta) {
  if (!fst->WeightTypesMatch(weight_threshold, "Prune")) {
    fst->SetProperties(kError, kError);
    return;
  }
  PruneArgs4 args(fst, weight_threshold, state_threshold, delta);
  Apply<Operation<PruneArgs4>>("Prune", fst->ArcType(), &args);
}

// 1
REGISTER_FST_OPERATION(Prune, StdArc, PruneArgs1);
REGISTER_FST_OPERATION(Prune, LogArc, PruneArgs1);
REGISTER_FST_OPERATION(Prune, Log64Arc, PruneArgs1);

// 2
REGISTER_FST_OPERATION(Prune, StdArc, PruneArgs2);
REGISTER_FST_OPERATION(Prune, LogArc, PruneArgs2);
REGISTER_FST_OPERATION(Prune, Log64Arc, PruneArgs2);

// 3
REGISTER_FST_OPERATION(Prune, StdArc, PruneArgs3);
REGISTER_FST_OPERATION(Prune, LogArc, PruneArgs3);
REGISTER_FST_OPERATION(Prune, Log64Arc, PruneArgs3);

// 4
REGISTER_FST_OPERATION(Prune, StdArc, PruneArgs4);
REGISTER_FST_OPERATION(Prune, LogArc, PruneArgs4);
REGISTER_FST_OPERATION(Prune, Log64Arc, PruneArgs4);

}  // namespace script
}  // namespace fst
